"""
Test interactif d'intrusion — SysGuard-AI Two-Tier (Tier 1 + Tier 2)

Simule des attaques fenetres par fenetres (comme Falco le ferait en live).
Tier 1 (Autoencoder) : detection en < 1 ms.
Tier 2 (LLM via OpenRouter) : arbitrage automatique sur chaque anomalie detectee.

Pas besoin de Kubernetes ni de Falco : le modele est charge directement.
La cle API est lue depuis .env (ou passee en variable OPENROUTER_API_KEY).

Usage :
  cd ai_model
  python test_intrusion.py           # Two-Tier complet (Tier 1 + Tier 2)
  python test_intrusion.py --no-llm  # Tier 1 seulement
"""

import io
import json
import os
import sys
import time
import threading
import numpy as np
import torch
import httpx

# Force UTF-8 sur Windows pour les couleurs ANSI et les caracteres speciaux
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from autoencoder import SysGuardAutoencoder, INPUT_DIM

from syscall_mapping import REVERSE_SYSCALL_TABLE, SC

try:
    from _secrets import load_secrets
    _cfg = load_secrets()
    LLM_KEY = _cfg.get("OPENROUTER_API_KEY", "")
    LLM_MODEL = _cfg.get("LLM_MODEL", "deepseek/deepseek-v4-flash:free")
except Exception:
    LLM_KEY = ""
    LLM_MODEL = "deepseek/deepseek-v4-flash:free"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# Couleurs ANSI
# ---------------------------------------------------------------------------
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(SCRIPT_DIR, "saved_model.pth")
THRESHOLD_PATH = os.path.join(SCRIPT_DIR, "threshold.json")

# Indices syscall : même table que agent/preprocessor (voir syscall_mapping.py)


# ---------------------------------------------------------------------------
# Chargement modele + seuils
# ---------------------------------------------------------------------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"{RED}Modele introuvable : {MODEL_PATH}{RESET}")
        print("Lancez d'abord : python train_autoencoder.py")
        sys.exit(1)
    model = SysGuardAutoencoder(INPUT_DIM, dropout=0.0)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_thresholds():
    if not os.path.exists(THRESHOLD_PATH):
        print(f"{YELLOW}threshold.json introuvable — utilisation de valeurs par defaut{RESET}")
        return {"alpha": 0.05, "beta": 0.125, "rare_dims": {}}
    with open(THRESHOLD_PATH) as f:
        data = json.load(f)
    rare_dims = data.get("rare_dims", {})
    n_rare = len(rare_dims.get("rare_dim_indices", []))
    print(f"  {DIM}[Threshold] alpha={data['alpha']:.2e}  beta={data['beta']:.2e}  "
          f"rare_dims={n_rare}{RESET}")
    return {"alpha": data["alpha"], "beta": data["beta"], "rare_dims": rare_dims}


# ---------------------------------------------------------------------------
# Inference sur un vecteur
# ---------------------------------------------------------------------------

def _check_rare_syscalls(vector: np.ndarray, rare_dims: dict) -> bool:
    """
    Retourne True si au moins une dimension rare dépasse le seuil d'occurrences.
    Permet la détection d'attaques furtives même quand MSE < alpha.
    """
    indices = rare_dims.get("rare_dim_indices", [])
    threshold = rare_dims.get("count_threshold", 5)
    if not indices:
        return False
    SCALE = 1000.0
    for idx in indices:
        if idx < len(vector) and vector[idx] * SCALE >= threshold:
            return True
    return False


def infer(model, vector: np.ndarray, alpha: float, beta: float, rare_dims: dict = None):
    v = vector / (vector.sum() + 1e-9)
    t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        recon = model(t)
    mse = float(torch.mean((t - recon) ** 2).item())

    rare_alert = _check_rare_syscalls(v, rare_dims or {})

    if mse >= beta:
        verdict = "CRITIQUE"
        color = RED
    elif mse >= alpha:
        verdict = "ANOMALIE"
        color = YELLOW
    elif rare_alert:
        verdict = "FURTIVE→T2"
        color = CYAN
    else:
        verdict = "NORMAL"
        color = GREEN

    top_idx = np.argsort(vector)[::-1][:5]
    rev_sc = REVERSE_SYSCALL_TABLE
    top = [(rev_sc.get(i, f"syscall_{i}"), int(vector[i])) for i in top_idx if vector[i] > 0]
    return mse, verdict, color, top


# ---------------------------------------------------------------------------
# Generateurs de vecteurs
# ---------------------------------------------------------------------------

def _base_web(rng):
    """Trafic HTTP normal (profil DVWA)."""
    v = np.zeros(INPUT_DIM, dtype=np.float32)
    prof = {
        "read": .13, "write": .12, "close": .11, "epoll_wait": .13,
        "recvfrom": .07, "sendto": .07, "futex": .06, "openat": .03,
        "stat": .03, "fstat": .02, "poll": .02, "clock_gettime": .015,
        "socket": .005, "accept": .005, "mmap": .003, "brk": .002,
    }
    total = rng.integers(800, 1600)
    for name, ratio in prof.items():
        v[SC[name]] = rng.poisson(int(total * ratio))
    return v


def gen_normal(rng, n=1):
    return [_base_web(rng) for _ in range(n)]


def gen_reverse_shell(rng, n=8, intensity_ramp=True):
    """
    Reverse shell : les premiers vecteurs ressemblent au trafic normal,
    puis l'intensite monte (simule l'etablissement de la connexion).
    """
    vectors = []
    for i in range(n):
        v = _base_web(rng)
        ratio = (i + 1) / n if intensity_ramp else 1.0
        scale = rng.uniform(3.0, 7.0) * ratio
        v[SC["execve"]]  += int(rng.poisson(500 * scale))
        v[SC["connect"]] += int(rng.poisson(300 * scale))
        v[SC["dup2"]]    += int(rng.poisson(200 * scale))
        v[SC["socket"]]  += int(rng.poisson(150 * scale))
        v[SC["read"]]    += int(rng.poisson(100 * scale))
        v[SC["write"]]   += int(rng.poisson(100 * scale))
        vectors.append(v)
    return vectors


def gen_crypto_mining(rng, n=8):
    """
    Crypto-mining : explosion de clone/fork + sched_yield (threads mineurs).
    """
    vectors = []
    for i in range(n):
        v = _base_web(rng)
        threads = rng.integers(4, 16)
        scale = rng.uniform(2.0, 6.0)
        v[SC["clone"]]      += int(rng.poisson(400 * scale * threads))
        v[SC["fork"]]       += int(rng.poisson(200 * scale))
        v[SC["sched_yield"]] += int(rng.poisson(600 * scale * threads))
        v[SC["futex"]]      += int(rng.poisson(300 * scale * threads))
        v[SC["mmap"]]       += int(rng.poisson(100 * scale))
        v[SC["socket"]]     += int(rng.poisson(50 * scale))
        v[SC["connect"]]    += int(rng.poisson(40 * scale))
        vectors.append(v)
    return vectors


def gen_privilege_escalation(rng, n=8):
    """
    Escalade de privileges : ptrace, setuid, keyctl, capset —
    syscalls jamais vus dans le trafic normal DVWA.
    """
    vectors = []
    for i in range(n):
        v = _base_web(rng)
        scale = rng.uniform(2.0, 5.0)
        v[SC["ptrace"]]           += int(rng.poisson(300 * scale))
        v[SC["process_vm_readv"]] += int(rng.poisson(150 * scale))
        v[SC["keyctl"]]           += int(rng.poisson(100 * scale))
        v[SC["setuid"]]           += int(rng.poisson(80 * scale))
        v[SC["setgid"]]           += int(rng.poisson(80 * scale))
        v[SC["capset"]]           += int(rng.poisson(60 * scale))
        v[SC["chown"]]            += int(rng.poisson(50 * scale))
        v[SC["openat"]]           += int(rng.poisson(200 * scale))
        vectors.append(v)
    return vectors


def gen_stealth_attack(rng, n=8):
    """
    Attaque furtive améliorée : faible intensité MSE + syscalls rares.

    Version 1 (MSE seule) : execve/connect/dup2 dilués → non détectés par Tier 1.
    Version 2 (rare_dims) : ptrace/keyctl/setuid injectés en faible quantité →
      MSE reste sous α MAIS rare_dims déclenche le Tier 2 (nouvelle logique).
    """
    vectors = []
    for i in range(n):
        v = _base_web(rng)
        scale = rng.uniform(0.3, 0.8)
        # Composante à signal faible (inchangée)
        v[SC["execve"]]  += int(rng.poisson(80 * scale))
        v[SC["connect"]] += int(rng.poisson(40 * scale))
        v[SC["dup2"]]    += int(rng.poisson(30 * scale))
        # Composante rare : ces syscalls apparaissent quasiment jamais en trafic web normal
        # → rare_dims les repère même si MSE globale reste faible
        v[SC["ptrace"]]  += int(rng.poisson(8))
        v[SC["keyctl"]]  += int(rng.poisson(6))
        v[SC["setuid"]]  += int(rng.poisson(5))
        vectors.append(v)
    return vectors


def gen_blended_reverse_shell_vectors(rng, n=10):
    """Reverse shell dilue dans un nominal volumineux (voir evaluate_model)."""
    from evaluate_model import build_blended_reverse_shell_counts

    arr = build_blended_reverse_shell_counts(rng, n)
    return [np.asarray(arr[i], dtype=np.float32).copy() for i in range(n)]


def gen_diluted_mining_vectors(rng, n=10):
    from evaluate_model import build_diluted_mining_counts

    arr = build_diluted_mining_counts(rng, n)
    return [np.asarray(arr[i], dtype=np.float32).copy() for i in range(n)]


def gen_web_overlap_privesc_vectors(rng, n=10):
    from evaluate_model import build_web_overlap_privesc_counts

    arr = build_web_overlap_privesc_counts(rng, n)
    return [np.asarray(arr[i], dtype=np.float32).copy() for i in range(n)]


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

BANNER = (
    f"\n{CYAN}{BOLD}"
    "  +==================================================+\n"
    "  |    SysGuard-AI  --  Test d'Intrusion Live       |\n"
    "  |  Tier 1 (Autoencoder) + Tier 2 (LLM arbitre)   |\n"
    "  +==================================================+"
    f"{RESET}"
)

MENU = f"""
{BOLD}Choisissez un scenario d'attaque :{RESET}

  {CYAN}1{RESET}  Trafic normal            (baseline, tout doit etre NORMAL)
  {RED}2{RESET}  Reverse Shell            (execve + connect + dup2)
  {RED}3{RESET}  Crypto-mining            (clone x N + sched_yield massif)
  {RED}4{RESET}  Escalade de privileges   (ptrace + setuid + capset)
  {YELLOW}5{RESET}  Attaque furtive          (faible intensite — teste les limites)
  {BOLD}6{RESET}  Sequence complete        (normal → attaque → normal)
  {DIM}7{RESET}  RS melange nominal       (attaque diluee — stress realiste)
  {DIM}8{RESET}  Mining dilue               (threads + DVWA encore visible)
  {DIM}9{RESET}  Privesc + nominal fort    (histogramme encore \"web-like\")
  {DIM}q{RESET}  Quitter
"""


def bar(mse, alpha, beta, width=40):
    """Barre de progression MSE / seuils."""
    ref = beta * 1.5
    pos = min(int(mse / ref * width), width)
    a_pos = min(int(alpha / ref * width), width)
    b_pos = min(int(beta / ref * width), width)

    bar_chars = list("-" * width)
    for i in range(pos):
        bar_chars[i] = "█"
    if 0 <= a_pos < width:
        bar_chars[a_pos] = "α"
    if 0 <= b_pos < width:
        bar_chars[b_pos] = "β"

    if mse >= beta:
        color = RED
    elif mse >= alpha:
        color = YELLOW
    else:
        color = GREEN

    return f"{color}[{''.join(bar_chars)}]{RESET}"


# ---------------------------------------------------------------------------
# Tier 2 — Appel LLM asynchrone
# ---------------------------------------------------------------------------

def _build_llm_prompt(mse, severity, top_syscalls, scenario_label):
    syscall_str = ", ".join(f"{n}({c})" for n, c in top_syscalls[:10])
    return (
        "Tu es un expert en securite Kubernetes (DFIR).\n"
        "L'Autoencoder SysGuard-AI (Tier 1) a detecte une anomalie.\n"
        "Reponds UNIQUEMENT en JSON strict (sans texte autour).\n\n"
        f"Contexte : score MSE={mse:.6f}, severite={severity}, "
        f"scenario={scenario_label}\n"
        f"Top syscalls : {syscall_str}\n\n"
        "Format :\n"
        '{"verdict":"MENACE" ou "FAUX_POSITIF","confidence":0.0-1.0,'
        '"attack_type":"...","severity_cvss":"CRITIQUE|ELEVEE|MOYENNE|FAIBLE",'
        '"quarantine":"MAINTENIR|LEVER","explanation":"..."}'
    )


def call_tier2(mse, severity, top_syscalls, scenario_label, result_box):
    """Appel LLM en thread background — ecrit dans result_box[0]."""
    if not LLM_KEY:
        result_box[0] = {"error": "no_key"}
        return
    prompt = _build_llm_prompt(mse, severity, top_syscalls, scenario_label)
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://sysguard-ai.local",
        "X-Title":       "SysGuard-AI Intrusion Test",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system",
             "content": "Tu es un expert DFIR Kubernetes. Reponds uniquement en JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens":  200,
    }
    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        content = (resp.json().get("choices") or [{}])[0] \
                      .get("message", {}).get("content") or ""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        result_box[0] = json.loads(content) if content else {"error": "empty"}
    except Exception as e:
        result_box[0] = {"error": str(e)}


def print_tier2_result(result, latency_s):
    """Affiche le verdict Tier 2 apres reception."""
    if not result or "error" in result:
        err = result.get("error", "?") if result else "?"
        if err == "no_key":
            print(f"  {DIM}Tier 2 : cle API absente — passer en mode --no-llm{RESET}")
        else:
            print(f"  {DIM}Tier 2 : {err}{RESET}")
        return

    v   = result.get("verdict", "?")
    c   = result.get("confidence", 0)
    q   = result.get("quarantine", "?")
    att = result.get("attack_type", "")
    sev = result.get("severity_cvss", "")
    exp = result.get("explanation", "")

    v_color = RED if v == "MENACE" else GREEN
    q_color = RED if q == "MAINTENIR" else GREEN

    print(f"\n  {BOLD}{'─'*56}{RESET}")
    print(f"  {CYAN}{BOLD}[TIER 2 — LLM]{RESET}  latence : {latency_s*1000:.0f} ms")
    print(f"  Verdict    : {v_color}{BOLD}{v}{RESET}  (confiance : {c:.0%})")
    if att:
        print(f"  Attaque    : {att}  [{sev}]")
    print(f"  Quarantaine: {q_color}{BOLD}{q}{RESET}")
    if exp:
        # Tronquer a 80 chars pour l'affichage terminal
        print(f"  Analyse    : {DIM}{exp[:100]}{'...' if len(exp)>100 else ''}{RESET}")
    print(f"  {BOLD}{'─'*56}{RESET}")


def print_window(idx, total, vector, mse, verdict, color, top, alpha, beta,
                 use_llm=True, scenario_label="", delay=0.04):
    """Affiche une fenetre d'analyse. Lance le Tier 2 si anomalie detectee."""
    top_str = "  ".join(f"{name}:{count}" for name, count in top[:5])
    b = bar(mse, alpha, beta)

    print(f"\n  {DIM}Fenetre {idx+1:2d}/{total}{RESET}  {b}  MSE={mse:.2e}")
    print(f"  Top syscalls  : {DIM}{top_str}{RESET}")
    print(f"  Verdict       : {color}{BOLD}{verdict}{RESET}", end="")

    if verdict == "CRITIQUE":
        print(f"  {RED}*** QUARANTAINE IMMEDIATE ***{RESET}")
    elif verdict == "ANOMALIE":
        print(f"  {YELLOW}[→ Tier 2 en cours...]{RESET}")
    else:
        print()

    # Lancer Tier 2 si anomalie et LLM actif
    if use_llm and verdict in ("ANOMALIE", "CRITIQUE") and LLM_KEY:
        result_box = [None]
        t0 = time.perf_counter()
        thread = threading.Thread(
            target=call_tier2,
            args=(mse, verdict, top, scenario_label, result_box),
            daemon=True,
        )
        thread.start()
        # Afficher une animation d'attente pendant que le LLM reflechit
        spin = ["|", "/", "-", "\\"]
        i = 0
        while thread.is_alive():
            print(f"\r  {CYAN}Tier 2 LLM {spin[i % 4]}{RESET}   ", end="", flush=True)
            i += 1
            time.sleep(0.2)
        thread.join()
        print("\r" + " " * 30 + "\r", end="")  # effacer la ligne spinner
        latency = time.perf_counter() - t0
        print_tier2_result(result_box[0], latency)
    else:
        time.sleep(delay)


def run_scenario_data(model, alpha, beta, vectors, label, use_llm=True, rare_dims=None):
    """Execute un scenario sans I/O terminal — retourne un dict structure pour le dashboard."""
    windows = []
    detected = 0
    critical = 0
    is_normal = "normal" in label.lower()
    _rare = rare_dims or {}

    for i, v in enumerate(vectors):
        mse, verdict, _, top = infer(model, v, alpha, beta, rare_dims=_rare)
        if verdict != "NORMAL":
            detected += 1
        if verdict == "CRITIQUE":
            critical += 1

        tier2 = None
        tier2_latency_ms = None
        if use_llm and verdict in ("ANOMALIE", "CRITIQUE", "FURTIVE→T2") and LLM_KEY:
            result_box = [None]
            t0 = time.perf_counter()
            call_tier2(mse, verdict, top, label, result_box)
            tier2_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            tier2 = result_box[0]

        windows.append({
            "index": i + 1,
            "mse": mse,
            "verdict": verdict,
            "top_syscalls": [{"name": n, "count": c} for n, c in top],
            "tier2": tier2,
            "tier2_latency_ms": tier2_latency_ms,
        })

    summary = {
        "total": len(vectors),
        "detected": detected,
        "critical": critical,
        "false_positives": detected if is_normal else 0,
        "detection_rate": None if is_normal else round(detected / max(len(vectors), 1), 3),
    }
    return {
        "label": label,
        "windows": windows,
        "summary": summary,
        "alpha": alpha,
        "beta": beta,
        "tier2_enabled": bool(use_llm and LLM_KEY),
    }


SCENARIO_CATALOG = {
    "normal": {
        "label": "Trafic normal",
        "description": "Baseline DVWA — aucune alerte attendue.",
        "generator": lambda rng, n: gen_normal(rng, n=n),
        "use_llm": False,
    },
    "reverse_shell": {
        "label": "Reverse Shell",
        "description": "execve + connect + dup2 — shell distant.",
        "generator": lambda rng, n: gen_reverse_shell(rng, n=n),
        "use_llm": True,
    },
    "crypto_mining": {
        "label": "Crypto-mining",
        "description": "Explosion clone/fork + sched_yield.",
        "generator": lambda rng, n: gen_crypto_mining(rng, n=n),
        "use_llm": True,
    },
    "privilege_escalation": {
        "label": "Escalade de privileges",
        "description": "ptrace, setuid, capset — jamais vus en trafic normal.",
        "generator": lambda rng, n: gen_privilege_escalation(rng, n=n),
        "use_llm": True,
    },
    "stealth": {
        "label": "Attaque furtive",
        "description": "Faible intensite — teste les limites du Tier 1.",
        "generator": lambda rng, n: gen_stealth_attack(rng, n=n),
        "use_llm": True,
    },
    "blended_rs": {
        "label": "RS melange nominal",
        "description": "Reverse shell sous-dominant dans une fenetre DVWA encore bruyante (stress realiste).",
        "generator": lambda rng, n: gen_blended_reverse_shell_vectors(rng, n=n),
        "use_llm": True,
    },
    "diluted_mining": {
        "label": "Mining dilue",
        "description": "Threads mineurs + HTTP toujours present — rapport signal/bruit modeste.",
        "generator": lambda rng, n: gen_diluted_mining_vectors(rng, n=n),
        "use_llm": True,
    },
    "web_overlap_privesc": {
        "label": "Privesc + nominal fort",
        "description": "ptrace/keyctl injetes sans effondrer le bloc read/epoll/recv.",
        "generator": lambda rng, n: gen_web_overlap_privesc_vectors(rng, n=n),
        "use_llm": True,
    },
}


def run_scenario(model, alpha, beta, vectors, label, rng, use_llm=True, rare_dims=None):
    total = len(vectors)
    detected = 0
    critical = 0
    stealthy = 0
    _rare = rare_dims or {}

    tier2_tag = f"{CYAN}[Tier 1 + Tier 2]{RESET}" if (use_llm and LLM_KEY) \
                else f"{DIM}[Tier 1 seulement]{RESET}"
    print(f"\n{BOLD}  [ {label} ] — {total} fenetres de 10s  {tier2_tag}{RESET}")
    print(f"  Seuils : alpha={alpha:.2e} (P99)   beta={beta:.2e} (P99.9)  "
          f"rare_dims={len(_rare.get('rare_dim_indices', []))}")
    print("  " + "─" * 60)

    for i, v in enumerate(vectors):
        mse, verdict, color, top = infer(model, v, alpha, beta, rare_dims=_rare)
        if verdict != "NORMAL":
            detected += 1
        if verdict == "CRITIQUE":
            critical += 1
        if verdict == "FURTIVE→T2":
            stealthy += 1
        print_window(i, total, v, mse, verdict, color, top, alpha, beta,
                     use_llm=use_llm, scenario_label=label)

    print("\n  " + "─" * 60)
    print(f"  {BOLD}Bilan :{RESET}")
    print(f"    Fenetres analysees : {total}")
    if "normal" in label.lower():
        fp = detected
        print(f"    Faux positifs      : {YELLOW if fp > 0 else GREEN}{fp}{RESET} / {total}")
    else:
        print(f"    Detectees          : {GREEN if detected > 0 else RED}{detected}{RESET} / {total}")
        print(f"    Critiques (beta)   : {RED if critical > 0 else DIM}{critical}{RESET} / {total}")
        if stealthy > 0:
            print(f"    Furtifs→Tier2      : {CYAN}{stealthy}{RESET} / {total}")
        pct = detected / total * 100
        color_pct = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
        print(f"    Taux de detection  : {color_pct}{BOLD}{pct:.1f}%{RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true",
                        help="Desactive le Tier 2 (Tier 1 seulement)")
    args = parser.parse_args()
    use_llm = not args.no_llm

    print(BANNER)

    # Statut LLM
    if use_llm and LLM_KEY:
        print(f"  {CYAN}Mode Two-Tier : Tier 1 (Autoencoder) + Tier 2 (LLM {LLM_MODEL}){RESET}")
    elif use_llm and not LLM_KEY:
        print(f"  {YELLOW}Cle LLM absente — mode Tier 1 seulement.{RESET}")
        print(f"  {DIM}(Ajouter OPENROUTER_API_KEY dans .env pour activer le Tier 2){RESET}")
        use_llm = False
    else:
        print(f"  {DIM}Mode Tier 1 seulement (--no-llm).{RESET}")

    print(f"  Chargement du modele depuis : {MODEL_PATH}")
    model = load_model()
    thresholds = load_thresholds()
    alpha = thresholds["alpha"]
    beta = thresholds["beta"]
    rare_dims = thresholds.get("rare_dims", {})
    n_rare = len(rare_dims.get("rare_dim_indices", []))
    print(f"  Modele charge. alpha={alpha:.2e}  beta={beta:.2e}  rare_dims={n_rare}")

    rng = np.random.default_rng(42)

    while True:
        print(MENU)
        choice = input(f"  {BOLD}Votre choix :{RESET} ").strip().lower()

        if choice == "q":
            print(f"\n{DIM}Au revoir.{RESET}\n")
            break

        elif choice == "1":
            run_scenario(model, alpha, beta, gen_normal(rng, n=10),
                         "Trafic Normal", rng, use_llm=False, rare_dims=rare_dims)

        elif choice == "2":
            run_scenario(model, alpha, beta, gen_reverse_shell(rng, n=10),
                         "Reverse Shell", rng, use_llm=use_llm, rare_dims=rare_dims)

        elif choice == "3":
            run_scenario(model, alpha, beta, gen_crypto_mining(rng, n=10),
                         "Crypto-Mining", rng, use_llm=use_llm, rare_dims=rare_dims)

        elif choice == "4":
            run_scenario(model, alpha, beta, gen_privilege_escalation(rng, n=10),
                         "Escalade de Privileges", rng, use_llm=use_llm, rare_dims=rare_dims)

        elif choice == "5":
            run_scenario(model, alpha, beta, gen_stealth_attack(rng, n=10),
                         "Attaque Furtive (stealth)", rng, use_llm=use_llm, rare_dims=rare_dims)

        elif choice == "6":
            print(f"\n{BOLD}  Sequence complete : normal (5) → attaque (8) → normal (5){RESET}")
            vectors_seq = (
                gen_normal(rng, n=5) +
                gen_reverse_shell(rng, n=8, intensity_ramp=True) +
                gen_normal(rng, n=5)
            )
            run_scenario(model, alpha, beta, vectors_seq,
                         "Sequence Complete", rng, use_llm=use_llm, rare_dims=rare_dims)

        elif choice == "7":
            run_scenario(model, alpha, beta, gen_blended_reverse_shell_vectors(rng, n=10),
                         "RS melange nominal", rng, use_llm=use_llm, rare_dims=rare_dims)

        elif choice == "8":
            run_scenario(model, alpha, beta, gen_diluted_mining_vectors(rng, n=10),
                         "Mining dilue", rng, use_llm=use_llm, rare_dims=rare_dims)

        elif choice == "9":
            run_scenario(model, alpha, beta, gen_web_overlap_privesc_vectors(rng, n=10),
                         "Privesc + nominal fort", rng, use_llm=use_llm, rare_dims=rare_dims)

        else:
            print(f"  {RED}Choix invalide.{RESET}")

        input(f"\n  {DIM}[Appuyez sur Entree pour continuer]{RESET}")


if __name__ == "__main__":
    main()
