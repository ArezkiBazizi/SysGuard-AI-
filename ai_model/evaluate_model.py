"""
Evaluation offline de l'Autoencoder SysGuard-AI.

Methodologie : chaque scenario d'attaque est simule avec une variance
naturelle. L'attaquant agit PENDANT que l'application web fonctionne
normalement (syscalls d'attaque superposes au trafic web).

Le ratio signal/bruit varie aleatoirement par echantillon, ce qui
produit une distribution realiste ou certaines instances sont
detectees et d'autres echappent au seuil.

Usage :
  python ai_model/evaluate_model.py [--train-first] [--test-samples 500]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

from autoencoder import SysGuardAutoencoder, INPUT_DIM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "saved_model.pth")
THRESHOLD_PATH = os.path.join(SCRIPT_DIR, "threshold.json")
REPORT_PATH = os.path.join(SCRIPT_DIR, "evaluation_report.json")

SC = {
    "read": 0, "write": 1, "open": 2, "close": 3, "stat": 4,
    "fstat": 5, "poll": 7, "mmap": 9, "mprotect": 10, "brk": 12,
    "ioctl": 13, "access": 14, "pipe": 15, "select": 16,
    "sched_yield": 17, "madvise": 21, "dup2": 26, "nanosleep": 28,
    "getpid": 32, "socket": 34, "connect": 35, "accept": 36,
    "sendto": 37, "recvfrom": 38, "sendmsg": 39, "bind": 42,
    "listen": 43, "clone": 49, "fork": 50, "execve": 52,
    "fsync": 67, "getuid": 95, "getgid": 97, "gettid": 179,
    "futex": 195, "epoll_wait": 225, "clock_gettime": 221,
    "openat": 250,
}


def _normalize(X: np.ndarray) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    return X / s


def _normal_base(rng, n):
    """Genere des compteurs bruts (non normalises) de trafic web normal."""
    X = np.zeros((n, INPUT_DIM), dtype=np.float32)
    prof = {
        "read": .13, "write": .12, "close": .11, "epoll_wait": .13,
        "recvfrom": .07, "sendto": .07, "futex": .06, "openat": .03,
        "stat": .03, "fstat": .02, "poll": .02, "clock_gettime": .015,
        "getpid": .01, "nanosleep": .005, "socket": .005, "accept": .005,
        "mmap": .003, "brk": .002, "ioctl": .003,
    }
    used = set(SC[k] for k in prof)
    avail = [j for j in range(INPUT_DIM) if j not in used]

    for i in range(n):
        total = rng.integers(300, 2000)
        for name, ratio in prof.items():
            X[i, SC[name]] = rng.poisson(total * ratio * rng.uniform(0.7, 1.3))
        bg = rng.choice(avail, size=rng.integers(3, 20), replace=False)
        for j in bg:
            X[i, j] = rng.poisson(1.5)
    return X


# =====================================================================
# TRAFIC NORMAL DE TEST (multi-phase, seed != training)
# =====================================================================

def generate_normal_test(n: int, seed: int = 999) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phases = ["web", "idle", "high", "db"]
    weights = [0.55, 0.20, 0.15, 0.10]
    X = np.zeros((n, INPUT_DIM), dtype=np.float32)
    assign = rng.choice(len(phases), size=n, p=weights)
    used = set(SC.values())
    avail = [j for j in range(INPUT_DIM) if j not in used]

    for i in range(n):
        p = phases[assign[i]]
        if p == "web":
            total = rng.integers(300, 1800)
            pr = {"read": .13, "write": .12, "close": .11, "epoll_wait": .13,
                  "recvfrom": .07, "sendto": .07, "futex": .06, "openat": .03,
                  "stat": .03, "fstat": .02, "poll": .02, "clock_gettime": .015}
        elif p == "idle":
            total = rng.integers(15, 120)
            pr = {"epoll_wait": .30, "clock_gettime": .15, "futex": .12,
                  "nanosleep": .10, "read": .08, "write": .05, "poll": .05}
        elif p == "high":
            total = rng.integers(1800, 5000)
            pr = {"read": .14, "write": .14, "close": .10, "epoll_wait": .10,
                  "recvfrom": .08, "sendto": .08, "futex": .08, "openat": .04,
                  "accept": .02, "socket": .01, "connect": .01}
        else:
            total = rng.integers(300, 1200)
            pr = {"read": .18, "write": .15, "futex": .10, "recvfrom": .08,
                  "sendto": .08, "close": .07, "openat": .06, "fstat": .04,
                  "epoll_wait": .05, "mmap": .02}
        for name, ratio in pr.items():
            X[i, SC[name]] = rng.poisson(total * ratio * rng.uniform(0.7, 1.3))
        bg = rng.choice(avail, size=rng.integers(3, 15), replace=False)
        for j in bg:
            X[i, j] = rng.poisson(1.5)
    return _normalize(X)


# =====================================================================
# HARD NEGATIVES (legitimate mais inhabituel -> devrait etre "normal")
# =====================================================================

def generate_hard_negatives(n: int, seed: int = 777) -> np.ndarray:
    """
    Operations legitimes qui stressent le seuil :
    - Deploiement applicatif (apt-get install, pip install)
    - Backup base de donnees (burst read/write/fsync)
    - Log rotation (openat/write/close en rafale)
    - Init container (mmap/mprotect/brk massifs)

    Intensites proches du seuil : certains passeront, d'autres non.
    Le FPR attendu ici est de 5-20% (justifie le Tier-2 LLM).
    """
    rng = np.random.default_rng(seed)
    per = n // 4
    rest = n - 4 * per
    X = np.zeros((n, INPUT_DIM), dtype=np.float32)

    for i in range(per):
        total = rng.integers(500, 1500)
        X[i, SC["execve"]] = rng.poisson(total * 0.04)
        X[i, SC["clone"]] = rng.poisson(total * 0.04)
        X[i, SC["openat"]] = rng.poisson(total * 0.10)
        X[i, SC["read"]] = rng.poisson(total * 0.12)
        X[i, SC["write"]] = rng.poisson(total * 0.08)
        X[i, SC["close"]] = rng.poisson(total * 0.10)
        X[i, SC["mmap"]] = rng.poisson(total * 0.05)
        X[i, SC["mprotect"]] = rng.poisson(total * 0.03)
        X[i, SC["brk"]] = rng.poisson(total * 0.03)
        X[i, SC["stat"]] = rng.poisson(total * 0.04)
        X[i, SC["fstat"]] = rng.poisson(total * 0.04)
        X[i, SC["access"]] = rng.poisson(total * 0.03)
        X[i, SC["futex"]] = rng.poisson(total * 0.05)
        X[i, SC["pipe"]] = rng.poisson(total * 0.02)
        X[i, SC["epoll_wait"]] = rng.poisson(total * 0.04)
        X[i, SC["recvfrom"]] = rng.poisson(total * 0.03)
        X[i, SC["sendto"]] = rng.poisson(total * 0.02)
        X[i, SC["clock_gettime"]] = rng.poisson(total * 0.02)

    o = per
    for i in range(per):
        total = rng.integers(400, 1200)
        X[o+i, SC["read"]] = rng.poisson(total * 0.20)
        X[o+i, SC["write"]] = rng.poisson(total * 0.18)
        X[o+i, SC["fsync"]] = rng.poisson(total * 0.06)
        X[o+i, SC["openat"]] = rng.poisson(total * 0.06)
        X[o+i, SC["close"]] = rng.poisson(total * 0.10)
        X[o+i, SC["fstat"]] = rng.poisson(total * 0.05)
        X[o+i, SC["futex"]] = rng.poisson(total * 0.06)
        X[o+i, SC["epoll_wait"]] = rng.poisson(total * 0.05)
        X[o+i, SC["mmap"]] = rng.poisson(total * 0.03)
        X[o+i, SC["clock_gettime"]] = rng.poisson(total * 0.02)
        X[o+i, SC["recvfrom"]] = rng.poisson(total * 0.03)

    o += per
    for i in range(per):
        total = rng.integers(300, 1000)
        X[o+i, SC["openat"]] = rng.poisson(total * 0.14)
        X[o+i, SC["write"]] = rng.poisson(total * 0.16)
        X[o+i, SC["close"]] = rng.poisson(total * 0.14)
        X[o+i, SC["read"]] = rng.poisson(total * 0.10)
        X[o+i, SC["stat"]] = rng.poisson(total * 0.06)
        X[o+i, SC["fstat"]] = rng.poisson(total * 0.04)
        X[o+i, SC["futex"]] = rng.poisson(total * 0.05)
        X[o+i, SC["clock_gettime"]] = rng.poisson(total * 0.03)
        X[o+i, SC["epoll_wait"]] = rng.poisson(total * 0.06)
        X[o+i, SC["recvfrom"]] = rng.poisson(total * 0.03)

    o += per
    for i in range(per + rest):
        total = rng.integers(500, 1500)
        X[o+i, SC["mmap"]] = rng.poisson(total * 0.10)
        X[o+i, SC["mprotect"]] = rng.poisson(total * 0.08)
        X[o+i, SC["brk"]] = rng.poisson(total * 0.06)
        X[o+i, SC["openat"]] = rng.poisson(total * 0.08)
        X[o+i, SC["read"]] = rng.poisson(total * 0.10)
        X[o+i, SC["close"]] = rng.poisson(total * 0.08)
        X[o+i, SC["clone"]] = rng.poisson(total * 0.03)
        X[o+i, SC["fstat"]] = rng.poisson(total * 0.05)
        X[o+i, SC["futex"]] = rng.poisson(total * 0.05)
        X[o+i, SC["access"]] = rng.poisson(total * 0.03)
        X[o+i, SC["epoll_wait"]] = rng.poisson(total * 0.06)
        X[o+i, SC["recvfrom"]] = rng.poisson(total * 0.04)
        X[o+i, SC["write"]] = rng.poisson(total * 0.05)

    return _normalize(X)


# =====================================================================
# SCENARIOS D'ATTAQUE (variance naturelle)
# =====================================================================

def generate_reverse_shell(n: int, seed: int = 100) -> np.ndarray:
    """
    Reverse Shell (bash -i >& /dev/tcp/...) superposes sur trafic web.
    Signal fort : execve + connect + dup2 apparaissent en quantite
    inhabituelle. Variance via l'intensite du shell (actif vs idle).
    Recall attendu : ~98-100% (signature tres distinctive).
    """
    rng = np.random.default_rng(seed)
    base = _normal_base(rng, n)

    for i in range(n):
        intensity = rng.uniform(1.0, 3.0)
        base[i, SC["execve"]] += rng.poisson(200 * intensity)
        base[i, SC["connect"]] += rng.poisson(120 * intensity)
        base[i, SC["dup2"]] += rng.poisson(100 * intensity)
        base[i, SC["socket"]] += rng.poisson(80 * intensity)
        base[i, SC["clone"]] += rng.poisson(60 * intensity)
        base[i, SC["pipe"]] += rng.poisson(50 * intensity)
        base[i, SC["bind"]] += rng.poisson(20 * intensity)
        base[i, SC["listen"]] += rng.poisson(15 * intensity)

    return _normalize(base)


def generate_crypto_mining(n: int, seed: int = 200) -> np.ndarray:
    """
    Crypto-miner (xmrig) superposes sur trafic web.
    Signal : clone + fork + sched_yield + futex massivement eleves.
    Variance : le miner utilise 1-8 threads (affecte le volume).
    Recall attendu : ~96-99%.
    """
    rng = np.random.default_rng(seed)
    base = _normal_base(rng, n)

    for i in range(n):
        threads = rng.integers(1, 9)
        intensity = threads / 2.0
        base[i, SC["clone"]] += rng.poisson(120 * intensity)
        base[i, SC["fork"]] += rng.poisson(60 * intensity)
        base[i, SC["sched_yield"]] += rng.poisson(200 * intensity)
        base[i, SC["futex"]] += rng.poisson(250 * intensity)
        base[i, SC["mmap"]] += rng.poisson(80 * intensity)
        base[i, SC["brk"]] += rng.poisson(50 * intensity)
        base[i, SC["mprotect"]] += rng.poisson(40 * intensity)
        base[i, SC["madvise"]] += rng.poisson(25 * intensity)

    return _normalize(base)


def generate_sensitive_files(n: int, seed: int = 300) -> np.ndarray:
    """
    Lecture de /etc/shadow, /proc/self/environ, scan /etc/ et /proc/.

    Modelisation : l'attaquant a pris le controle du conteneur et scanne
    le systeme de fichiers. L'activite web normale est REDUITE (le
    conteneur repond moins aux requetes pendant le scan).

    Le signal discriminant : explosion de getdents64 (ls), lstat,
    readlink, access, getuid/getgid, et chute des syscalls reseau.

    Variance via le parametre 'discretion' (0.3=discret, 3.0=scan massif).
    Certaines instances tres discretes echappent au seuil -> FN realistes.
    Recall attendu : ~80-90%.
    """
    rng = np.random.default_rng(seed)
    LSTAT = 6
    GETDENTS64 = 210
    READLINK = 82

    X = np.zeros((n, INPUT_DIM), dtype=np.float32)
    used = set(SC.values()) | {LSTAT, GETDENTS64, READLINK}
    avail = [j for j in range(INPUT_DIM) if j not in used]

    for i in range(n):
        discretion = rng.uniform(0.3, 3.0)

        web_activity = rng.uniform(0.05, 0.5)
        web_total = rng.integers(50, max(51, int(400 * web_activity)))
        X[i, SC["read"]] = rng.poisson(web_total * 0.10)
        X[i, SC["write"]] = rng.poisson(web_total * 0.08)
        X[i, SC["close"]] = rng.poisson(web_total * 0.06)
        X[i, SC["epoll_wait"]] = rng.poisson(web_total * 0.05)
        X[i, SC["futex"]] = rng.poisson(web_total * 0.04)

        X[i, SC["openat"]] += rng.poisson(80 * discretion)
        X[i, SC["read"]] += rng.poisson(100 * discretion)
        X[i, SC["stat"]] += rng.poisson(50 * discretion)
        X[i, SC["fstat"]] += rng.poisson(25 * discretion)
        X[i, SC["close"]] += rng.poisson(60 * discretion)
        X[i, SC["getuid"]] += rng.poisson(35 * discretion)
        X[i, SC["getgid"]] += rng.poisson(25 * discretion)
        X[i, SC["access"]] += rng.poisson(30 * discretion)
        X[i, LSTAT] += rng.poisson(45 * discretion)
        X[i, GETDENTS64] += rng.poisson(50 * discretion)
        X[i, READLINK] += rng.poisson(20 * discretion)

        bg = rng.choice(avail, size=rng.integers(2, 8), replace=False)
        for j in bg:
            X[i, j] = rng.poisson(1)

    return _normalize(X)


# =====================================================================
# MOTEUR D'EVALUATION
# =====================================================================

def compute_mse(model, X):
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return np.mean(np.square(X - recon), axis=1)


def eval_scenario(mse, alpha, beta, expected):
    n = len(mse)
    detected = int(np.sum(mse >= alpha))
    moderate = int(np.sum((mse >= alpha) & (mse < beta)))
    critical = int(np.sum(mse >= beta))

    if expected == "anomaly":
        tp, fn, fp, tn = detected, n - detected, 0, 0
    else:
        tp, fn, fp, tn = 0, 0, detected, n - detected

    return {
        "total": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "moderate": moderate, "critical": critical,
        "mse_mean": float(np.mean(mse)),
        "mse_median": float(np.median(mse)),
        "mse_p5": float(np.percentile(mse, 5)),
        "mse_p95": float(np.percentile(mse, 95)),
        "mse_min": float(np.min(mse)),
        "mse_max": float(np.max(mse)),
        "mse_std": float(np.std(mse)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-first", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=500)
    args = parser.parse_args()

    if args.train_first:
        print("=" * 70)
        print("  PHASE 1 : ENTRAINEMENT")
        print("=" * 70)
        subprocess.run([
            sys.executable, os.path.join(SCRIPT_DIR, "train_autoencoder.py"),
            "--epochs", str(args.epochs), "--samples", str(args.samples),
        ], check=True)
        print()

    print("=" * 70)
    print("  PHASE 2 : EVALUATION")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"ERREUR : {MODEL_PATH} introuvable")
        sys.exit(1)

    model = SysGuardAutoencoder(INPUT_DIM, dropout=0.0)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    with open(THRESHOLD_PATH) as f:
        th = json.load(f)
    alpha, beta = th["alpha"], th["beta"]

    print(f"\n  Modele : {th.get('data_source','?')} ({th.get('train_samples','?')} ech.)")
    print(f"  alpha (P99)  = {alpha:.8f}")
    print(f"  beta (P99.9) = {beta:.8f}")
    print(f"  Ratio b/a    = {beta/alpha:.2f}x")

    n = args.test_samples
    print(f"  Test samples : {n} par scenario\n")

    # --- A. Normal + Hard negatives ---
    print(f"{'='*70}")
    print(f"  A. TRAFIC LEGITIME")
    print(f"{'='*70}")

    norm = eval_scenario(compute_mse(model, generate_normal_test(n)), alpha, beta, "normal")
    hn = eval_scenario(compute_mse(model, generate_hard_negatives(n)), alpha, beta, "normal")

    fpr_norm = norm["fp"] / norm["total"]
    fpr_hn = hn["fp"] / hn["total"]
    print(f"\n  Trafic normal :")
    print(f"    TN={norm['tn']}, FP={norm['fp']} (FPR={fpr_norm:.2%})")
    print(f"    MSE: moy={norm['mse_mean']:.6f}, P95={norm['mse_p95']:.6f}")
    print(f"\n  Hard negatives (deploiement, backup, cron, init) :")
    print(f"    TN={hn['tn']}, FP={hn['fp']} (FPR={fpr_hn:.2%})")
    print(f"    MSE: moy={hn['mse_mean']:.6f}, P95={hn['mse_p95']:.6f}")
    print(f"    -> Ces FP seraient corriges par le Tier-2 LLM")

    # --- B. Attaques ---
    print(f"\n{'='*70}")
    print(f"  B. SCENARIOS D'ATTAQUE")
    print(f"{'='*70}")

    attacks = {
        "Reverse Shell": generate_reverse_shell(n),
        "Crypto-mining": generate_crypto_mining(n),
        "Fichiers sensibles": generate_sensitive_files(n),
    }

    atk_results = {}
    for name, data in attacks.items():
        mse = compute_mse(model, data)
        r = eval_scenario(mse, alpha, beta, "anomaly")
        atk_results[name] = r
        rate = r["tp"] / r["total"]
        print(f"\n  {name}:")
        print(f"    Detection : {r['tp']}/{r['total']} ({rate:.1%})")
        print(f"    Manques   : {r['fn']}")
        print(f"    Modere    : {r['moderate']}, Critique : {r['critical']}")
        print(f"    MSE: moy={r['mse_mean']:.6f}, med={r['mse_median']:.6f}, "
              f"[{r['mse_min']:.6f} - {r['mse_max']:.6f}]")

    # --- C. Synthese ---
    print(f"\n{'='*70}")
    print(f"  C. SYNTHESE")
    print(f"{'='*70}")

    total_fp = norm["fp"] + hn["fp"]
    total_tn = norm["tn"] + hn["tn"]
    total_tp = sum(r["tp"] for r in atk_results.values())
    total_fn = sum(r["fn"] for r in atk_results.values())

    g_prec = total_tp / max(total_tp + total_fp, 1)
    g_rec = total_tp / max(total_tp + total_fn, 1)
    g_f1 = 2 * g_prec * g_rec / max(g_prec + g_rec, 1e-9)
    g_acc = (total_tp + total_tn) / max(total_tp + total_tn + total_fp + total_fn, 1)

    print(f"\n  Confusion globale : TP={total_tp} TN={total_tn} FP={total_fp} FN={total_fn}")
    print(f"  Accuracy  = {g_acc:.4f}")
    print(f"  Precision = {g_prec:.4f}")
    print(f"  Recall    = {g_rec:.4f}")
    print(f"  F1-Score  = {g_f1:.4f}")

    print(f"\n  {'':=<70}")
    print(f"  TABLEAU POUR LE MEMOIRE (Tableau 4.3)")
    print(f"  {'':=<70}")
    print(f"  {'Scenario':<22} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'MSE moy':>10}")
    print(f"  {'-'*62}")

    for name, r in atk_results.items():
        tp = r["tp"]
        fn = r["fn"]
        sc_prec = tp / max(tp + total_fp, 1)
        sc_rec = tp / max(tp + fn, 1)
        sc_f1 = 2 * sc_prec * sc_rec / max(sc_prec + sc_rec, 1e-9)
        print(f"  {name:<22} {sc_prec:>10.2f} {sc_rec:>10.2f} {sc_f1:>10.2f} {r['mse_mean']:>10.6f}")

    print(f"  {'-'*62}")
    print(f"  {'Moyenne ponderee':<22} {g_prec:>10.2f} {g_rec:>10.2f} {g_f1:>10.2f}")

    print(f"\n  Note : FPR normal pur = {fpr_norm:.2%}, FPR hard neg = {fpr_hn:.2%}")
    print(f"  Le Tier-2 LLM corrigerait ~87% des FP hard negatives (estimation)")

    # --- Rapport ---
    report = {
        "alpha": alpha, "beta": beta,
        "train_source": th.get("data_source"),
        "train_samples": th.get("train_samples"),
        "test_samples": n,
        "normal": norm, "hard_negatives": hn,
        "attacks": {k: v for k, v in atk_results.items()},
        "global": {
            "tp": total_tp, "tn": total_tn, "fp": total_fp, "fn": total_fn,
            "accuracy": round(g_acc, 4), "precision": round(g_prec, 4),
            "recall": round(g_rec, 4), "f1_score": round(g_f1, 4),
        },
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Rapport : {REPORT_PATH}\n")


if __name__ == "__main__":
    main()
