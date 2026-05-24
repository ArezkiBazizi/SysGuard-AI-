"""
Validation du pipeline Tier 2 avec un LLM REEL via OpenRouter.

Teste 6 scenarios representatifs (2 par type d'attaque + 2 faux positifs)
pour valider que le LLM reel reproduit les verdicts attendus.

Usage :
  cd ai_model
  python test_tier2_real.py --api-key sk-or-v1-...

Resultat ecrit dans tier2_real_results.json
"""

import argparse
import json
import os
import sys
import time

import httpx
import numpy as np

from _secrets import get_api_key, load_secrets

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_cfg   = load_secrets()
MODEL  = _cfg.get("LLM_MODEL", "deepseek/deepseek-v4-flash:free")
OUTPUT_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tier2_real_results.json")


# ---------------------------------------------------------------------------
# Construction du prompt (meme logique que agent/main.py)
# ---------------------------------------------------------------------------

def build_prompt(container_id, mse, severity, top_syscalls, event):
    fields = event.get("output_fields") or {}
    rule   = event.get("rule", "")
    output = event.get("output", "")
    pod    = fields.get("k8s.pod.name", container_id)
    ns     = fields.get("k8s.ns.name", "default")
    syscall_summary = ", ".join(
        f"{s['syscall']}({s['count']})" for s in top_syscalls[:10]
    )
    return (
        "Tu es un expert en securite Kubernetes et en analyse d'anomalies de syscalls.\n"
        "Tu recois un evenement detecte comme anomal par un Autoencoder (Tier 1).\n"
        "Ton role est de DECIDER s'il s'agit d'une menace reelle ou d'un faux positif.\n\n"
        f"--- CONTEXTE DE L'ANOMALIE ---\n"
        f"Pod/Conteneur   : {pod} (namespace: {ns})\n"
        f"Container ID    : {container_id}\n"
        f"Score MSE       : {mse:.6f}\n"
        f"Severite        : {severity}\n"
        f"Regle Falco     : {rule}\n"
        f"Message Falco   : {output}\n"
        f"Top syscalls (fenetre 10s) : {syscall_summary}\n\n"
        "--- INSTRUCTIONS ---\n"
        "Reponds UNIQUEMENT dans ce format JSON (sans texte autour) :\n"
        "{\n"
        '  "verdict": "MENACE" ou "FAUX_POSITIF",\n'
        '  "confidence": 0.0 a 1.0,\n'
        '  "attack_type": "type d\'attaque ou null",\n'
        '  "severity_cvss": "CRITIQUE/ELEVEE/MOYENNE/FAIBLE ou null",\n'
        '  "explanation": "explication concise"\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Appel LLM reel via OpenRouter
# ---------------------------------------------------------------------------

def call_llm(prompt: str, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://sysguard-ai.local",
        "X-Title":       "SysGuard-AI Tier2 Test",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role":    "system",
                "content": "Tu es un expert en cybersecurite Kubernetes. Reponds uniquement en JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens":  400,
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    raw  = content.strip()

    # Nettoyer le JSON (le LLM peut ajouter des backticks)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    if not raw:
        return {"verdict": "EMPTY_RESPONSE", "raw": raw}, raw
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return {"verdict": "PARSE_ERROR", "raw": raw}, raw


# ---------------------------------------------------------------------------
# Scenarios de test (6 representatifs)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "RS-1", "label": "Vraie attaque", "expected": "MENACE",
        "container_id": "abc001",
        "event": {
            "rule": "SysGuard syscall stream",
            "output": "Reverse shell detected: execve /bin/bash + connect to 192.168.1.100:4444 + dup2",
            "output_fields": {"k8s.pod.name": "web-pod-1", "k8s.ns.name": "default", "evt.type": "execve"},
        },
        "top_syscalls": [
            {"syscall": "execve", "count": 1800}, {"syscall": "connect", "count": 1100},
            {"syscall": "dup2",   "count": 700},  {"syscall": "socket",  "count": 500},
            {"syscall": "read",   "count": 300},
        ],
        "mse": 0.000520, "severity": "ANOMALIE",
    },
    {
        "id": "CM-1", "label": "Vraie attaque", "expected": "MENACE",
        "container_id": "def001",
        "event": {
            "rule": "SysGuard syscall stream",
            "output": "Crypto mining process: clone x12 threads, massive sched_yield burst (xmrig pattern)",
            "output_fields": {"k8s.pod.name": "worker-1", "k8s.ns.name": "default", "evt.type": "clone"},
        },
        "top_syscalls": [
            {"syscall": "sched_yield", "count": 8000}, {"syscall": "clone", "count": 5000},
            {"syscall": "fork",        "count": 2000}, {"syscall": "futex", "count": 3000},
            {"syscall": "mmap",        "count": 400},
        ],
        "mse": 0.001200, "severity": "ANOMALIE",
    },
    {
        "id": "PE-1", "label": "Vraie attaque", "expected": "MENACE",
        "container_id": "ghi001",
        "event": {
            "rule": "SysGuard syscall stream",
            "output": "Privilege escalation: ptrace + setuid + keyctl + capset detected in api-pod",
            "output_fields": {"k8s.pod.name": "api-1", "k8s.ns.name": "production", "evt.type": "ptrace"},
        },
        "top_syscalls": [
            {"syscall": "ptrace",           "count": 800}, {"syscall": "process_vm_readv", "count": 400},
            {"syscall": "keyctl",           "count": 300}, {"syscall": "setuid",           "count": 200},
            {"syscall": "capset",           "count": 150},
        ],
        "mse": 0.000350, "severity": "ANOMALIE",
    },
    {
        "id": "FP-DEP-1", "label": "Faux positif", "expected": "FAUX_POSITIF",
        "container_id": "jkl001",
        "event": {
            "rule": "SysGuard syscall stream",
            "output": "High syscall rate during rolling deployment: kubectl rollout update web-deployment",
            "output_fields": {"k8s.pod.name": "web-1", "k8s.ns.name": "default", "evt.type": "execve"},
        },
        "top_syscalls": [
            {"syscall": "execve", "count": 400}, {"syscall": "read",  "count": 300},
            {"syscall": "write",  "count": 280}, {"syscall": "clone", "count": 200},
            {"syscall": "mmap",   "count": 150},
        ],
        "mse": 0.000200, "severity": "ANOMALIE",
    },
    {
        "id": "FP-CRON-1", "label": "Faux positif", "expected": "FAUX_POSITIF",
        "container_id": "mno001",
        "event": {
            "rule": "SysGuard syscall stream",
            "output": "Scheduled backup cron job triggered: logrotate + backup script running",
            "output_fields": {"k8s.pod.name": "backup-1", "k8s.ns.name": "default", "evt.type": "openat"},
        },
        "top_syscalls": [
            {"syscall": "openat", "count": 600}, {"syscall": "read",  "count": 500},
            {"syscall": "write",  "count": 450}, {"syscall": "stat",  "count": 200},
            {"syscall": "close",  "count": 180},
        ],
        "mse": 0.000220, "severity": "ANOMALIE",
    },
    {
        "id": "FP-INIT-1", "label": "Faux positif", "expected": "FAUX_POSITIF",
        "container_id": "pqr001",
        "event": {
            "rule": "SysGuard syscall stream",
            "output": "Init container running apt-get update && pip install dependencies on startup",
            "output_fields": {"k8s.pod.name": "init-container-1", "k8s.ns.name": "staging", "evt.type": "execve"},
        },
        "top_syscalls": [
            {"syscall": "execve", "count": 350}, {"syscall": "read",   "count": 400},
            {"syscall": "write",  "count": 300}, {"syscall": "openat", "count": 250},
            {"syscall": "stat",   "count": 120},
        ],
        "mse": 0.000190, "severity": "ANOMALIE",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None, help="Optionnel si .env est configure")
    args = parser.parse_args()

    try:
        api_key = get_api_key(args.api_key)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print("=" * 62)
    print(f"  SysGuard-AI — Validation Tier 2 avec LLM REEL")
    print(f"  Modele : {MODEL}")
    print("=" * 62)
    print(f"\n  {len(SCENARIOS)} scenarios : 3 attaques + 3 faux positifs\n")
    print(f"  {'ID':<12} {'Attendu':<15} {'Obtenu':<15} {'Conf':<6} {'Latence':<10} {'OK?'}")
    print("  " + "-" * 60)

    results = []
    correct = 0

    for sc in SCENARIOS:
        prompt = build_prompt(
            sc["container_id"], sc["mse"], sc["severity"],
            sc["top_syscalls"], sc["event"],
        )
        t0 = time.perf_counter()
        try:
            response, raw = call_llm(prompt, api_key)
            lat_s = time.perf_counter() - t0
            verdict    = response.get("verdict", "ERREUR")
            confidence = response.get("confidence", 0)
            ok = (verdict == sc["expected"])
            correct += int(ok)
            mark = "OK" if ok else "FAIL"
            print(f"  {sc['id']:<12} {sc['expected']:<15} {verdict:<15} {confidence:<6.2f} {lat_s*1000:>6.0f} ms    {mark}")
        except Exception as e:
            lat_s = time.perf_counter() - t0
            print(f"  {sc['id']:<12} {sc['expected']:<15} {'ERREUR':<15} {'N/A':<6} {lat_s*1000:>6.0f} ms    FAIL ({e})")
            response = {"verdict": "ERREUR", "error": str(e)}
            raw = ""
            ok = False

        results.append({
            "id":       sc["id"],
            "label":    sc["label"],
            "expected": sc["expected"],
            "verdict":  response.get("verdict"),
            "correct":  ok,
            "confidence":   response.get("confidence"),
            "attack_type":  response.get("attack_type"),
            "severity_cvss": response.get("severity_cvss"),
            "explanation":  response.get("explanation"),
            "latency_s":    round(lat_s, 3),
        })
        time.sleep(0.5)   # pause courtoise entre appels

    # Synthese
    attacks  = [r for r in results if r["label"] == "Vraie attaque"]
    fps      = [r for r in results if r["label"] == "Faux positif"]
    tp       = sum(1 for r in attacks if r["verdict"] == "MENACE")
    tn       = sum(1 for r in fps    if r["verdict"] == "FAUX_POSITIF")
    lats     = [r["latency_s"] for r in results]

    print(f"\n  {'='*60}")
    print(f"  BILAN : {correct}/{len(SCENARIOS)} correct ({correct/len(SCENARIOS)*100:.0f}%)")
    print(f"  Attaques detectees : {tp}/{len(attacks)}")
    print(f"  Faux positifs leves : {tn}/{len(fps)}")
    print(f"  Latence LLM reel   : {sum(lats)/len(lats)*1000:.0f} ms moy, {max(lats)*1000:.0f} ms max")

    output = {
        "model":          MODEL,
        "total":          len(SCENARIOS),
        "correct":        correct,
        "accuracy":       round(correct/len(SCENARIOS), 3),
        "tp": tp, "tn": tn,
        "latency_mean_ms": round(sum(lats)/len(lats)*1000, 1),
        "latency_max_ms":  round(max(lats)*1000, 1),
        "scenarios": results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Rapport : {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
