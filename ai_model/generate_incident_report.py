"""
Generateur de rapport d'incident Tier 2 — SysGuard-AI.

Pour chaque alerte MENACE du Tier 1, ce script soumet le contexte au LLM
qui redige un rapport d'incident structure (verdict + analyse + actions).

Usage :
  cd ai_model
  python generate_incident_report.py --api-key sk-or-v1-...
  python generate_incident_report.py --api-key sk-or-v1-... --scenario 2

Scenarios disponibles :
  1 = Reverse Shell
  2 = Crypto-mining
  3 = Privilege Escalation
  4 = Acces fichiers sensibles (/etc/shadow)

Rapport sauvegarde dans : incident_reports/incident_<id>.json et .txt
"""

import argparse
import json
import os
import sys
import time
import textwrap
from datetime import datetime, timezone

import httpx

from _secrets import get_api_key, load_secrets

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_cfg   = load_secrets()
MODEL  = _cfg.get("LLM_MODEL", "deepseek/deepseek-v4-flash:free")
REPORTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "incident_reports")


# ---------------------------------------------------------------------------
# Prompt avec demande de rapport complet
# ---------------------------------------------------------------------------

def build_report_prompt(scenario: dict) -> str:
    sc      = scenario
    syscall_summary = ", ".join(
        f"{s['syscall']}({s['count']})" for s in sc["top_syscalls"]
    )
    return (
        "Tu es un expert en cybersecurite Kubernetes et en reponse a incident (DFIR).\n"
        "Le systeme SysGuard-AI (Autoencoder Tier 1) a detecte une anomalie dans un cluster Kubernetes.\n"
        "Ton role : analyser le contexte et produire un rapport d'incident complet et structure.\n\n"
        "=== CONTEXTE DE L'ALERTE TIER 1 ===\n"
        f"Horodatage        : {sc['timestamp']}\n"
        f"Pod/Conteneur     : {sc['pod']} (namespace: {sc['namespace']})\n"
        f"Image Docker      : {sc['image']}\n"
        f"Container ID      : {sc['container_id']}\n"
        f"Score MSE         : {sc['mse']:.6f}\n"
        f"Seuils            : alpha={sc['alpha']:.6f}, beta={sc['beta']:.6f}\n"
        f"Severite Tier 1   : {sc['severity']}\n"
        f"Regle Falco       : {sc['falco_rule']}\n"
        f"Message Falco     : {sc['falco_output']}\n"
        f"Top syscalls (10s): {syscall_summary}\n"
        f"Action appliquee  : {sc['action_taken']}\n\n"
        "=== INSTRUCTIONS ===\n"
        "Redige un rapport d'incident COMPLET au format JSON strict (sans texte autour).\n"
        "Le rapport doit contenir EXACTEMENT ces champs :\n"
        "{\n"
        '  "verdict": "MENACE" ou "FAUX_POSITIF",\n'
        '  "confidence": 0.0 a 1.0,\n'
        '  "attack_type": "nom de l\'attaque (ex: Reverse Shell, Cryptomining, etc.)",\n'
        '  "severity_cvss": "CRITIQUE" | "ELEVEE" | "MOYENNE" | "FAIBLE",\n'
        '  "mitre_tactic": "tactique MITRE ATT&CK correspondante",\n'
        '  "mitre_technique": "technique MITRE ATT&CK (ex: T1059.004)",\n'
        '  "summary": "resume executif de l\'incident en 2-3 phrases",\n'
        '  "timeline": [\n'
        '    {"t": "T+0s",  "event": "description de ce qui s\'est passe"},\n'
        '    {"t": "T+Xs",  "event": "..."}\n'
        '  ],\n'
        '  "iocs": ["liste des indicateurs de compromission observes"],\n'
        '  "blast_radius": "description de l\'impact potentiel si non contenu",\n'
        '  "recommended_actions": [\n'
        '    "action 1 recommandee",\n'
        '    "action 2 recommandee"\n'
        '  ],\n'
        '  "quarantine_decision": "MAINTENIR" | "LEVER",\n'
        '  "analyst_note": "note technique pour l\'analyste SOC"\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Scenarios d'alerte MENACE
# ---------------------------------------------------------------------------

SCENARIOS = {
    1: {
        "id": "INC-001",
        "name": "Reverse Shell",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pod": "web-frontend-7d9f8b-x4k2p",
        "namespace": "production",
        "image": "nginx:1.25-alpine",
        "container_id": "3f7a9c2e1b4d",
        "mse": 0.000520,
        "alpha": 0.000173,
        "beta": 0.001920,
        "severity": "ANOMALIE",
        "falco_rule": "SysGuard syscall stream",
        "falco_output": "Suspicious process execve+connect+dup2 detected in nginx container: /bin/bash -> connect(192.168.1.100:4444) with file descriptor duplication",
        "action_taken": "NetworkPolicy deny-all appliquee (quarantaine reseau revers.",
        "top_syscalls": [
            {"syscall": "execve",  "count": 1800},
            {"syscall": "connect", "count": 1100},
            {"syscall": "dup2",    "count": 700},
            {"syscall": "socket",  "count": 500},
            {"syscall": "read",    "count": 300},
            {"syscall": "write",   "count": 280},
            {"syscall": "close",   "count": 250},
            {"syscall": "wait4",   "count": 200},
            {"syscall": "fork",    "count": 180},
            {"syscall": "stat",    "count": 120},
        ],
        "expected_verdict": "MENACE",
    },
    2: {
        "id": "INC-002",
        "name": "Crypto-mining",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pod": "api-worker-5c8b9d-z3m7q",
        "namespace": "default",
        "image": "python:3.11-slim",
        "container_id": "8e2b1f4a7c9d",
        "mse": 0.001200,
        "alpha": 0.000173,
        "beta": 0.001920,
        "severity": "ANOMALIE",
        "falco_rule": "SysGuard syscall stream",
        "falco_output": "Cryptomining pattern detected: clone x5000 threads + sched_yield x8000 burst (xmrig signature) in api-worker pod",
        "action_taken": "NetworkPolicy deny-all appliquee (quarantaine reseau).",
        "top_syscalls": [
            {"syscall": "sched_yield", "count": 8000},
            {"syscall": "clone",       "count": 5000},
            {"syscall": "futex",       "count": 3000},
            {"syscall": "fork",        "count": 2000},
            {"syscall": "mmap",        "count": 400},
            {"syscall": "read",        "count": 350},
            {"syscall": "write",       "count": 300},
            {"syscall": "brk",         "count": 250},
            {"syscall": "mprotect",    "count": 200},
            {"syscall": "close",       "count": 180},
        ],
        "expected_verdict": "MENACE",
    },
    3: {
        "id": "INC-003",
        "name": "Privilege Escalation",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pod": "backend-api-4b7c8e-p9n1x",
        "namespace": "production",
        "image": "myapp/backend:2.1.0",
        "container_id": "5c3d9f2a8b7e",
        "mse": 0.000350,
        "alpha": 0.000173,
        "beta": 0.001920,
        "severity": "ANOMALIE",
        "falco_rule": "SysGuard syscall stream",
        "falco_output": "Privilege escalation sequence: ptrace+process_vm_readv+keyctl+setuid+capset detected — possible container escape attempt in backend-api",
        "action_taken": "NetworkPolicy deny-all appliquee (quarantaine reseau).",
        "top_syscalls": [
            {"syscall": "ptrace",            "count": 800},
            {"syscall": "process_vm_readv",  "count": 400},
            {"syscall": "keyctl",            "count": 300},
            {"syscall": "setuid",            "count": 200},
            {"syscall": "capset",            "count": 150},
            {"syscall": "execve",            "count": 120},
            {"syscall": "open",              "count": 100},
            {"syscall": "read",              "count": 90},
            {"syscall": "write",             "count": 80},
            {"syscall": "mmap",              "count": 70},
        ],
        "expected_verdict": "MENACE",
    },
    4: {
        "id": "INC-004",
        "name": "Acces fichiers sensibles",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pod": "monitoring-agent-2x9k1-t8p3m",
        "namespace": "kube-system",
        "image": "custom/monitor:1.0",
        "container_id": "9a4f7e2c1b8d",
        "mse": 0.000210,
        "alpha": 0.000173,
        "beta": 0.001920,
        "severity": "ANOMALIE",
        "falco_rule": "SysGuard syscall stream",
        "falco_output": "Sensitive file access: repeated openat(/etc/shadow, /etc/passwd, /root/.ssh/id_rsa) detected — credential harvesting pattern",
        "action_taken": "NetworkPolicy deny-all appliquee (quarantaine reseau).",
        "top_syscalls": [
            {"syscall": "openat", "count": 900},
            {"syscall": "read",   "count": 750},
            {"syscall": "close",  "count": 600},
            {"syscall": "stat",   "count": 400},
            {"syscall": "lstat",  "count": 300},
            {"syscall": "getdents","count": 200},
            {"syscall": "access", "count": 180},
            {"syscall": "write",  "count": 120},
            {"syscall": "mmap",   "count": 100},
            {"syscall": "brk",    "count": 80},
        ],
        "expected_verdict": "MENACE",
    },
}


# ---------------------------------------------------------------------------
# Appel LLM
# ---------------------------------------------------------------------------

def call_llm_report(prompt: str, api_key: str) -> tuple[dict, str, float]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://sysguard-ai.local",
        "X-Title":       "SysGuard-AI Incident Report",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role":    "system",
                "content": (
                    "Tu es un expert DFIR (Digital Forensics and Incident Response) "
                    "specialise Kubernetes. Reponds UNIQUEMENT en JSON strict."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "max_tokens":  900,
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
    lat = time.perf_counter() - t0
    resp.raise_for_status()

    data    = resp.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    content = content.strip()

    # Nettoyer les balises markdown
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    if not content:
        return {"error": "empty_response"}, "", lat
    try:
        return json.loads(content), content, lat
    except json.JSONDecodeError as e:
        return {"error": f"parse_error: {e}", "raw": content}, content, lat


# ---------------------------------------------------------------------------
# Formatage texte du rapport
# ---------------------------------------------------------------------------

def format_report_txt(sc_meta: dict, report: dict, lat: float) -> str:
    sep = "=" * 68
    thin = "-" * 68
    lines = [
        sep,
        f"  RAPPORT D'INCIDENT SYSGUARD-AI  —  {sc_meta['id']}",
        f"  {sc_meta['name'].upper()}",
        sep,
        f"  Horodatage    : {sc_meta['timestamp']}",
        f"  Pod           : {sc_meta['pod']} ({sc_meta['namespace']})",
        f"  Image         : {sc_meta['image']}",
        f"  Genere par    : LLM {MODEL} via OpenRouter",
        f"  Latence LLM   : {lat*1000:.0f} ms",
        thin,
        "",
        f"  VERDICT        : {report.get('verdict', 'N/A')}",
        f"  CONFIANCE      : {report.get('confidence', 'N/A')}",
        f"  TYPE D'ATTAQUE : {report.get('attack_type', 'N/A')}",
        f"  SEVERITE       : {report.get('severity_cvss', 'N/A')}",
        f"  MITRE TACTIC   : {report.get('mitre_tactic', 'N/A')}",
        f"  MITRE TECHNIQUE: {report.get('mitre_technique', 'N/A')}",
        f"  QUARANTAINE    : {report.get('quarantine_decision', 'N/A')}",
        "",
        thin,
        "  RESUME EXECUTIF",
        thin,
    ]
    summary = report.get("summary", "N/A")
    for line in textwrap.wrap(summary, width=64):
        lines.append(f"  {line}")

    lines += ["", thin, "  CHRONOLOGIE DE L'INCIDENT", thin]
    for event in report.get("timeline", []):
        lines.append(f"  {event.get('t','?'):>6}  {event.get('event','')}")

    lines += ["", thin, "  INDICATEURS DE COMPROMISSION (IOC)", thin]
    for ioc in report.get("iocs", []):
        lines.append(f"  [!] {ioc}")

    lines += ["", thin, "  PERIMETRE D'IMPACT (BLAST RADIUS)", thin]
    blast = report.get("blast_radius", "N/A")
    for line in textwrap.wrap(blast, width=64):
        lines.append(f"  {line}")

    lines += ["", thin, "  ACTIONS RECOMMANDEES", thin]
    for i, action in enumerate(report.get("recommended_actions", []), 1):
        for j, line in enumerate(textwrap.wrap(action, width=60)):
            prefix = f"  {i}. " if j == 0 else "     "
            lines.append(f"{prefix}{line}")

    lines += ["", thin, "  NOTE ANALYSTE SOC", thin]
    note = report.get("analyst_note", "N/A")
    for line in textwrap.wrap(note, width=64):
        lines.append(f"  {line}")

    lines += ["", sep]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_scenario_report(scenario_id: int, api_key: str = None) -> dict:
    """Genere un rapport d'incident pour un scenario et le sauvegarde sur disque."""
    api_key = get_api_key(api_key)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    sc_meta = SCENARIOS[scenario_id]
    prompt = build_report_prompt(sc_meta)
    report, _raw, lat = call_llm_report(prompt, api_key)

    if "error" in report:
        return {"error": report["error"], "meta": sc_meta}

    json_path = os.path.join(REPORTS_DIR, f"incident_{sc_meta['id'].lower()}.json")
    full_json = {
        "meta": sc_meta,
        "model": MODEL,
        "latency_ms": round(lat * 1000, 1),
        "report": report,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(REPORTS_DIR, f"incident_{sc_meta['id'].lower()}.txt")
    txt_report = format_report_txt(sc_meta, report, lat)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_report)

    return {
        "meta": sc_meta,
        "report": report,
        "latency_ms": round(lat * 1000, 1),
        "json_path": json_path,
        "txt_path": txt_path,
        "txt_content": txt_report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",  default=None, help="Optionnel si .env est configure")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], default=None,
                        help="1=ReverseShell 2=Cryptomining 3=PrivEsc 4=FileAccess (defaut: tous)")
    args = parser.parse_args()

    try:
        api_key = get_api_key(args.api_key)
    except ValueError as e:
        print(e)
        sys.exit(1)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    ids_to_run = [args.scenario] if args.scenario else list(SCENARIOS.keys())

    print("=" * 68)
    print("  SysGuard-AI — Generateur de Rapports d'Incident (Tier 2 LLM)")
    print(f"  Modele : {MODEL}")
    print("=" * 68)

    for sid in ids_to_run:
        sc_meta = SCENARIOS[sid]
        print(f"\n  [{sc_meta['id']}] {sc_meta['name']} ...")

        prompt = build_report_prompt(sc_meta)
        report, raw, lat = call_llm_report(prompt, api_key)

        if "error" in report:
            print(f"  ERREUR : {report['error']}")
            continue

        # Sauvegarder JSON
        json_path = os.path.join(REPORTS_DIR, f"incident_{sc_meta['id'].lower()}.json")
        full_json  = {
            "meta":   sc_meta,
            "model":  MODEL,
            "latency_ms": round(lat * 1000, 1),
            "report": report,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_json, f, indent=2, ensure_ascii=False)

        # Sauvegarder TXT lisible
        txt_path  = os.path.join(REPORTS_DIR, f"incident_{sc_meta['id'].lower()}.txt")
        txt_report = format_report_txt(sc_meta, report, lat)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_report)

        # Afficher dans le terminal
        print(txt_report)
        print(f"\n  Rapport JSON : {json_path}")
        print(f"  Rapport TXT  : {txt_path}")

        if len(ids_to_run) > 1:
            time.sleep(1.0)   # pause courtoise entre appels API


if __name__ == "__main__":
    main()
