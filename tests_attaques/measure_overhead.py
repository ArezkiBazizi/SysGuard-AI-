"""
Mesure l'overhead CPU, mémoire et latence de l'agent SysGuard-AI.

Mesure séparée pour :
  - Tier 1 (Autoencoder seul)   : latence de détection
  - Tier 2 (LLM inclus)         : latence totale
  - Overhead CPU/RAM sous charge

Les résultats sont exportés en JSON pour intégration dans le mémoire.

Usage :
  1. Lancer l'agent : uvicorn agent.main:app --port 8000
  2. python tests_attaques/measure_overhead.py --agent-url http://localhost:8000 --pid <PID_AGENT>
"""

import argparse
import json
import os
import time
import statistics

try:
    import psutil
except ImportError:
    psutil = None

import httpx


def measure_process(pid: int) -> dict:
    if psutil is None:
        return {"error": "psutil non installé"}
    try:
        proc = psutil.Process(pid)
        cpu = proc.cpu_percent(interval=1.0)
        mem = proc.memory_info()
        return {
            "cpu_percent": cpu,
            "rss_mb": round(mem.rss / (1024 * 1024), 2),
            "vms_mb": round(mem.vms / (1024 * 1024), 2),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        return {"error": str(e)}


def benchmark_tier1(client: httpx.Client, base_url: str, n: int = 500) -> dict:
    """
    Mesure la latence du Tier 1 (Autoencoder).
    Envoie des événements normaux et collecte tier1_latency_ms.
    """
    latencies = []
    for i in range(n):
        syscalls = ["read", "write", "close", "epoll_wait", "recvfrom",
                     "sendto", "futex", "openat", "stat", "fstat"]
        payload = {
            "output": "Normal traffic benchmark",
            "output_fields": {
                "container.id": f"bench-{i % 10}",
                "evt.type": syscalls[i % len(syscalls)],
            },
        }
        resp = client.post(f"{base_url}/falco-webhook", json=payload)
        if resp.status_code == 200:
            body = resp.json()
            t1_ms = body.get("tier1_latency_ms")
            if t1_ms is not None:
                latencies.append(t1_ms)

    if not latencies:
        return {"error": "Aucune latence collectée"}

    return {
        "samples": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
    }


def benchmark_e2e(client: httpx.Client, base_url: str, n: int = 100) -> dict:
    """
    Mesure la latence end-to-end (incluant le réseau HTTP local).
    """
    latencies = []
    syscalls = ["read", "write", "close", "epoll_wait", "recvfrom"]
    for i in range(n):
        payload = {
            "output": "E2E benchmark",
            "output_fields": {
                "container.id": f"e2e-bench-{i % 5}",
                "evt.type": syscalls[i % len(syscalls)],
            },
        }
        t0 = time.perf_counter()
        resp = client.post(f"{base_url}/falco-webhook", json=payload)
        t1 = time.perf_counter()
        if resp.status_code == 200:
            latencies.append((t1 - t0) * 1000)

    if not latencies:
        return {"error": "Aucune latence collectée"}

    return {
        "samples": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Mesure overhead SysGuard-AI")
    parser.add_argument("--agent-url", default="http://localhost:8000")
    parser.add_argument("--pid", type=int, default=0, help="PID du processus agent")
    parser.add_argument("--requests", type=int, default=500)
    args = parser.parse_args()

    client = httpx.Client(timeout=10.0)

    print("=" * 60)
    print("  SysGuard-AI — Mesure d'overhead")
    print("=" * 60)

    print("\n[1/4] Mesure baseline (avant charge)...")
    baseline = measure_process(args.pid) if args.pid else {}
    if baseline:
        print(f"  CPU: {baseline.get('cpu_percent', 'N/A')}%, "
              f"RSS: {baseline.get('rss_mb', 'N/A')} Mo")

    print(f"\n[2/4] Benchmark Tier 1 — Autoencoder ({args.requests} requêtes)...")
    tier1_stats = benchmark_tier1(client, args.agent_url, args.requests)
    if "mean_ms" in tier1_stats:
        print(f"  Latence moyenne  : {tier1_stats['mean_ms']} ms")
        print(f"  Latence médiane  : {tier1_stats['median_ms']} ms")
        print(f"  Latence P95      : {tier1_stats['p95_ms']} ms")
        print(f"  Latence P99      : {tier1_stats['p99_ms']} ms")

    print(f"\n[3/4] Benchmark end-to-end (100 requêtes)...")
    e2e_stats = benchmark_e2e(client, args.agent_url, 100)
    if "mean_ms" in e2e_stats:
        print(f"  Latence E2E moyenne : {e2e_stats['mean_ms']} ms")
        print(f"  Latence E2E P95     : {e2e_stats['p95_ms']} ms")

    print("\n[4/4] Mesure sous charge...")
    under_load = measure_process(args.pid) if args.pid else {}
    if under_load:
        print(f"  CPU: {under_load.get('cpu_percent', 'N/A')}%, "
              f"RSS: {under_load.get('rss_mb', 'N/A')} Mo")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": baseline,
        "under_load": under_load,
        "tier1_latency": tier1_stats,
        "e2e_latency": e2e_stats,
        "total_requests": args.requests,
    }

    if baseline.get("cpu_percent") is not None and under_load.get("cpu_percent") is not None:
        report["cpu_overhead_percent"] = round(
            under_load["cpu_percent"] - baseline["cpu_percent"], 2
        )

    report_path = os.path.join(os.path.dirname(__file__), "overhead_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Rapport enregistré : {report_path}")
    client.close()


if __name__ == "__main__":
    main()
