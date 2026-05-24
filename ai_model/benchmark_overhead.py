"""
Benchmark de latence et d'overhead CPU — SysGuard-AI Tier 1

Mesure trois metriques :
  1. Latence d'inference PyTorch (ns / ms) — statistiques P50/P95/P99
  2. Overhead CPU de l'inference (% CPU) via psutil
  3. Empreinte memoire du modele charge

Usage :
  cd ai_model
  python benchmark_overhead.py

Les resultats sont ecrits dans benchmark_results.json
(a inclure dans le memoire).
"""

import json
import os
import sys
import time

import numpy as np
import psutil
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoencoder import SysGuardAutoencoder, INPUT_DIM

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(SCRIPT_DIR, "saved_model.pth")
OUTPUT_PATH   = os.path.join(SCRIPT_DIR, "benchmark_results.json")

N_WARMUP      = 200    # inferences de chauffe (JIT PyTorch)
N_BENCH       = 2000   # inferences mesurees
N_CPU_SAMPLES = 500    # echantillons pour l'overhead CPU


def load_model():
    model = SysGuardAutoencoder(INPUT_DIM, dropout=0.0)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def random_vector(rng):
    """Vecteur d'histogramme normalise L1 aleatoire."""
    v = rng.exponential(1.0, size=INPUT_DIM).astype(np.float32)
    v /= v.sum()
    return torch.tensor(v).unsqueeze(0)


def bench_latency(model, rng):
    """Mesure la latence forward-pass sur N_BENCH repetitions."""
    # Chauffe
    for _ in range(N_WARMUP):
        with torch.no_grad():
            model(random_vector(rng))

    # Mesure
    latencies_ns = []
    for _ in range(N_BENCH):
        x = random_vector(rng)
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            model(x)
        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)

    arr = np.array(latencies_ns)
    return {
        "unit": "microseconds",
        "n_samples": N_BENCH,
        "mean_us":   round(float(np.mean(arr)) / 1000, 2),
        "median_us": round(float(np.median(arr)) / 1000, 2),
        "p50_us":    round(float(np.percentile(arr, 50)) / 1000, 2),
        "p95_us":    round(float(np.percentile(arr, 95)) / 1000, 2),
        "p99_us":    round(float(np.percentile(arr, 99)) / 1000, 2),
        "max_us":    round(float(np.max(arr)) / 1000, 2),
        "min_us":    round(float(np.min(arr)) / 1000, 2),
    }


def bench_cpu_overhead(model, rng, lat_median_us):
    """
    Calcul de l'overhead CPU par la methode analytique :

      overhead = (temps_inference / temps_fenetre) * 100

    Avec rate-limiting a 1 inference/seconde par conteneur,
    l'overhead est : lat_median_us / 1_000_000 * 100
    Sans rate-limiting (1 inference/fenetre de 10s) :
      overhead = lat_median_us / 10_000_000 * 100

    On mesure aussi le CPU% instantane du processus pendant
    une rafale d'inferences (burst) pour estimer le pic.
    """
    n_cpu = psutil.cpu_count(logical=True) or 1
    proc  = psutil.Process(os.getpid())

    # Mesure CPU pendant une rafale d'inferences sur 2 secondes
    proc.cpu_percent(interval=None)  # init
    time.sleep(0.5)

    t_start = time.perf_counter()
    n_inf   = 0
    while time.perf_counter() - t_start < 2.0:
        x = random_vector(rng)
        with torch.no_grad():
            model(x)
        n_inf += 1
    burst_cpu = proc.cpu_percent(interval=None)  # % sur 1 coeur

    # Normalise sur l'ensemble des coeurs (valeur noeud)
    burst_cpu_node = burst_cpu / n_cpu

    # Overhead analytique (mode production : 1 inf / 10s window)
    overhead_per_window = (lat_median_us / 10_000_000) * 100
    # Mode rate-limited (1 inf / s)
    overhead_rate_limited = (lat_median_us / 1_000_000) * 100

    return {
        "method": "analytique + burst",
        "burst_cpu_single_core_pct":  round(burst_cpu,            2),
        "burst_cpu_node_pct":         round(burst_cpu_node,       3),
        "overhead_per_window_pct":    round(overhead_per_window,  4),
        "overhead_rate_limited_pct":  round(overhead_rate_limited, 4),
        "n_cpu_logical":              n_cpu,
        "burst_inferences_per_sec":   round(n_inf / 2.0, 0),
        "note": (
            f"Burst = rafale continue ({int(n_inf/2)}/s sur 1 coeur). "
            f"Mode production (rate-limit 1 inf/s) : overhead={overhead_rate_limited:.3f}%. "
            f"Mode 1 inf/fenetre-10s : overhead={overhead_per_window:.4f}%."
        ),
    }


def bench_memory(model):
    """Empreinte memoire du modele en RAM (parametres + buffers)."""
    param_bytes = sum(p.nbytes for p in model.parameters())
    buf_bytes   = sum(b.nbytes for b in model.buffers())
    total_bytes = param_bytes + buf_bytes

    # Memoire du processus (RSS) apres chargement
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 ** 2)

    return {
        "model_params_kb":    round(param_bytes / 1024, 1),
        "model_buffers_kb":   round(buf_bytes   / 1024, 1),
        "model_total_kb":     round(total_bytes / 1024, 1),
        "process_rss_mb":     round(rss_mb, 1),
        "n_parameters":       sum(p.numel() for p in model.parameters()),
    }


def run_benchmark(model_path=None, output_path=None, n_warmup=N_WARMUP, n_bench=N_BENCH):
    """Execute le benchmark et retourne les resultats (optionnellement sauvegardes en JSON)."""
    model_path = model_path or MODEL_PATH
    output_path = output_path or OUTPUT_PATH
    rng = np.random.default_rng(42)

    model = SysGuardAutoencoder(INPUT_DIM, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    lat = bench_latency(model, rng)
    cpu = bench_cpu_overhead(model, rng, lat["median_us"])
    mem = bench_memory(model)

    results = {
        "model_path": model_path,
        "input_dim": INPUT_DIM,
        "latency": lat,
        "cpu_overhead": cpu,
        "memory": mem,
        "system": {
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_phys": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    print("=" * 56)
    print("  SysGuard-AI — Benchmark Overhead Tier 1")
    print("=" * 56)

    print(f"\n[1/3] Chargement du modele depuis {MODEL_PATH} ...")
    results = run_benchmark(MODEL_PATH, OUTPUT_PATH)
    lat = results["latency"]
    cpu = results["cpu_overhead"]
    mem = results["memory"]
    n_params = mem["n_parameters"]
    print(f"      Modele charge : {n_params:,} parametres")

    print(f"\n[2/3] Benchmark latence ({N_BENCH} inferences) ...")
    print(f"      Latence P50  : {lat['p50_us']:.1f} µs")
    print(f"      Latence P95  : {lat['p95_us']:.1f} µs")
    print(f"      Latence P99  : {lat['p99_us']:.1f} µs")
    print(f"      Latence max  : {lat['max_us']:.1f} µs")

    print(f"\n[3/3] Benchmark overhead CPU (burst 2s + calcul analytique) ...")
    print(f"      Burst inferences/s    : {int(cpu['burst_inferences_per_sec'])}")
    print(f"      CPU burst (1 coeur)   : {cpu['burst_cpu_single_core_pct']:.1f}%")
    print(f"      CPU burst (noeud)     : {cpu['burst_cpu_node_pct']:.2f}%  ({cpu['n_cpu_logical']} coeurs)")
    print(f"      Overhead prod (1inf/s): {cpu['overhead_rate_limited_pct']:.4f}%")
    print(f"      Overhead prod (1/10s) : {cpu['overhead_per_window_pct']:.5f}%")

    print(f"\n      Modele RAM    : {mem['model_total_kb']:.1f} Ko")
    print(f"      Processus RSS : {mem['process_rss_mb']:.1f} Mo")

    print(f"\n  Rapport sauvegarde : {OUTPUT_PATH}")
    print()
    print("  RESUME POUR LE MEMOIRE :")
    print(f"  Latence inference  : {lat['p50_us']:.1f} µs (mediane), "
          f"{lat['p99_us']:.1f} µs (P99)")
    print(f"  Overhead CPU (prod): {cpu['overhead_rate_limited_pct']:.4f}% (rate-limited 1 inf/s)")
    print(f"  RAM modele         : {mem['model_total_kb']:.1f} Ko")
    print(f"  RAM processus      : {mem['process_rss_mb']:.1f} Mo")
    print()


if __name__ == "__main__":
    main()
