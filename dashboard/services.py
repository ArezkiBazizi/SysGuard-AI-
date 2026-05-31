"""Couche service pour le dashboard Streamlit SysGuard-AI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AI_MODEL = ROOT / "ai_model"
INCIDENT_DIR = AI_MODEL / "incident_reports"

sys.path.insert(0, str(AI_MODEL))

from test_intrusion import (  # noqa: E402
    SCENARIO_CATALOG,
    load_thresholds,
    run_scenario_data,
)
from evaluate_model import run_evaluation  # noqa: E402
from benchmark_overhead import run_benchmark  # noqa: E402
from generate_incident_report import SCENARIOS, generate_scenario_report  # noqa: E402

MODEL_PATH = AI_MODEL / "saved_model.pth"
THRESHOLD_PATH = AI_MODEL / "threshold.json"
EVAL_REPORT_PATH = AI_MODEL / "evaluation_report.json"
BENCHMARK_PATH = AI_MODEL / "benchmark_results.json"
COMPARISON_PATH = AI_MODEL / "comparison_report.json"


def model_ready() -> bool:
    return MODEL_PATH.exists() and THRESHOLD_PATH.exists()


def load_threshold_info() -> Dict[str, Any]:
    if not THRESHOLD_PATH.exists():
        return {}
    with open(THRESHOLD_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_scenario_options() -> List[tuple[str, str]]:
    return [(key, meta["label"]) for key, meta in SCENARIO_CATALOG.items()]


def get_scenario_description(scenario_key: str) -> str:
    return SCENARIO_CATALOG[scenario_key]["description"]


def _load_model_safe():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    from autoencoder import SysGuardAutoencoder, INPUT_DIM
    import torch
    model = SysGuardAutoencoder(INPUT_DIM, dropout=0.0)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def run_attack_simulation(
    scenario_key: str,
    n_windows: int = 10,
    seed: int = 42,
    use_llm: bool = True,
) -> Dict[str, Any]:
    meta = SCENARIO_CATALOG[scenario_key]
    model = _load_model_safe()
    thresholds = load_thresholds()
    rng = np.random.default_rng(seed)
    vectors = meta["generator"](rng, n=n_windows)
    llm = use_llm and meta.get("use_llm", True)
    return run_scenario_data(
        model,
        thresholds["alpha"],
        thresholds["beta"],
        vectors,
        meta["label"],
        use_llm=llm,
    )


def run_custom_injection(syscall_counts: dict) -> Dict[str, Any]:
    """
    Construit un vecteur syscall à partir des compteurs fournis,
    le passe au modèle et retourne la MSE + verdict.
    Utilisé par la page 'Injection custom' (vrai test d'intrusion).
    """
    import torch
    from autoencoder import SysGuardAutoencoder, INPUT_DIM  # noqa: F401

    with open(THRESHOLD_PATH, encoding="utf-8") as f:
        th = json.load(f)
    alpha = th["alpha"]
    beta = th["beta"]

    # Mapping syscall name -> index (extrait de test_intrusion.py SC dict)
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
        "openat": 250, "ptrace": 65, "keyctl": 219, "setuid": 70,
        "setgid": 71, "capset": 125, "chown": 73, "process_vm_readv": 310,
    }

    v = np.zeros(INPUT_DIM, dtype=np.float32)
    for sc_name, count in syscall_counts.items():
        if sc_name in SC:
            v[SC[sc_name]] = float(count)

    total = v.sum()
    if total > 0:
        v_norm = v / total
    else:
        v_norm = v.copy()

    model = _load_model_safe()
    t = torch.tensor(v_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        recon = model(t)
    mse = float(torch.mean((t - recon) ** 2).item())

    if mse >= beta:
        verdict = "CRITIQUE"
    elif mse >= alpha:
        verdict = "ANOMALIE"
    else:
        verdict = "NORMAL"

    return {
        "mse": mse,
        "verdict": verdict,
        "alpha": alpha,
        "beta": beta,
        "total_syscalls": int(total),
        "active_dims": int((v > 0).sum()),
    }


def run_full_evaluation(n_test: int = 200) -> Dict[str, Any]:
    return run_evaluation(
        str(MODEL_PATH),
        str(THRESHOLD_PATH),
        n_test,
        report_path=str(EVAL_REPORT_PATH),
    )


def run_performance_benchmark() -> Dict[str, Any]:
    return run_benchmark(str(MODEL_PATH), str(BENCHMARK_PATH))


def list_incident_reports() -> List[Dict[str, Any]]:
    if not INCIDENT_DIR.exists():
        return []
    reports = []
    for json_file in sorted(INCIDENT_DIR.glob("incident_*.json")):
        data = load_json(json_file) or {}
        meta = data.get("meta", {})
        report = data.get("report", {})
        txt_path = json_file.with_suffix(".txt")
        reports.append({
            "id": meta.get("id", json_file.stem),
            "name": meta.get("name", json_file.stem),
            "json_path": str(json_file),
            "txt_path": str(txt_path) if txt_path.exists() else None,
            "verdict": report.get("verdict", "—"),
            "attack_type": report.get("attack_type", "—"),
            "latency_ms": data.get("latency_ms"),
            "data": data,
        })
    return reports


def read_text_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def generate_incident(scenario_id: int, api_key: Optional[str] = None) -> Dict[str, Any]:
    return generate_scenario_report(scenario_id, api_key=api_key)


def incident_scenario_options() -> List[tuple[int, str]]:
    return [(sid, f"{meta['id']} — {meta['name']}") for sid, meta in SCENARIOS.items()]
