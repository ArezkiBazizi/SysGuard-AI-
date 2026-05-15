"""
Agent SysGuard-AI — Point d'entrée FastAPI.

Architecture Two-Tier à réponse graduée :
  Tier 1 (Sentinelle) : Autoencoder PyTorch — détection en < 15 ms
  Tier 2 (Décideur)   : LLM — arbitre actif (maintien/levée quarantaine)

Deux seuils de décision :
  α (alpha) = P99 de la MSE d'entraînement  → anomalie modérée
  β (beta)  = P99.9 de la MSE d'entraînement → anomalie critique

Principe : aucune action irréversible automatique (Human-in-the-Loop).
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai_model"))
from autoencoder import SysGuardAutoencoder, INPUT_DIM as MODEL_INPUT_DIM

from .preprocessor import (
    INPUT_DIM, update_histogram, get_histogram_vector,
    get_raw_counts, get_top_syscalls,
)
from .k8s_enforcer import quarantine_pod, lift_quarantine, kill_pod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("sysguard.agent")

app = FastAPI(
    title="SysGuard-AI Agent",
    description="Détection d'anomalies syscall et remédiation graduée pour Kubernetes",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Chargement du modèle et des seuils
# ---------------------------------------------------------------------------

def _load_autoencoder_model() -> SysGuardAutoencoder:
    model_path = os.getenv("MODEL_PATH", "ai_model/saved_model.pth")
    logger.info("Chargement du modèle depuis %s", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : '{model_path}'. "
            "Exécutez d'abord : python ai_model/create_dummy_model.py"
        )
    model = SysGuardAutoencoder(INPUT_DIM, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    logger.info("Modele charge (%d parametres)",
                sum(p.numel() for p in model.parameters()))
    return model


def _load_thresholds() -> Dict[str, float]:
    threshold_path = os.getenv("THRESHOLD_PATH", "ai_model/threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            data = json.load(f)
        alpha = data.get("alpha", data.get("threshold", 0.05))
        beta = data.get("beta", alpha * 2.5)
        logger.info("Seuils charges depuis %s : alpha=%.6f, beta=%.6f", threshold_path, alpha, beta)
        return {"alpha": alpha, "beta": beta}

    alpha = float(os.getenv("THRESHOLD_ALPHA", "0.05"))
    beta = float(os.getenv("THRESHOLD_BETA", str(alpha * 2.5)))
    logger.info("Seuils depuis env : alpha=%.6f, beta=%.6f", alpha, beta)
    return {"alpha": alpha, "beta": beta}


MODEL = _load_autoencoder_model()
THRESHOLDS = _load_thresholds()
ALPHA = THRESHOLDS["alpha"]
BETA = THRESHOLDS["beta"]

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

INCIDENT_LOG: List[Dict] = []

# Rate limiting : avec la regle catch-all, chaque syscall genere un webhook.
# Lancer l'inference a chaque evenement (~1000+/s) saturerait le CPU.
# On limite a une inference par seconde et par conteneur.
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "1.0"))
_last_inference: Dict[str, float] = {}


# ---------------------------------------------------------------------------
# Tier 2 — LLM Arbitre Actif
# ---------------------------------------------------------------------------

def _build_llm_prompt(
    container_id: str,
    mse: float,
    severity: str,
    top_syscalls: list,
    event: Dict[str, Any],
) -> str:
    output = event.get("output", "")
    fields = event.get("output_fields") or {}
    rule = event.get("rule", "")
    pod_name = fields.get("k8s.pod.name", fields.get("container.id", container_id))
    namespace = fields.get("k8s.ns.name", "unknown")

    syscall_summary = ", ".join(
        f"{s['syscall']}({s['count']})" for s in top_syscalls[:10]
    )

    return (
        "Tu es un expert en sécurité Kubernetes et en analyse d'anomalies de syscalls.\n"
        "Tu reçois un événement détecté comme anomal par un Autoencoder (Tier 1).\n"
        "Ton rôle est de DÉCIDER s'il s'agit d'une menace réelle ou d'un faux positif.\n\n"
        f"--- CONTEXTE DE L'ANOMALIE ---\n"
        f"Pod/Conteneur   : {pod_name} (namespace: {namespace})\n"
        f"Container ID    : {container_id}\n"
        f"Score MSE       : {mse:.6f}\n"
        f"Sévérité        : {severity}\n"
        f"Règle Falco     : {rule}\n"
        f"Message Falco   : {output}\n"
        f"Top syscalls (fenêtre 10s) : {syscall_summary}\n"
        f"Champs Falco    : {json.dumps(fields, default=str)}\n\n"
        "--- INSTRUCTIONS ---\n"
        "1. Analyse le profil de syscalls et le contexte.\n"
        "2. Détermine s'il s'agit de :\n"
        "   a) MENACE CONFIRMÉE : attaque réelle (reverse shell, crypto-mining, "
        "exfiltration, escalade de privilèges, etc.)\n"
        "   b) FAUX POSITIF : comportement légitime atypique (déploiement, "
        "cron job, mise à jour, migration, etc.)\n\n"
        "3. Réponds OBLIGATOIREMENT dans ce format JSON :\n"
        "{\n"
        '  "verdict": "MENACE" ou "FAUX_POSITIF",\n'
        '  "confidence": 0.0 à 1.0,\n'
        '  "attack_type": "type d\'attaque ou null",\n'
        '  "severity_cvss": "CRITIQUE/ÉLEVÉE/MOYENNE/FAIBLE ou null",\n'
        '  "explanation": "explication concise en français",\n'
        '  "recommendations": ["action 1", "action 2", "action 3"]\n'
        "}\n\n"
        "Réponds UNIQUEMENT avec le JSON, sans texte autour."
    )


async def _call_llm_arbiter(
    container_id: str,
    mse: float,
    severity: str,
    top_syscalls: list,
    event: Dict[str, Any],
) -> Optional[Dict]:
    if not LLM_API_KEY:
        logger.info("Pas de cle API LLM configuree -- Tier 2 desactive")
        return None

    prompt = _build_llm_prompt(container_id, mse, severity, top_syscalls, event)

    try:
        if LLM_PROVIDER.lower() == "huggingface":
            hf_url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
            headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(hf_url, headers=headers, json={"inputs": prompt})
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                text = data[0]["generated_text"]
            else:
                text = str(data)
        else:
            openai_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "Tu es un expert en cybersécurité Kubernetes. Réponds uniquement en JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(openai_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]

        return json.loads(text)

    except json.JSONDecodeError as e:
        logger.warning("Reponse LLM non-JSON: %s -- raw: %s", e, text[:200])
        return {"verdict": "MENACE", "confidence": 0.5, "explanation": text[:500], "parse_error": True}
    except Exception as e:
        logger.error("Erreur lors de l'appel LLM: %s", e)
        return None


# ---------------------------------------------------------------------------
# Rapport d'incident
# ---------------------------------------------------------------------------

def _generate_incident_report(
    container_id: str, mse: float, severity: str, top_syscalls: list,
    llm_verdict: Optional[Dict], remediation_result: dict, event: Dict[str, Any],
) -> Dict:
    fields = event.get("output_fields") or {}
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "incident_id": f"SGA-{int(time.time())}",
        "container_id": container_id,
        "pod_name": remediation_result.get("pod_name", "unknown"),
        "namespace": fields.get("k8s.ns.name", os.getenv("K8S_NAMESPACE", "default")),
        "detection": {
            "tier": "Tier 1 — Autoencoder (Sentinelle)",
            "mse_score": mse,
            "alpha_threshold": ALPHA,
            "beta_threshold": BETA,
            "severity": severity,
            "top_syscalls": top_syscalls[:5],
            "falco_rule": event.get("rule", ""),
        },
        "analysis": {
            "tier": "Tier 2 — LLM (Décideur)",
            "llm_model": LLM_MODEL if LLM_API_KEY else "disabled",
            "verdict": llm_verdict if llm_verdict else "LLM non disponible",
        },
        "remediation": {
            "action_taken": remediation_result.get("action", "none"),
            "details": remediation_result.get("message", ""),
        },
    }
    if llm_verdict and llm_verdict.get("verdict") == "MENACE":
        report["recommendations"] = llm_verdict.get("recommendations", [])
    return report


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "2.0.0",
        "architecture": "Two-Tier (Autoencoder PyTorch + LLM)",
        "thresholds": {"alpha": ALPHA, "beta": BETA},
        "input_dim": INPUT_DIM,
        "llm_enabled": bool(LLM_API_KEY),
        "active_incidents": len(INCIDENT_LOG),
    }


@app.get("/incidents")
async def get_incidents(limit: int = 20):
    return {"incidents": INCIDENT_LOG[-limit:], "total": len(INCIDENT_LOG)}


@app.post("/admin/kill-pod")
async def admin_kill_pod(request: Request):
    """Suppression manuelle — opérateur humain uniquement."""
    data = await request.json()
    cid = data.get("container_id", "")
    if not cid:
        return JSONResponse(status_code=400, content={"error": "container_id requis"})
    result = kill_pod(cid)
    logger.warning("ACTION MANUELLE — kill_pod: %s", result)
    return result


@app.post("/admin/lift-quarantine")
async def admin_lift_quarantine(request: Request):
    """Levée manuelle de quarantaine."""
    data = await request.json()
    cid = data.get("container_id", "")
    if not cid:
        return JSONResponse(status_code=400, content={"error": "container_id requis"})
    return lift_quarantine(cid)


@app.post("/falco-webhook")
async def receive_event(request: Request):
    """
    Pipeline de décision graduée :
      1. Histogramme syscall (tumbling window 10s)
      2. Autoencoder → MSE
      3. MSE < α : Normal | α ≤ MSE < β : Modérée | MSE ≥ β : Critique
      4. Tier 2 LLM : MENACE → maintenir quarantaine | FAUX_POSITIF → lever
    """
    try:
        data = await request.json()
    except Exception as e:
        logger.error("Erreur parsing JSON: %s", e)
        return JSONResponse(status_code=400, content={"error": "invalid_json"})

    t_start = time.perf_counter()

    output_fields = data.get("output_fields") or {}
    container_id = (
        output_fields.get("container.id")
        or output_fields.get("container.image.id")
        or ""
    )
    syscall_type = output_fields.get("evt.type")

    if not container_id:
        return {"status": "ignored", "reason": "no_container_id"}

    # --- Ingestion : toujours mettre a jour l'histogramme ---
    update_histogram(container_id, syscall_type)

    # --- Rate limiting : inference au max 1x/s par conteneur ---
    now = time.time()
    last_inf = _last_inference.get(container_id, 0.0)
    if now - last_inf < INFERENCE_INTERVAL:
        return {
            "status": "ok",
            "verdict": "accumulating",
            "container_id": container_id,
        }
    _last_inference[container_id] = now

    # --- Tier 1 : Inference Autoencoder ---
    vector = get_histogram_vector(container_id)

    with torch.no_grad():
        input_tensor = torch.tensor(vector, dtype=torch.float32)
        reconstruction = MODEL(input_tensor).numpy()

    mse = float(np.mean(np.square(vector - reconstruction)))
    tier1_latency_ms = (time.perf_counter() - t_start) * 1000

    # --- Normal ---
    if mse < ALPHA:
        return {
            "status": "ok",
            "verdict": "normal",
            "mse": round(mse, 8),
            "thresholds": {"alpha": ALPHA, "beta": BETA},
            "container_id": container_id,
            "tier1_latency_ms": round(tier1_latency_ms, 2),
        }

    # --- Anomalie détectée ---
    severity = "CRITIQUE" if mse >= BETA else "MODÉRÉE"
    top_syscalls = get_top_syscalls(container_id, top_n=10)

    logger.warning(
        "ANOMALIE %s sur %s (MSE=%.6f, alpha=%.6f, beta=%.6f, latence=%.1fms)",
        severity, container_id, mse, ALPHA, BETA, tier1_latency_ms,
    )

    # --- Quarantaine immédiate (réversible) ---
    quarantine_result = quarantine_pod(container_id)

    # --- Tier 2 : LLM ---
    t_tier2 = time.perf_counter()
    llm_verdict = await _call_llm_arbiter(container_id, mse, severity, top_syscalls, data)
    tier2_latency_ms = (time.perf_counter() - t_tier2) * 1000

    final_action = "quarantine_maintained"
    if llm_verdict:
        if llm_verdict.get("verdict") == "FAUX_POSITIF":
            lift_quarantine(container_id)
            final_action = "quarantine_lifted"
            logger.info("LLM -> FAUX POSITIF -- quarantaine levee")
        else:
            final_action = "quarantine_maintained_threat_confirmed"
            logger.warning("LLM -> MENACE CONFIRMEE -- quarantaine maintenue")

    # --- Rapport d'incident ---
    report = _generate_incident_report(
        container_id, mse, severity, top_syscalls,
        llm_verdict, quarantine_result, data,
    )
    INCIDENT_LOG.append(report)

    total_ms = (time.perf_counter() - t_start) * 1000
    return {
        "status": "ok",
        "verdict": "anomaly",
        "severity": severity,
        "mse": round(mse, 8),
        "thresholds": {"alpha": ALPHA, "beta": BETA},
        "container_id": container_id,
        "final_action": final_action,
        "tier1_latency_ms": round(tier1_latency_ms, 2),
        "tier2_latency_ms": round(tier2_latency_ms, 2) if llm_verdict else None,
        "total_latency_ms": round(total_ms, 2),
        "llm_verdict": llm_verdict,
        "top_syscalls": top_syscalls[:5],
        "incident_id": report["incident_id"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
