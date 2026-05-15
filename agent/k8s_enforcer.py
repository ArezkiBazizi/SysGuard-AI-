"""
Module de remédiation Kubernetes pour SysGuard-AI.

Stratégie de réponse graduée et réversible :
  - quarantine_pod()    : NetworkPolicy deny-all (réversible, automatique)
  - lift_quarantine()   : Suppression de la NetworkPolicy (réversible, automatique)
  - kill_pod()          : Suppression du pod (irréversible, JAMAIS automatique)

Principe fondamental : un système autonome ne doit jamais prendre
d'action irréversible. Seul l'opérateur humain peut décider de
supprimer un pod (Human-in-the-Loop).
"""

import os
import time
import logging
from typing import Optional

from kubernetes import client, config
from kubernetes.client import ApiException

logger = logging.getLogger("sysguard.enforcer")

_core_v1: Optional[client.CoreV1Api] = None
_networking_v1: Optional[client.NetworkingV1Api] = None

QUARANTINE_LABEL = "sysguard-quarantine"
QUARANTINE_POLICY_PREFIX = "sysguard-quarantine-"


def _get_core_v1() -> Optional[client.CoreV1Api]:
    global _core_v1
    if _core_v1 is not None:
        return _core_v1
    try:
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        _core_v1 = client.CoreV1Api()
        return _core_v1
    except Exception as e:
        logger.warning("Impossible de se connecter au cluster K8s: %s", e)
        return None


def _get_networking_v1() -> Optional[client.NetworkingV1Api]:
    global _networking_v1
    if _networking_v1 is not None:
        return _networking_v1
    try:
        _get_core_v1()
        _networking_v1 = client.NetworkingV1Api()
        return _networking_v1
    except Exception as e:
        logger.warning("Impossible d'initialiser NetworkingV1Api: %s", e)
        return None


def _find_pod_by_container_id(
    api: client.CoreV1Api, container_id: str, namespace: str
) -> Optional[client.V1Pod]:
    """
    Recherche un pod contenant le container_id donné.
    Pour un POC, on parcourt les pods du namespace.
    """
    try:
        pods = api.list_namespaced_pod(namespace=namespace)
    except ApiException as e:
        logger.error("Erreur lors du listage des pods: %s", e)
        return None

    for pod in pods.items:
        statuses = (pod.status.container_statuses or []) + \
                   (pod.status.init_container_statuses or [])
        for status in statuses:
            cid = status.container_id or ""
            if not cid:
                continue
            if container_id in cid:
                return pod
    return None


def _resolve_pod(container_id: str) -> tuple:
    """Résout le container_id en (api, pod, namespace) ou lève une exception."""
    if not container_id:
        raise ValueError("Aucun container_id fourni")

    api = _get_core_v1()
    if api is None:
        raise ConnectionError(
            "Pas de cluster Kubernetes (Minikube arrêté ou kubeconfig absent)"
        )

    namespace = os.getenv("K8S_NAMESPACE", "default")
    pod = _find_pod_by_container_id(api, container_id, namespace=namespace)
    if pod is None:
        raise LookupError(f"Aucun pod trouvé pour container_id={container_id}")

    return api, pod, namespace


def quarantine_pod(container_id: str) -> dict:
    """
    Place le pod en quarantaine réseau via une NetworkPolicy deny-all.

    Action RÉVERSIBLE : le pod reste vivant, ses processus continuent,
    les preuves forensiques sont préservées. Seul le trafic réseau
    (ingress et egress) est bloqué.

    C'est la SEULE action automatique du système.

    Returns:
        dict avec les clés: success, pod_name, action, message
    """
    dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    try:
        api, pod, namespace = _resolve_pod(container_id)
    except (ValueError, ConnectionError, LookupError) as e:
        logger.warning("[Quarantine] %s", e)
        return {"success": False, "action": "quarantine", "message": str(e)}

    pod_name = pod.metadata.name
    net_api = _get_networking_v1()
    if net_api is None:
        msg = "NetworkingV1Api indisponible"
        logger.error("[Quarantine] %s", msg)
        return {"success": False, "pod_name": pod_name, "action": "quarantine", "message": msg}

    try:
        api.patch_namespaced_pod(
            name=pod_name,
            namespace=namespace,
            body={"metadata": {"labels": {
                QUARANTINE_LABEL: "true",
                "sysguard-quarantine-time": str(int(time.time())),
            }}},
        )
    except ApiException as e:
        msg = f"Erreur lors du label de quarantaine: {e}"
        logger.error("[Quarantine] %s", msg)
        return {"success": False, "pod_name": pod_name, "action": "quarantine", "message": msg}

    policy_name = f"{QUARANTINE_POLICY_PREFIX}{pod_name}"
    policy = client.V1NetworkPolicy(
        metadata=client.V1ObjectMeta(name=policy_name, namespace=namespace),
        spec=client.V1NetworkPolicySpec(
            pod_selector=client.V1LabelSelector(
                match_labels={QUARANTINE_LABEL: "true"},
            ),
            policy_types=["Ingress", "Egress"],
            ingress=[],
            egress=[],
        ),
    )

    if dry_run:
        msg = f"DRY_RUN: NetworkPolicy {policy_name} non appliquée"
        logger.info("[Quarantine] %s", msg)
        return {"success": True, "pod_name": pod_name, "action": "quarantine_dry_run", "message": msg}

    try:
        net_api.create_namespaced_network_policy(namespace=namespace, body=policy)
        msg = f"Pod {pod_name} mis en quarantaine (NetworkPolicy deny-all)"
        logger.info("[Quarantine] %s", msg)
        return {"success": True, "pod_name": pod_name, "action": "quarantine", "message": msg}
    except ApiException as e:
        if e.status == 409:
            msg = f"NetworkPolicy {policy_name} existe déjà (pod déjà en quarantaine)"
            logger.info("[Quarantine] %s", msg)
            return {"success": True, "pod_name": pod_name, "action": "quarantine_exists", "message": msg}
        msg = f"Erreur lors de la création de la NetworkPolicy: {e}"
        logger.error("[Quarantine] %s", msg)
        return {"success": False, "pod_name": pod_name, "action": "quarantine", "message": msg}


def lift_quarantine(container_id: str) -> dict:
    """
    Lève la quarantaine réseau d'un pod.

    Appelée par le Tier 2 (LLM) lorsqu'un faux positif est identifié.
    Supprime la NetworkPolicy et retire le label de quarantaine.

    Returns:
        dict avec les clés: success, pod_name, action, message
    """
    dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    try:
        api, pod, namespace = _resolve_pod(container_id)
    except (ValueError, ConnectionError, LookupError) as e:
        logger.warning("[LiftQuarantine] %s", e)
        return {"success": False, "action": "lift_quarantine", "message": str(e)}

    pod_name = pod.metadata.name
    net_api = _get_networking_v1()
    if net_api is None:
        msg = "NetworkingV1Api indisponible"
        logger.error("[LiftQuarantine] %s", msg)
        return {"success": False, "pod_name": pod_name, "action": "lift_quarantine", "message": msg}

    policy_name = f"{QUARANTINE_POLICY_PREFIX}{pod_name}"

    if dry_run:
        msg = f"DRY_RUN: Levée de quarantaine simulée pour {pod_name}"
        logger.info("[LiftQuarantine] %s", msg)
        return {"success": True, "pod_name": pod_name, "action": "lift_quarantine_dry_run", "message": msg}

    try:
        net_api.delete_namespaced_network_policy(name=policy_name, namespace=namespace)
        logger.info("[LiftQuarantine] NetworkPolicy %s supprimée", policy_name)
    except ApiException as e:
        if e.status == 404:
            logger.info("[LiftQuarantine] NetworkPolicy %s n'existait pas", policy_name)
        else:
            msg = f"Erreur lors de la suppression de la NetworkPolicy: {e}"
            logger.error("[LiftQuarantine] %s", msg)
            return {"success": False, "pod_name": pod_name, "action": "lift_quarantine", "message": msg}

    try:
        api.patch_namespaced_pod(
            name=pod_name,
            namespace=namespace,
            body={"metadata": {"labels": {
                QUARANTINE_LABEL: None,
                "sysguard-quarantine-time": None,
            }}},
        )
    except ApiException as e:
        logger.warning("[LiftQuarantine] Erreur lors du retrait du label: %s", e)

    msg = f"Quarantaine levée pour {pod_name} — conteneur reprend son fonctionnement normal"
    logger.info("[LiftQuarantine] %s", msg)
    return {"success": True, "pod_name": pod_name, "action": "lift_quarantine", "message": msg}


def kill_pod(container_id: str) -> dict:
    """
    Supprime le pod contenant le conteneur identifié.

    ⚠️  ACTION IRRÉVERSIBLE — NE DOIT JAMAIS ÊTRE APPELÉE AUTOMATIQUEMENT.

    Cette fonction existe uniquement pour être invoquée manuellement
    par un opérateur humain via l'API d'administration, ou en mode DRY_RUN
    pour la démonstration.

    Returns:
        dict avec les clés: success, pod_name, action, message
    """
    dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    try:
        api, pod, namespace = _resolve_pod(container_id)
    except (ValueError, ConnectionError, LookupError) as e:
        logger.warning("[KillPod] %s", e)
        return {"success": False, "action": "kill_pod", "message": str(e)}

    pod_name = pod.metadata.name

    if dry_run:
        msg = f"DRY_RUN: Pod {pod_name} non supprimé"
        logger.info("[KillPod] %s", msg)
        return {"success": True, "pod_name": pod_name, "action": "kill_pod_dry_run", "message": msg}

    try:
        api.delete_namespaced_pod(name=pod_name, namespace=namespace)
        msg = f"Pod {pod_name} supprimé (action manuelle opérateur)"
        logger.info("[KillPod] %s", msg)
        return {"success": True, "pod_name": pod_name, "action": "kill_pod", "message": msg}
    except ApiException as e:
        msg = f"Erreur lors de la suppression du pod {pod_name}: {e}"
        logger.error("[KillPod] %s", msg)
        return {"success": False, "pod_name": pod_name, "action": "kill_pod", "message": msg}
