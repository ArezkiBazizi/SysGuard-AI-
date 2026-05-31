"""
Collecteur de données Falco pour l'entraînement de l'Autoencoder.

Lance un serveur FastAPI temporaire qui reçoit les webhooks Falco,
agrège les syscalls en histogrammes par conteneur (fenêtre 10s),
et enregistre chaque vecteur dans un CSV.

Le CSV produit est directement utilisable par train_autoencoder.py --csv.

Usage :
  # 1. Configurer Falco Sidekick pour envoyer vers http://<IP>:8001/collect
  # 2. Lancer le collecteur :
  python ai_model/collect_falco_data.py --duration 43200 --output ai_model/dataset/normal_traffic.csv
  #    (43200s = 12 heures)
  # 3. Entraîner le modèle :
  python ai_model/train_autoencoder.py --csv ai_model/dataset/normal_traffic.csv

Notes :
  - Pendant la collecte, ne générer QUE du trafic normal (pas d'attaques).
  - La durée recommandée est de 12h minimum pour un POC,
    24h+ pour un environnement de production.
"""

import argparse
import csv
import os
import signal
import sys
import time
import threading
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.preprocessor import INPUT_DIM, SYSCALL_TABLE, REVERSE_SYSCALL_TABLE

WINDOW_SECONDS = 10

_OTHER_INDEX = INPUT_DIM - 1


class FalcoCollector:
    """
    Agrège les événements Falco en histogrammes de syscalls
    et les écrit dans un fichier CSV.
    """

    def __init__(self, output_path: str, window_seconds: int = WINDOW_SECONDS):
        self.output_path = output_path
        self.window_seconds = window_seconds
        self._buffers: Dict[str, Dict] = {}
        self._vectors_written = 0
        self._events_received = 0
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        header = [REVERSE_SYSCALL_TABLE.get(i, f"syscall_{i}") for i in range(INPUT_DIM)]

        # Mode reprise : si le fichier existe avec le bon header, on ajoute
        # a la suite sans ecraser les vecteurs deja collectes.
        # C'est le comportement critique pour survivre aux redemarrages de pod.
        resumed = False
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                with open(output_path, "r", newline="") as f:
                    existing_header = next(csv.reader(f), None)
                if existing_header == header:
                    # Compter les vecteurs deja ecrits (lignes - 1 pour l'entete)
                    with open(output_path, "r", newline="") as f:
                        line_count = sum(1 for _ in f)
                    self._vectors_written = max(0, line_count - 1)
                    resumed = True
            except Exception:
                pass

        if not resumed:
            # Nouveau fichier : ecrire l'entete
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

        mode = "REPRISE" if resumed else "NOUVEAU"
        print(f"[Collector] Mode           : {mode}")
        print(f"[Collector] Fichier        : {output_path}")
        if resumed:
            print(f"[Collector] Vecteurs deja collectes : {self._vectors_written}")
        print(f"[Collector] Fenetre tumbling : {window_seconds}s")
        print(f"[Collector] Dimension du vecteur : {INPUT_DIM}")

    def _get_syscall_index(self, syscall_type: Optional[str]) -> int:
        if not syscall_type:
            return _OTHER_INDEX
        return SYSCALL_TABLE.get(syscall_type, _OTHER_INDEX)

    def process_event(self, container_id: str, syscall_type: Optional[str]) -> None:
        if not container_id:
            return

        with self._lock:
            self._events_received += 1
            now = time.time()
            buf = self._buffers.get(container_id)

            if buf is None:
                buf = {
                    "counts": np.zeros(INPUT_DIM, dtype=np.float32),
                    "window_start": now,
                }
                self._buffers[container_id] = buf

            if now - buf["window_start"] >= self.window_seconds:
                self._flush_buffer(container_id, buf)
                buf["counts"][:] = 0.0
                buf["window_start"] = now

            idx = self._get_syscall_index(syscall_type)
            buf["counts"][idx] += 1.0

    def _flush_buffer(self, container_id: str, buf: dict) -> None:
        counts = buf["counts"]
        total = float(np.sum(counts))
        if total == 0:
            return

        normalized = counts / total
        with open(self.output_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{v:.8f}" for v in normalized])
        self._vectors_written += 1

    def flush_all(self) -> None:
        with self._lock:
            for cid, buf in list(self._buffers.items()):
                self._flush_buffer(cid, buf)
            self._buffers.clear()

    def get_stats(self) -> dict:
        return {
            "events_received": self._events_received,
            "vectors_written": self._vectors_written,
            "active_containers": len(self._buffers),
            "output_path": self.output_path,
        }


def _timestamped_output_path(output_path: str) -> str:
    output_dir = os.path.dirname(output_path) or "."
    base_name = os.path.basename(output_path)
    stem, ext = os.path.splitext(base_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ext = ext or ".csv"

    candidate = os.path.join(output_dir, f"{stem}_{timestamp}{ext}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(output_dir, f"{stem}_{timestamp}_{counter}{ext}")
        counter += 1
    return candidate


def _resolve_output_path(output_path: str, timestamp_output: bool) -> str:
    if timestamp_output or os.path.exists(output_path):
        return _timestamped_output_path(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Collecteur de données Falco pour SysGuard-AI")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--duration", type=int, default=43200,
                        help="Durée de collecte en secondes (défaut: 43200 = 12h)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__), "dataset", "normal_traffic.csv"))
    parser.add_argument(
        "--timestamp-output",
        action="store_true",
        help="Ajoute un timestamp au nom du CSV pour éviter tout écrasement",
    )
    args = parser.parse_args()

    output_path = _resolve_output_path(args.output, args.timestamp_output)
    if output_path != args.output:
        print(f"[Collector] Nouveau fichier de sortie : {output_path}")

    collector = FalcoCollector(output_path)

    from fastapi import FastAPI, Request
    import uvicorn

    collect_app = FastAPI(title="SysGuard-AI Data Collector")

    @collect_app.post("/collect")
    async def collect_event(request: Request):
        data = await request.json()
        fields = data.get("output_fields") or {}
        container_id = fields.get("container.id") or fields.get("container.image.id") or ""
        syscall_type = fields.get("evt.type")
        collector.process_event(container_id, syscall_type)
        return {"status": "collected"}

    @collect_app.get("/stats")
    async def stats():
        return collector.get_stats()

    start_time = time.time()
    print(f"[Collector] Démarrage de la collecte pour {args.duration}s ({args.duration / 3600:.1f}h)")
    print(f"[Collector] Endpoint : http://{args.host}:{args.port}/collect")
    print(f"[Collector] Configurer Falco Sidekick webhook vers cette URL")

    def shutdown_timer():
        time.sleep(args.duration)
        elapsed = time.time() - start_time
        print(f"\n[Collector] Durée atteinte ({elapsed / 3600:.1f}h). Arrêt...")
        collector.flush_all()
        stats = collector.get_stats()
        print(f"[Collector] Résumé final :")
        print(f"  - Événements reçus  : {stats['events_received']}")
        print(f"  - Vecteurs écrits   : {stats['vectors_written']}")
        print(f"  - Fichier de sortie : {args.output}")
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Thread(target=shutdown_timer, daemon=True)
    timer.start()

    def handle_sigterm(*_):
        collector.flush_all()
        stats = collector.get_stats()
        print(f"\n[Collector] Arrêt — {stats['vectors_written']} vecteurs enregistrés")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    status_interval = 60

    def status_printer():
        while True:
            time.sleep(status_interval)
            stats = collector.get_stats()
            elapsed = time.time() - start_time
            print(
                f"[Collector] {elapsed / 60:.0f}min — "
                f"{stats['events_received']} events, "
                f"{stats['vectors_written']} vecteurs, "
                f"{stats['active_containers']} conteneurs actifs"
            )

    status_thread = threading.Thread(target=status_printer, daemon=True)
    status_thread.start()

    uvicorn.run(collect_app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
