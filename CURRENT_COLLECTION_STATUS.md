# SysGuard-AI Status

Current mode: collection was started, but it did not finish cleanly.

Done:

- Minikube installed and cluster running.
- Falco + Falco Sidekick deployed.
- RBAC and DVWA deployed.
- Docker image built and agent deployed.
- Falco Sidekick switched from sysguard-agent to sysguard-collector.
- Collector deployed with PVC storage.

Current state:

- The collector pod restarted several times.
- The current CSV contains only the latest run, not the original full 12h capture.
- Current snapshot observed: 337 lines in /data/normal_traffic.csv, so 336 vectors + header.

Why it matters:

- The CSV is stored on a PVC, but the collector recreates the file on startup.
- A pod restart therefore overwrites previous collected data.

Useful checks:

- kubectl exec deployment/sysguard-collector -n default -- curl -s http://127.0.0.1:8001/stats
- kubectl exec deployment/sysguard-collector -n default -- sh -c 'wc -l /data/normal_traffic.csv && ls -lh /data/normal_traffic.csv'

Recommended next step:

- Do not train yet if you need the intended 12h dataset.
- First fix the collector restart issue, then rerun the 12h capture.
- After a clean run, copy the CSV locally and run: python ai_model/train_autoencoder.py --csv ai_model/dataset/normal_traffic.csv
