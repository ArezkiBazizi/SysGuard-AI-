# SysGuard-AI

**Cadre autonome de détection d'intrusions et de confinement granulaire dans les environnements conteneurisés via l'analyse comportementale (eBPF & ML).**

Proof of Concept académique — Mémoire de Master 2.

---

## Paradigme : Observe – Learn – Respond

SysGuard-AI implémente un paradigme **"Observe–Learn–Respond"** (inspiré de la boucle OODA de Boyd) pour la sécurité runtime des clusters Kubernetes :

1. **Observe** — Collecte exhaustive des 414 types de syscalls instrumentables (noyau Linux ≥ 5.15) via Falco/eBPF
2. **Learn** — Détection d'anomalies non-supervisée par Autoencoder (architecture en sablier 414→207→100→207→414)
3. **Respond** — Réponse graduée et réversible : quarantaine réseau + arbitrage LLM

### Trois contributions clés

| # | Contribution | vs. État de l'art (KubAnomaly) |
|---|-------------|-------------------------------|
| 1 | **Apprentissage non-supervisé** (Autoencoder) | KubAnomaly utilise un réseau supervisé (labels requis) |
| 2 | **Observation exhaustive** (414 features) | KubAnomaly se limite à 31 features manuellement sélectionnées |
| 3 | **Boucle fermée de réponse graduée** (Two-Tier + quarantaine réversible) | KubAnomaly ne propose aucune remédiation |

---

## Architecture Two-Tier

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLUSTER KUBERNETES (Minikube)                     │
│                                                                     │
│  ┌──────────┐    syscalls    ┌─────────┐    JSON    ┌────────────┐ │
│  │ Pod DVWA │ ──────────►   │  Falco  │ ────────► │  Sidekick  │ │
│  │ (victim) │  eBPF/kernel  │(DaemonSet│  gRPC     │ (webhook)  │ │
│  └──────────┘               └─────────┘           └──────┬─────┘ │
└──────────────────────────────────────────────────────────┼───────┘
                                                           │ HTTP POST
                                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT SYSGUARD-AI (FastAPI)                                        │
│                                                                     │
│  ┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐  │
│  │ /falco-      │──►│  preprocessor.py  │──►│   Autoencoder    │  │
│  │  webhook     │   │ Histogramme 414-d │   │  (Tier 1)        │  │
│  └──────────────┘   │ Tumbling 10s      │   │  MSE → α / β    │  │
│                     └───────────────────┘   └────────┬─────────┘  │
│                                                      │             │
│            ┌─────────────────────────────────────────┘             │
│            │                                                       │
│            ▼                                                       │
│   MSE < α : Normal ─── aucune action                              │
│                                                                     │
│   α ≤ MSE < β : Anomalie MODÉRÉE                                  │
│     ├── quarantine_pod() ← NetworkPolicy deny-all (RÉVERSIBLE)    │
│     └── → Tier 2 (LLM) ──┬── FAUX_POSITIF → lift_quarantine()    │
│                           └── MENACE → maintenir + rapport         │
│                                                                     │
│   MSE ≥ β : Anomalie CRITIQUE                                     │
│     ├── quarantine_pod() ← immédiate                               │
│     ├── alerte critique → opérateur                                │
│     └── → Tier 2 (LLM) → rapport d'incident                      │
│                                                                     │
│   ⚠️ AUCUNE ACTION IRRÉVERSIBLE AUTOMATIQUE (Human-in-the-Loop)   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Runtime security | Falco (eBPF) |
| Event routing | Falco Sidekick (webhook HTTP) |
| Serveur HTTP | FastAPI, Uvicorn |
| ML / Inférence | PyTorch |
| Tier 2 (LLM) | OpenRouter API (DeepSeek V4 Flash, compatible OpenAI API) |
| Orchestration | Kubernetes API (client Python) |
| Cluster | Minikube (dev) / GKE / EKS (prod) |

---

## Structure du dépôt

```
SysGuard-AI/
├── agent/                          # Agent temps réel (FastAPI)
│   ├── main.py                     # Endpoints, pipeline Two-Tier, LLM arbitre
│   ├── preprocessor.py             # Buffer syscalls, histogramme 414-d, tumbling 10s
│   ├── k8s_enforcer.py             # quarantine_pod, lift_quarantine, kill_pod (manuel)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── __init__.py
├── ai_model/                       # Entraînement & modèle
│   ├── train_autoencoder.py        # Entraînement (simulé ou CSV réel), double seuil
│   ├── evaluate_model.py           # Évaluation offline : Precision/Recall/F1
│   ├── test_intrusion.py           # Test interactif Tier 1 (menu scénarios d'attaque)
│   ├── benchmark_overhead.py       # Latence & overhead CPU/RAM du modèle
│   ├── test_tier2_mock.py          # Validation structurelle pipeline Tier 2 (LLM mock)
│   ├── test_tier2_real.py          # Validation expérimentale Tier 2 (LLM réel, OpenRouter)
│   ├── generate_incident_report.py # Rapport d'incident complet par LLM (MITRE, IOC, timeline)
│   ├── incident_reports/           # Rapports générés (.json + .txt)
│   ├── tier2_real_results.json     # Résultats du dernier test LLM réel
│   ├── collect_falco_data.py       # Collecteur Falco → CSV (12h+)
│   ├── dataset/                    # normal_traffic_*.csv (données collectées)
│   ├── saved_model.pt              # Autoencoder entraîné (PyTorch)
│   └── threshold.json              # Seuils α (P99) et β (P99.9)
├── k8s-infra/                      # Manifestes Kubernetes
│   ├── falco-values.yaml           # Helm values pour Falco + Sidekick
│   ├── agent-deployment.yaml       # Déploiement de l'agent
│   ├── victim-app.yaml             # DVWA (application victime)
│   └── rbac.yaml                   # ServiceAccount + ClusterRole
├── tests_attaques/                 # Évaluation
│   ├── simulate_attacks.py         # 4 scénarios d'attaque + métriques
│   └── measure_overhead.py         # Benchmark latence Tier1/E2E + CPU/RAM
└── README.md
```

---

## Spécifications techniques

### Autoencoder (Tier 1 — Sentinelle)

| Paramètre | Valeur |
|-----------|--------|
| Entrée | Vecteur 414-d (histogramme normalisé L1 des syscalls) |
| Architecture | 414 → 207 → 100 → 207 → 414 |
| Activations | ReLU (couches cachées), Sigmoid (sortie) |
| Régularisation | Dropout (p=0.2) |
| Perte | MSE (Mean Squared Error) |
| Entraînement | Non-supervisé (trafic normal uniquement) |
| Seuil α | P99 de la MSE d'entraînement → anomalie modérée |
| Seuil β | P99.9 de la MSE d'entraînement → anomalie critique |

### LLM (Tier 2 — Décideur)

| Paramètre | Valeur |
|-----------|--------|
| Rôle | Arbitre actif : maintient ou lève la quarantaine du Tier 1 |
| Modèle validé | DeepSeek V4 Flash via OpenRouter (compatible OpenAI API) |
| Verdicts | `MENACE` ou `FAUX_POSITIF` |
| Actions | Maintenir quarantaine + alerte / Lever quarantaine automatiquement |
| Latence mesurée | 6–28 s (tier gratuit) ; 1–3 s en production dédiée |
| Validation expérimentale | **6/6 verdicts corrects** (Recall=1.00, Precision=1.00) |
| Faux positifs levés | **100%** des faux positifs du panel (3/3) |

### Remédiation

| Action | Type | Automatique ? |
|--------|------|--------------|
| Quarantaine réseau (NetworkPolicy deny-all) | Réversible | Oui |
| Levée de quarantaine | Réversible | Oui (si LLM → FAUX_POSITIF) |
| Suppression du pod | **Irréversible** | **Non — opérateur humain uniquement** |

### API Agent

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/healthz` | GET | Santé du service, seuils α/β, statut LLM |
| `/falco-webhook` | POST | Réception Falco → détection → remédiation graduée |
| `/incidents` | GET | Derniers rapports d'incident |
| `/admin/kill-pod` | POST | Suppression manuelle (opérateur humain) |
| `/admin/lift-quarantine` | POST | Levée manuelle de quarantaine |

### Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `MODEL_PATH` | `ai_model/saved_model.h5` | Chemin du modèle Keras |
| `THRESHOLD_PATH` | `ai_model/threshold.json` | Seuils α et β |
| `THRESHOLD_ALPHA` | `0.05` | Seuil α (fallback si pas de JSON) |
| `THRESHOLD_BETA` | `0.125` | Seuil β (fallback) |
| `LLM_API_KEY` | (vide) | Clé API OpenRouter (`sk-or-v1-...`) ou OpenAI |
| `LLM_PROVIDER` | `openrouter` | `openrouter` ou `openai` |
| `LLM_MODEL` | `deepseek/deepseek-v4-flash:free` | Modèle LLM (tout modèle OpenRouter gratuit) |
| `K8S_NAMESPACE` | `default` | Namespace Kubernetes ciblé |
| `DRY_RUN` | `false` | `true` pour simuler sans appliquer de remédiation |
| `PORT` | `8000` | Port du serveur FastAPI |

---

## Guide de démarrage rapide

### Prérequis

- Python 3.11+
- PyTorch 2.x (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- `httpx` (`pip install httpx`) — pour le test Tier 2 LLM réel
- kubectl + Minikube (pour le déploiement complet uniquement)

### 1. Entraîner le modèle (données simulées)

```bash
# Créer le modèle avec données simulées
python ai_model/train_autoencoder.py --epochs 50 --samples 10000

# Ou : modèle factice rapide pour tester le pipeline
python ai_model/create_dummy_model.py
```

### 2. Tester le modèle — simulation d'intrusions Two-Tier en temps réel

> **Aucun cluster nécessaire.** Tier 1 (Autoencoder) + Tier 2 (LLM arbitre) s'exécutent directement. La clé API est lue depuis `.env`.

```bash
cd ai_model

# Mode Two-Tier complet (Tier 1 + Tier 2 LLM)
python test_intrusion.py

# Mode Tier 1 seulement (sans appel LLM)
python test_intrusion.py --no-llm
```

Menu interactif :

| Choix | Scénario | Tier 1 | Tier 2 LLM |
|-------|----------|--------|-----------|
| `1` | Trafic normal (baseline) | NORMAL | non déclenché |
| `2` | Reverse Shell (`execve+connect+dup2`) | ANOMALIE → quarantaine | arbitrage automatique |
| `3` | Crypto-mining (`clone×N + sched_yield`) | ANOMALIE → quarantaine | arbitrage automatique |
| `4` | Escalade de privilèges (`ptrace+setuid`) | ANOMALIE → quarantaine | arbitrage automatique |
| `5` | Attaque furtive (faible intensité) | NORMAL (limite Tier 1) | non déclenché |
| `6` | Séquence complète (normal → attaque → normal) | montée MSE progressive | déclenché sur pics |

**Flux par fenêtre détectée comme anomalie :**
```
Fenetre 3/10  [###α###----β------]  MSE=3.35e-04
Top syscalls  : execve:978  connect:625  dup2:380  socket:320
Verdict       : ANOMALIE  [→ Tier 2 en cours...]
                     ↓ (6–28 s)
──────────────────────────────────────────────────────
[TIER 2 — LLM]  latence : 8200 ms
Verdict    : MENACE  (confiance : 95%)
Attaque    : Reverse Shell  [CRITIQUE]
Quarantaine: MAINTENIR
Analyse    : Combinaison execve+connect+dup2 classique de reverse shell...
──────────────────────────────────────────────────────
```

---

### 3. Lancer l'agent (sans cluster)

```bash
pip install -r agent/requirements.txt
python -m uvicorn agent.main:app --host 0.0.0.0 --port 8000
```

### 4. Tester avec curl

```bash
# Événement normal
curl -X POST http://localhost:8000/falco-webhook \
  -H "Content-Type: application/json" \
  -d '{"rule":"","output":"Normal","output_fields":{"container.id":"test-123","evt.type":"read"}}'

# Événement suspect (reverse shell)
curl -X POST http://localhost:8000/falco-webhook \
  -H "Content-Type: application/json" \
  -d '{"rule":"Shell in container","output":"Reverse shell detected","output_fields":{"container.id":"attack-456","evt.type":"execve"}}'
```

### 5. Générer des rapports d'incident (Tier 2 LLM)

```bash
cd ai_model

# Un seul scénario : 1=Reverse Shell, 2=Crypto-mining, 3=PrivEsc, 4=Fichiers sensibles
python generate_incident_report.py --api-key sk-or-v1-VOTRE_CLE --scenario 1

# Tous les scénarios d'un coup
python generate_incident_report.py --api-key sk-or-v1-VOTRE_CLE
```

Chaque rapport inclut : **verdict + confiance, classification MITRE ATT&CK, chronologie horodatée, IOC, blast radius, actions recommandées, note SOC**.

Exemple de sortie (Reverse Shell — INC-001) :

```
VERDICT        : MENACE         (confiance : 0.95)
TYPE D'ATTAQUE : Reverse Shell
SEVERITE       : CRITIQUE
MITRE TACTIC   : Execution
MITRE TECHNIQUE: T1059.004

CHRONOLOGIE
  T-5s  Attaquant exécute /bin/bash via exploit dans le container nginx
  T-2s  connect() vers 192.168.1.100:4444 + dup2() redirection stdin/stdout
  T+0s  SysGuard-AI détecte l'anomalie (execve:1800, connect:1100, dup2:700)
  T+1s  NetworkPolicy deny-all appliquée automatiquement

IOC
  [!] 192.168.1.100:4444
  [!] /bin/bash dans le container nginx:1.25-alpine
  [!] Container ID 3f7a9c2e1b4d

QUARANTAINE : MAINTENIR
```

Rapports sauvegardés dans `ai_model/incident_reports/incident_<id>.json` et `.txt`.

### 7. Valider le pipeline Tier 2 (6 scénarios verdict binaire)

```bash
cd ai_model
python test_tier2_real.py --api-key sk-or-v1-VOTRE_CLE
```

Résultats obtenus (DeepSeek V4 Flash, 6 scénarios) :

| Scénario | Attendu | Verdict LLM | Correct |
|----------|---------|-------------|---------|
| Reverse Shell (execve+connect+dup2) | MENACE | MENACE | ✓ |
| Crypto-mining (sched_yield ×8000) | MENACE | MENACE | ✓ |
| Privilege Escalation (ptrace+setuid) | MENACE | MENACE | ✓ |
| Rolling deployment kubectl | FAUX_POSITIF | FAUX_POSITIF | ✓ |
| Cron job backup (logrotate) | FAUX_POSITIF | FAUX_POSITIF | ✓ |
| Init container (apt-get + pip) | FAUX_POSITIF | FAUX_POSITIF | ✓ |
| **Total** | | | **6/6 (100%)** |

Latence : 6–28 s (tier gratuit OpenRouter). Rapport sauvegardé dans `tier2_real_results.json`.

> Clés API gratuites : [openrouter.ai](https://openrouter.ai) — modèles `deepseek/deepseek-v4-flash:free`, `google/gemma-4-31b-it:free`, etc.

### 8. Évaluation offline complète (métriques Precision/Recall/F1)

```bash
cd ai_model
python evaluate_model.py
```

Résultats sur 500 échantillons par scénario (modèle actuel) :

| Scénario | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Reverse Shell | 0.96 | 0.99 | **0.98** |
| Crypto-mining | 0.96 | 1.00 | **0.98** |
| Fichiers sensibles | 0.96 | 0.95 | **0.95** |
| **Moyenne pondérée** | **0.99** | **0.98** | **0.98** |

### 9. Déploiement sur Minikube (pipeline complet)

```bash
# Démarrer Minikube
minikube start --driver=docker

# Installer Falco + Sidekick
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm install falco falcosecurity/falco -f k8s-infra/falco-values.yaml -n falco --create-namespace

# RBAC + Application victime
kubectl apply -f k8s-infra/rbac.yaml
kubectl apply -f k8s-infra/victim-app.yaml

# Phase 1 : Collecte de données (12h minimum)
python ai_model/collect_falco_data.py --duration 43200 --port 8001

# Phase 2 : Entraînement sur données réelles
python ai_model/train_autoencoder.py --csv ai_model/dataset/normal_traffic.csv

# Phase 3 : Build et déploiement de l'agent
docker build -t sysguard-ai:latest .
kubectl apply -f k8s-infra/agent-deployment.yaml
```

---

## Flux de décision complet

```
Falco (eBPF) → Sidekick → POST /falco-webhook
                                │
                    ┌───────────┴───────────┐
                    │ Histogramme 414-d      │
                    │ Tumbling window (10s)  │
                    │ Normalisation L1       │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │    AUTOENCODER         │
                    │    MSE = ‖x - x̂‖²     │
                    └───────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
           MSE < α         α ≤ MSE < β      MSE ≥ β
           Normal          Modérée          Critique
             │               │               │
          (rien)      quarantine()     quarantine()
                          +                +
                       Tier 2           alerte
                       (LLM)          critique
                          │               +
                    ┌─────┴─────┐     Tier 2
                    │           │     (LLM)
              FAUX_POSITIF   MENACE      │
                    │           │     rapport
            lift_quarantine  maintenir  d'incident
                    +        quarantaine
                 reprise         +
                normal       rapport
                             alerte
                           opérateur
```

---

## Sécurité

- **Aucune action irréversible automatique** : la suppression de pod est réservée à l'opérateur humain
- **Quarantaine réversible** : NetworkPolicy deny-all, levable automatiquement par le LLM
- **Clés API** : ne pas commiter dans le repo — utiliser des secrets K8s
- **RBAC** : l'agent requiert des permissions spécifiques (voir `rbac.yaml`)
- **DRY_RUN** : activable pour tester sans impact réel sur le cluster

---

## Références

- Kotenko et al. (2024) — Modélisation des syscalls (414 types, Bag of System Calls)
- Tien et al. (2019) — KubAnomaly : anomaly detection for Docker orchestration
- Karn et al. (2020) — Cryptomining detection using system calls
- Kalafatidis et al. (2025) — LLM-enhanced IDS for Kubernetes
- Dai et al. (2025) — Automated attack investigation with LLMs
- Habibzadeh et al. (2025) — LLMs for cybersecurity (risques et limites)
- Falco : [falco.org](https://falco.org)
- Falco Sidekick : [github.com/falcosecurity/falcosidekick](https://github.com/falcosecurity/falcosidekick)

---

## Licence

Projet académique — Mémoire de Master 2.
