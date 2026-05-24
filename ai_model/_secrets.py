"""
Chargement automatique des secrets depuis .env (racine du projet).

Utilisation dans n'importe quel script ai_model/*.py :
    from _secrets import load_secrets
    cfg = load_secrets()
    api_key = cfg["OPENROUTER_API_KEY"]
"""

import os
from pathlib import Path


def load_secrets() -> dict:
    """Charge .env depuis la racine du projet et retourne un dict de variables."""
    # Cherche .env dans : dossier courant, parent (ai_model/..), grand-parent
    search_paths = [
        Path(__file__).parent,          # ai_model/
        Path(__file__).parent.parent,   # SysGuard-AI-/
        Path.cwd(),
        Path.cwd().parent,
    ]

    env_path = None
    for p in search_paths:
        candidate = p / ".env"
        if candidate.exists():
            env_path = candidate
            break

    secrets = {}

    if env_path:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    secrets[key.strip()] = value.strip()
        # Injecter dans os.environ pour compatibilité
        for k, v in secrets.items():
            os.environ.setdefault(k, v)
    else:
        # Fallback : lire depuis os.environ (variables d'env système)
        for key in ("OPENROUTER_API_KEY", "LLM_MODEL", "LLM_PROVIDER"):
            if key in os.environ:
                secrets[key] = os.environ[key]

    return secrets


def get_api_key(args_key: str | None = None) -> str:
    """
    Retourne la clé API selon la priorité :
      1. Argument CLI (--api-key) s'il est fourni
      2. Variable OPENROUTER_API_KEY du .env
      3. Variable d'environnement système OPENROUTER_API_KEY
    Leve ValueError si aucune clé n'est trouvee.
    """
    if args_key:
        return args_key

    cfg = load_secrets()
    key = cfg.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError(
            "Cle API introuvable.\n"
            "Solution 1 : ajouter OPENROUTER_API_KEY=sk-or-v1-... dans le fichier .env\n"
            "Solution 2 : passer --api-key sk-or-v1-... en argument\n"
            "Solution 3 : set OPENROUTER_API_KEY=sk-or-v1-... (variable systeme)"
        )
    return key
