"""
SysGuard-AI — Dashboard Streamlit.

Lancement :
  cd SysGuard-AI-
  pip install -r dashboard/requirements.txt
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys, os
# Garantir que les imports AI_MODEL fonctionnent quel que soit le CWD
_DASH_DIR = os.path.dirname(os.path.abspath(__file__))
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from services import (
    AI_MODEL,
    BENCHMARK_PATH,
    EVAL_REPORT_PATH,
    INCIDENT_DIR,
    generate_incident,
    get_scenario_description,
    get_scenario_options,
    incident_scenario_options,
    list_incident_reports,
    load_json,
    load_threshold_info,
    model_ready,
    read_text_file,
    run_attack_simulation,
    run_custom_injection,
    run_full_evaluation,
    run_performance_benchmark,
)

# ---------------------------------------------------------------------------
# Config globale
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SysGuard-AI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.4rem; max-width: 1100px; }
h1 { font-weight: 300; letter-spacing: -0.03em; color: #111; font-size: 1.9rem; margin-bottom: 0; }
h2, h3 { font-weight: 500; color: #222; }
[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #eee; }
[data-testid="stMetric"] {
    background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 10px 14px;
}
.sg-tag {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 500;
}
.sg-normal  { background: #ecfdf5; color: #047857; }
.sg-anomaly { background: #fffbeb; color: #b45309; }
.sg-critical { background: #fef2f2; color: #b91c1c; }
.sg-muted { color: #6b7280; font-size: 0.88rem; }
.sg-warn { background: #fffbeb; border-left: 3px solid #d97706;
           padding: 8px 12px; border-radius: 4px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

VERDICT_CSS = {"NORMAL": "sg-normal", "ANOMALIE": "sg-anomaly", "CRITIQUE": "sg-critical"}


def tag(v: str) -> str:
    return f'<span class="sg-tag {VERDICT_CSS.get(v, "sg-muted")}">{v}</span>'


def fmt_lat(ms) -> str:
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    return f"{ms:.0f} ms"


# ---------------------------------------------------------------------------
# Sidebar + navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Accueil": "home",
    "Simulation d'attaque": "simulation",
    "Injection custom": "injection",
    "Statistiques de détection": "evaluation",
    "Benchmark Tier 1": "benchmark",
    "Rapports d'incident": "reports",
}


def sidebar_nav() -> str:
    st.sidebar.title("SysGuard-AI")
    st.sidebar.caption("Console mémoire M2 — Arezki")

    choice = st.sidebar.radio(
        "nav", list(PAGES.keys()), label_visibility="collapsed"
    )
    st.sidebar.divider()

    if model_ready():
        th = load_threshold_info()
        st.sidebar.success("Modèle Option B chargé")
        st.sidebar.caption(f"α = {th.get('alpha', 0):.2e}  (P99)")
        st.sidebar.caption(f"β = {th.get('beta', 0):.2e}  (P99.9)")
        st.sidebar.caption(f"Train : {th.get('train_samples', '—'):,} vecteurs")

        # Indicateur Tier 2
        try:
            from _secrets import load_secrets  # type: ignore
            cfg = load_secrets()
            key = cfg.get("OPENROUTER_API_KEY", "")
        except Exception:
            key = ""
        if key:
            st.sidebar.success("Tier 2 LLM actif")
            st.sidebar.caption(cfg.get("LLM_MODEL", ""))
        else:
            st.sidebar.warning("Tier 2 désactivé")
            st.sidebar.caption("Ajoutez OPENROUTER_API_KEY dans .env")
    else:
        st.sidebar.error("Modèle absent")

    return PAGES[choice]


# ---------------------------------------------------------------------------
# Graphique MSE avec lignes seuils (plotly)
# ---------------------------------------------------------------------------

def mse_chart(windows: list, alpha: float, beta: float):
    x = [w["index"] for w in windows]
    y = [w["mse"] for w in windows]
    colors = [
        "#b91c1c" if w["verdict"] == "CRITIQUE"
        else "#d97706" if w["verdict"] == "ANOMALIE"
        else "#059669"
        for w in windows
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color="#6b7280", width=1.5),
        marker=dict(color=colors, size=9, line=dict(color="white", width=1)),
        name="MSE",
        hovertemplate="Fenêtre %{x}<br>MSE = %{y:.2e}<extra></extra>",
    ))
    fig.add_hline(y=alpha, line=dict(color="#d97706", width=1, dash="dash"),
                  annotation_text=f"α = {alpha:.2e}", annotation_position="top right",
                  annotation_font_color="#d97706")
    fig.add_hline(y=beta, line=dict(color="#b91c1c", width=1, dash="dash"),
                  annotation_text=f"β = {beta:.2e}", annotation_position="top right",
                  annotation_font_color="#b91c1c")
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=260,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(title="Fenêtre (10 s)", showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(title="MSE", showgrid=True, gridcolor="#f3f4f6"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page : Accueil
# ---------------------------------------------------------------------------

def page_home():
    st.title("SysGuard-AI")
    st.markdown('<p class="sg-muted">Détection d\'anomalies syscall · Autoencoder Tier 1 + LLM Tier 2</p>',
                unsafe_allow_html=True)

    if not model_ready():
        st.warning("`saved_model.pth` ou `threshold.json` introuvables dans `ai_model/`.")
        return

    th = load_threshold_info()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entraînement", f"{th.get('train_samples', 0):,} vecteurs")
    c2.metric("Seuil α (P99)", f"{th.get('alpha', 0):.2e}")
    c3.metric("Seuil β (P99.9)", f"{th.get('beta', 0):.2e}")
    c4.metric("Dimension entrée", "414 syscalls")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Tier 1 — Sentinelle")
        st.markdown("- Autoencoder PyTorch (414 dims)\n- Fenêtres de 10 s\n- Latence < 1 ms\n- Seuils α / β sur la MSE de reconstruction")
    with col_b:
        st.subheader("Tier 2 — Décideur")
        st.markdown("- LLM via OpenRouter\n- Rapport DFIR structuré\n- Arbitrage quarantaine\n- Human-in-the-Loop")

    st.markdown("---")
    st.subheader("Résultats disponibles")
    r1, r2, r3 = st.columns(3)
    eval_data = load_json(EVAL_REPORT_PATH)
    bench_data = load_json(BENCHMARK_PATH)
    reports = list_incident_reports()

    with r1:
        if eval_data and "global" in eval_data:
            g = eval_data["global"]
            st.metric("F1 (évaluation)", f"{g.get('f1_score', 0):.4f}")
            st.caption(f"Précision {g['precision']:.1%} · Rappel {g['recall']:.1%}")
        else:
            st.info("Évaluation non lancée")
    with r2:
        if bench_data:
            lat = bench_data["latency"]
            st.metric("Latence P50", f"{lat.get('p50_us', 0):.0f} µs")
            st.caption(f"P99 : {lat.get('p99_us', 0):.0f} µs")
        else:
            st.info("Benchmark non lancé")
    with r3:
        st.metric("Rapports incident", len(reports))
        st.caption("Tier 2 · LLM DFIR")


# ---------------------------------------------------------------------------
# Page : Simulation d'attaque
# ---------------------------------------------------------------------------

def page_simulation():
    st.title("Simulation d'attaque")
    st.markdown('<p class="sg-muted">Rejoue les scénarios du mémoire fenêtre par fenêtre (10 s chacune).</p>',
                unsafe_allow_html=True)

    if not model_ready():
        st.error("Modèle non disponible.")
        return

    options = get_scenario_options()
    labels = dict(options)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        scenario_key = st.selectbox("Scénario", [k for k, _ in options],
                                    format_func=lambda k: labels[k])
    with c2:
        n_windows = st.slider("Fenêtres", 4, 20, 10)
    with c3:
        seed = st.number_input("Seed RNG", 0, 9999, 42)

    use_llm = st.checkbox("Activer Tier 2 (LLM)", value=True,
                          help="Nécessite OPENROUTER_API_KEY dans .env")
    st.caption(get_scenario_description(scenario_key))

    if st.button("▶  Lancer la simulation", type="primary"):
        with st.spinner("Analyse en cours…"):
            result = run_attack_simulation(scenario_key, n_windows, int(seed), use_llm)
        st.session_state["sim_result"] = result
        st.session_state["sim_key"] = scenario_key

    result = st.session_state.get("sim_result")
    if not result:
        return

    # Avertissement si résultat d'un autre scénario
    if st.session_state.get("sim_key") != scenario_key:
        st.markdown('<div class="sg-warn">⚠ Résultats du scénario précédent — relancez pour mettre à jour.</div>',
                    unsafe_allow_html=True)

    summary = result["summary"]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Fenêtres analysées", summary["total"])
    if summary["detection_rate"] is not None:
        rate_pct = summary["detection_rate"] * 100
        s2.metric("Taux détection", f"{rate_pct:.0f} %",
                  delta="✓ 100 %" if rate_pct == 100 else None)
        s3.metric("Détectées (TP)", summary["detected"])
        s4.metric("Critiques (β)", summary["critical"])
    else:
        s2.metric("Faux positifs", summary["false_positives"],
                  delta="0 FP ✓" if summary["false_positives"] == 0 else None)
        s3.metric("Alertes (attendu : 0)", summary["detected"])
        s4.empty()

    if use_llm and not result.get("tier2_enabled"):
        st.info("Tier 2 désactivé — aucune clé API trouvée dans `.env`.")

    mse_chart(result["windows"], result["alpha"], result["beta"])

    # Tableau résultats
    rows = []
    for w in result["windows"]:
        top = ", ".join(f"{s['name']}({s['count']})" for s in w["top_syscalls"][:3])
        t2 = w.get("tier2") or {}
        rows.append({
            "Fen.": w["index"],
            "MSE": f"{w['mse']:.3e}",
            "Tier 1": w["verdict"],
            "Top syscalls": top,
            "Tier 2": t2.get("verdict", "—") if t2 and "error" not in t2 else "—",
            "Confiance": f"{t2.get('confidence', 0):.0%}" if t2.get("confidence") else "—",
            "Lat. T2": fmt_lat(w.get("tier2_latency_ms")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Détail d'une fenêtre
    st.markdown("---")
    idx = st.selectbox("Détail fenêtre", range(1, len(result["windows"]) + 1))
    w = result["windows"][idx - 1]
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown(f"**Verdict Tier 1 :** {tag(w['verdict'])}", unsafe_allow_html=True)
        st.markdown(f"**MSE :** `{w['mse']:.6f}`")
        st.markdown("**Top syscalls :**")
        for s in w["top_syscalls"]:
            st.write(f"  `{s['name']}` → {s['count']}")
    with dc2:
        t2 = w.get("tier2")
        if t2 and "error" not in t2:
            st.markdown(f"**Verdict Tier 2 :** {t2.get('verdict', '?')}")
            st.markdown(f"**Type :** {t2.get('attack_type', '?')} [{t2.get('severity_cvss', '?')}]")
            st.markdown(f"**Confiance :** {t2.get('confidence', 0):.0%}")
            st.markdown(f"**Quarantaine :** {t2.get('quarantine', '?')}")
            if t2.get("explanation"):
                st.caption(t2["explanation"][:300])
        elif use_llm and result.get("tier2_enabled"):
            st.caption("Fenêtre NORMAL — Tier 2 non sollicité.")
        else:
            st.caption("Tier 2 non activé.")


# ---------------------------------------------------------------------------
# Page : Injection custom (VRAI test d'intrusion)
# ---------------------------------------------------------------------------

def page_injection():
    st.title("Injection custom")
    st.markdown(
        '<p class="sg-muted">Définissez manuellement les compteurs syscall et observez la réaction du modèle en temps réel — vrai test d\'intrusion.</p>',
        unsafe_allow_html=True,
    )

    if not model_ready():
        st.error("Modèle non disponible.")
        return

    th = load_threshold_info()
    alpha = th.get("alpha", 0.000148)
    beta = th.get("beta", 0.001144)

    st.info(
        f"Seuils courants : α = {alpha:.2e} (ANOMALIE) · β = {beta:.2e} (CRITIQUE)\n\n"
        "Ajustez les sliders pour composer un vecteur syscall et cliquez **Analyser**."
    )

    GROUPS = {
        "Réseau / Shell (Reverse Shell)": {
            "execve": (0, 3000, 0, "Lance des processus — signature shell"),
            "connect": (0, 2000, 0, "Connexions sortantes"),
            "dup2": (0, 1500, 0, "Duplication descripteurs — redirection I/O"),
            "socket": (0, 1500, 0, "Création de sockets"),
            "bind": (0, 500, 0, "Liaison port réseau"),
            "listen": (0, 500, 0, "Écoute réseau"),
        },
        "Threads / CPU (Crypto-mining)": {
            "clone": (0, 8000, 0, "Création threads — signature mining"),
            "fork": (0, 4000, 0, "Fork processus"),
            "sched_yield": (0, 10000, 0, "Yield CPU — threads mineurs"),
            "futex": (0, 5000, 0, "Synchronisation threads"),
            "mmap": (0, 1000, 0, "Mappage mémoire"),
        },
        "Privilèges (Escalade)": {
            "ptrace": (0, 2000, 0, "Trace processus — injection mémoire"),
            "setuid": (0, 500, 0, "Changement UID"),
            "setgid": (0, 500, 0, "Changement GID"),
            "capset": (0, 300, 0, "Modification capabilities"),
            "chown": (0, 300, 0, "Changement propriétaire"),
            "keyctl": (0, 500, 0, "Manipulation clés kernel"),
        },
        "Fichiers sensibles": {
            "openat": (0, 3000, 0, "Ouverture fichiers — /etc/shadow, /root/.ssh"),
            "read": (0, 3000, 400, "Lectures fichiers"),
            "close": (0, 3000, 400, "Fermetures fichiers"),
            "stat": (0, 2000, 0, "Stat fichiers"),
        },
        "Trafic normal DVWA (baseline)": {
            "epoll_wait": (0, 2000, 0, "Attente événements réseau"),
            "recvfrom": (0, 1500, 0, "Réception paquets"),
            "sendto": (0, 1500, 0, "Envoi paquets"),
            "write": (0, 1500, 400, "Écritures"),
            "poll": (0, 500, 0, "Polling I/O"),
        },
    }

    syscall_counts: dict[str, int] = {}

    with st.expander("Charger un profil prédéfini", expanded=False):
        preset = st.radio(
            "Profil",
            ["Aucun", "Reverse Shell", "Crypto-mining", "Escalade de privilèges",
             "Fichiers sensibles", "Trafic normal"],
            horizontal=True,
        )
        PRESETS = {
            "Reverse Shell":  {"execve": 1800, "connect": 1100, "dup2": 700, "socket": 500},
            "Crypto-mining":  {"clone": 5000, "sched_yield": 8000, "futex": 3000, "fork": 2000},
            "Escalade de privilèges": {"ptrace": 900, "setuid": 240, "capset": 180, "keyctl": 300},
            "Fichiers sensibles": {"openat": 900, "read": 750, "close": 600, "stat": 400},
            "Trafic normal":  {"read": 200, "write": 180, "epoll_wait": 160, "recvfrom": 100,
                               "sendto": 90, "close": 120, "futex": 80, "openat": 40,
                               "stat": 35, "fstat": 25, "poll": 25, "clock_gettime": 15,
                               "getpid": 12, "socket": 7, "accept": 6, "mmap": 4, "brk": 3},
        }
        if preset != "Aucun":
            st.session_state["preset_counts"] = PRESETS.get(preset, {})

    preset_counts = st.session_state.get("preset_counts", {})

    for group_name, syscalls in GROUPS.items():
        st.markdown(f"**{group_name}**")
        cols = st.columns(min(len(syscalls), 3))
        for j, (sc_name, (min_v, max_v, default, desc)) in enumerate(syscalls.items()):
            default_v = preset_counts.get(sc_name, default)
            syscall_counts[sc_name] = cols[j % 3].slider(
                f"`{sc_name}`", min_v, max_v, default_v,
                help=desc, key=f"sc_{sc_name}"
            )

    st.markdown("---")
    if st.button("🔍  Analyser ce vecteur", type="primary"):
        with st.spinner("Inférence Tier 1…"):
            result = run_custom_injection(syscall_counts)
        st.session_state["inject_result"] = result

    result = st.session_state.get("inject_result")
    if not result:
        return

    mse = result["mse"]
    verdict = result["verdict"]
    alpha_v = result["alpha"]
    beta_v = result["beta"]

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.markdown(f"## {tag(verdict)}", unsafe_allow_html=True)
    col_r2.metric("MSE", f"{mse:.6f}")
    col_r3.metric("Seuil franchi",
                  "β (CRITIQUE)" if mse >= beta_v else
                  "α (ANOMALIE)" if mse >= alpha_v else
                  "Aucun (NORMAL)")

    # Jauge MSE visuelle
    ref = max(beta_v * 2, mse * 1.2)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mse,
        number=dict(valueformat=".2e", font=dict(size=20)),
        gauge=dict(
            axis=dict(range=[0, ref], tickformat=".1e"),
            bar=dict(color="#6b7280"),
            steps=[
                dict(range=[0, alpha_v], color="#ecfdf5"),
                dict(range=[alpha_v, beta_v], color="#fffbeb"),
                dict(range=[beta_v, ref], color="#fef2f2"),
            ],
            threshold=dict(
                line=dict(color="#b91c1c", width=2),
                thickness=0.75, value=beta_v,
            ),
        ),
        title=dict(text="Score MSE", font=dict(size=14)),
    ))
    fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=0),
                            paper_bgcolor="white")
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Top syscalls injectés
    top = [(k, v) for k, v in syscall_counts.items() if v > 0]
    top.sort(key=lambda x: x[1], reverse=True)
    if top:
        df_top = pd.DataFrame(top[:10], columns=["Syscall", "Count"])
        st.bar_chart(df_top.set_index("Syscall"))


# ---------------------------------------------------------------------------
# Page : Évaluation / Statistiques de détection
# ---------------------------------------------------------------------------

def page_evaluation():
    st.title("Statistiques de détection")
    st.markdown('<p class="sg-muted">Évaluation offline — trafic normal, hard negatives et 3 scénarios d\'attaque.</p>',
                unsafe_allow_html=True)

    if not model_ready():
        st.error("Modèle non disponible.")
        return

    n_test = st.slider("Échantillons par scénario", 50, 500, 200, step=50)

    if st.button("▶  Lancer l'évaluation", type="primary"):
        with st.spinner(f"Évaluation — {n_test} échantillons par classe…"):
            try:
                data = run_full_evaluation(n_test)
                st.session_state["eval_result"] = data
                st.success("Terminé.")
            except Exception as e:
                st.error(str(e))
                return

    data = st.session_state.get("eval_result") or load_json(EVAL_REPORT_PATH)
    if not data:
        st.info("Cliquez sur « Lancer l'évaluation ».")
        return

    th_source = data.get("train_source", "")
    if th_source:
        st.caption(f"Source entraînement : `{th_source}`")

    g = data.get("global", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1 Score",  f"{g.get('f1_score', 0):.4f}")
    m2.metric("Précision", f"{g.get('precision', 0):.1%}")
    m3.metric("Rappel",    f"{g.get('recall', 0):.1%}")
    m4.metric("Accuracy",  f"{g.get('accuracy', 0):.1%}")

    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        st.subheader("Trafic normal")
        norm = data.get("normal", {})
        fpr = norm.get("fp", 0) / max(norm.get("total", 1), 1)
        st.metric("FPR (faux positifs)", f"{fpr:.1%}",
                  delta=f"−{1-fpr:.1%} ✓" if fpr < 0.05 else None)
        st.caption(f"{norm.get('fp', 0)} FP sur {norm.get('total', 0)} échantillons")
    with cr:
        st.subheader("Hard negatives")
        hn = data.get("hard_negatives", {})
        fpr_hn = hn.get("fp", 0) / max(hn.get("total", 1), 1)
        st.metric("FPR (hard neg.)", f"{fpr_hn:.1%}",
                  delta="0 FP ✓" if fpr_hn == 0 else None)
        st.caption(f"{hn.get('fp', 0)} FP sur {hn.get('total', 0)} échantillons")

    st.subheader("Scénarios d'attaque")
    attacks = data.get("attacks", {})
    rows = []
    for name, stats in attacks.items():
        total = max(stats.get("total", 1), 1)
        recall = stats.get("tp", 0) / total
        rows.append({
            "Scénario": name,
            "Échantillons": stats.get("total", 0),
            "TP": stats.get("tp", 0),
            "FN": stats.get("fn", 0),
            "Rappel": f"{recall:.1%}",
            "MSE moy.": f"{stats.get('mse_mean', 0):.2e}",
            "MSE méd.": f"{stats.get('mse_median', 0):.2e}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if rows:
        fig_recall = go.Figure(go.Bar(
            x=[r["Scénario"] for r in rows],
            y=[float(r["Rappel"].strip("%")) / 100 for r in rows],
            marker_color=["#059669", "#059669", "#059669"],
            text=[r["Rappel"] for r in rows],
            textposition="auto",
        ))
        fig_recall.update_layout(
            height=220, yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white",
            plot_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_recall, use_container_width=True)


# ---------------------------------------------------------------------------
# Page : Benchmark Tier 1
# ---------------------------------------------------------------------------

def page_benchmark():
    st.title("Benchmark Tier 1")
    st.markdown('<p class="sg-muted">Latence d\'inférence, overhead CPU et empreinte mémoire.</p>',
                unsafe_allow_html=True)

    if not model_ready():
        st.error("Modèle non disponible.")
        return

    if st.button("▶  Lancer le benchmark (~30 s)", type="primary"):
        with st.spinner("Benchmark en cours…"):
            try:
                data = run_performance_benchmark()
                st.session_state["bench_result"] = data
                st.success("Terminé.")
            except Exception as e:
                st.error(str(e))

    data = st.session_state.get("bench_result") or load_json(BENCHMARK_PATH)
    if not data:
        st.info("Cliquez sur « Lancer le benchmark ».")
        return

    lat = data.get("latency", {})
    cpu = data.get("cpu_overhead", {})
    mem = data.get("memory", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P50", f"{lat.get('p50_us', 0):.0f} µs")
    c2.metric("P99", f"{lat.get('p99_us', 0):.0f} µs")
    c3.metric("Overhead prod", f"{cpu.get('overhead_rate_limited_pct', 0):.4f} %")
    c4.metric("RAM modèle", f"{mem.get('model_total_kb', 0):.0f} Ko")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribution latence")
        lat_df = pd.DataFrame({
            "Percentile": ["Min", "P50", "P95", "P99", "Max"],
            "µs": [lat.get("min_us"), lat.get("p50_us"),
                   lat.get("p95_us"), lat.get("p99_us"), lat.get("max_us")],
        })
        fig_lat = go.Figure(go.Bar(
            x=lat_df["Percentile"], y=lat_df["µs"],
            marker_color=["#34d399", "#34d399", "#fbbf24", "#f87171", "#b91c1c"],
            text=[f"{v:.1f}" for v in lat_df["µs"]], textposition="auto",
        ))
        fig_lat.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                               paper_bgcolor="white", plot_bgcolor="white",
                               yaxis_title="µs", showlegend=False)
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_b:
        st.subheader("Système")
        sys_info = data.get("system", {})
        info = [
            ("CPU logiques", sys_info.get("cpu_count", "—")),
            ("CPU physiques", sys_info.get("cpu_count_phys", "—")),
            ("RAM totale", f"{sys_info.get('ram_total_gb', '—')} Go"),
            ("PyTorch", sys_info.get("torch_version", "—")),
            ("Python", sys_info.get("python_version", "—")),
            ("Paramètres", f"{mem.get('n_parameters', 0):,}"),
            ("RSS processus", f"{mem.get('process_rss_mb', 0):.1f} Mo"),
            ("Inférences/s (burst)", f"{int(cpu.get('burst_inferences_per_sec', 0)):,}"),
        ]
        for k, v in info:
            st.write(f"**{k}** : {v}")

    if cpu.get("note"):
        st.caption(cpu["note"])


# ---------------------------------------------------------------------------
# Page : Rapports d'incident
# ---------------------------------------------------------------------------

def page_reports():
    st.title("Rapports d'incident")
    st.markdown('<p class="sg-muted">Consultez ou générez des rapports Tier 2 (LLM DFIR).</p>',
                unsafe_allow_html=True)

    tab_view, tab_gen = st.tabs(["Consulter", "Générer"])

    with tab_view:
        reports = list_incident_reports()
        if not reports:
            st.info("Aucun rapport. Générez-en un dans l'onglet « Générer ».")
        else:
            selected = st.selectbox(
                "Rapport",
                range(len(reports)),
                format_func=lambda i: f"{reports[i]['id']} — {reports[i]['name']}",
            )
            r = reports[selected]

            meta = r["data"].get("meta", {})
            alpha_rep = meta.get("alpha", 0)
            th_curr = load_threshold_info()
            alpha_curr = th_curr.get("alpha", 0)
            if alpha_rep and abs(alpha_rep - alpha_curr) / max(alpha_curr, 1e-10) > 0.05:
                st.markdown(
                    f'<div class="sg-warn">⚠ Ce rapport utilise des seuils legacy '
                    f'(α={alpha_rep:.2e}) différents des seuils courants '
                    f'(α={alpha_curr:.2e}).</div>',
                    unsafe_allow_html=True,
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Verdict", r["verdict"])
            c2.metric("Type d'attaque", r["attack_type"])
            c3.metric("Latence LLM", fmt_lat(r.get("latency_ms")))

            fmt = st.radio("Format", ["Texte", "JSON"], horizontal=True)
            if fmt == "Texte" and r["txt_path"]:
                st.code(read_text_file(r["txt_path"]), language=None)
            else:
                st.json(r["data"])

    with tab_gen:
        st.markdown("Génère un rapport d'incident complet via le **LLM Tier 2** pour les 4 scénarios du mémoire.")
        opts = incident_scenario_options()
        scenario_id = st.selectbox(
            "Scénario",
            [sid for sid, _ in opts],
            format_func=lambda sid: dict(opts)[sid],
        )
        api_key = st.text_input("Clé API OpenRouter (vide = `.env`)", type="password")

        if st.button("▶  Générer le rapport", type="primary"):
            with st.spinner("Appel LLM… (~30–60 s)"):
                try:
                    out = generate_incident(scenario_id, api_key or None)
                    if "error" in out:
                        st.error(out["error"])
                    else:
                        st.session_state["last_report"] = out
                        st.success(f"Rapport généré → `{out['json_path']}`")
                        st.caption(f"Latence LLM : {fmt_lat(out.get('latency_ms'))}")
                except Exception as e:
                    st.error(str(e))

        if "last_report" in st.session_state:
            out = st.session_state["last_report"]
            if out.get("txt_content"):
                st.code(out["txt_content"], language=None)


# ---------------------------------------------------------------------------
# Routeur principal
# ---------------------------------------------------------------------------

ROUTE = {
    "home": page_home,
    "simulation": page_simulation,
    "injection": page_injection,
    "evaluation": page_evaluation,
    "benchmark": page_benchmark,
    "reports": page_reports,
}

page_key = sidebar_nav()
ROUTE[page_key]()
