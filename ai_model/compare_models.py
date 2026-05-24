"""
Comparaison automatique des deux options d'entrainement -- SysGuard-AI

Option A : Entrainement sur le nouveau dataset reel uniquement
           (normal_traffic_20260516T231519Z.csv -- 7 096 vecteurs)

Option B : Entrainement combine sur les deux captures reelles
           (120212Z + 231519Z -- ratio ~41x, plus robuste)

Le script :
  1. Entraine les deux modeles (A et B) sequentiellement
  2. Evalue chaque modele sur les memes scenarios de test
  3. Affiche un tableau comparatif complet
  4. Sauvegarde les resultats dans comparison_report.json
  5. Declare un vainqueur base sur le F1-Score global

Usage :
  cd ai_model
  python compare_models.py [--epochs 50] [--test-samples 500]
  python compare_models.py --epochs 30 --test-samples 300  (plus rapide)
  python compare_models.py --skip-train  (re-evaluer uniquement)
"""

import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

CSV_A  = os.path.join(DATASET_DIR, "normal_traffic_20260516T231519Z.csv")
CSV_B1 = os.path.join(DATASET_DIR, "normal_traffic_20260516T120212Z.csv")
CSV_B2 = os.path.join(DATASET_DIR, "normal_traffic_20260516T231519Z.csv")

MODEL_A     = os.path.join(SCRIPT_DIR, "model_optionA.pth")
MODEL_B     = os.path.join(SCRIPT_DIR, "model_optionB.pth")
THRESHOLD_A = os.path.join(SCRIPT_DIR, "threshold_optionA.json")
THRESHOLD_B = os.path.join(SCRIPT_DIR, "threshold_optionB.json")
REPORT_A    = os.path.join(SCRIPT_DIR, "evaluation_report_optionA.json")
REPORT_B    = os.path.join(SCRIPT_DIR, "evaluation_report_optionB.json")
COMPARISON_REPORT = os.path.join(SCRIPT_DIR, "comparison_report.json")


def banner(title, width=72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def train_model(label, csv, csv2, output_model, output_threshold, epochs):
    banner(f"ENTRAINEMENT -- {label}")
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "train_autoencoder.py"),
        "--csv", csv,
        "--output-model", output_model,
        "--output-threshold", output_threshold,
        "--epochs", str(epochs),
    ]
    if csv2:
        cmd += ["--csv2", csv2]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"\n  [OK] {label} entraine en {elapsed:.1f}s")
    return elapsed


def evaluate_model(label, model_path, threshold_path, report_path, n_test):
    banner(f"EVALUATION -- {label}")
    sys.path.insert(0, SCRIPT_DIR)
    from evaluate_model import run_evaluation
    result = run_evaluation(
        model_path=model_path,
        threshold_path=threshold_path,
        n_test=n_test,
        report_path=report_path,
        label=label,
    )
    return result


def _delta(val_a, val_b, higher_is_better=True):
    diff = val_b - val_a
    if abs(diff) < 1e-9:
        return "=  ", 0.0
    if higher_is_better:
        arrow = "(B+)" if diff > 0 else "(A+)"
    else:
        arrow = "(A+)" if diff > 0 else "(B+)"
    return arrow, diff


def print_comparison(res_a, res_b):
    banner("TABLEAU COMPARATIF -- Option A vs Option B", width=78)

    g_a = res_a["global"]
    g_b = res_b["global"]

    rows = [
        ("Vecteurs entrainement",    res_a["train_samples"], res_b["train_samples"], True,  ""),
        ("Alpha (seuil P99)",        res_a["alpha"],         res_b["alpha"],         False, ".6f"),
        ("Beta  (seuil P99.9)",      res_a["beta"],          res_b["beta"],          False, ".6f"),
        ("FPR trafic normal",        res_a["fpr_normal"],    res_b["fpr_normal"],    False, ".2%"),
        ("FPR hard negatives",       res_a["fpr_hard_neg"],  res_b["fpr_hard_neg"],  False, ".2%"),
        ("Accuracy globale",         g_a["accuracy"],        g_b["accuracy"],        True,  ".4f"),
        ("Precision globale",        g_a["precision"],       g_b["precision"],       True,  ".4f"),
        ("Recall global",            g_a["recall"],          g_b["recall"],          True,  ".4f"),
        ("F1-Score global",          g_a["f1_score"],        g_b["f1_score"],        True,  ".4f"),
        ("TP (attaques detectees)",  g_a["tp"],              g_b["tp"],              True,  "d"),
        ("FP (fausses alarmes)",     g_a["fp"],              g_b["fp"],              False, "d"),
        ("FN (attaques manquees)",   g_a["fn"],              g_b["fn"],              False, "d"),
    ]

    col = 26
    print(f"\n  {'Metrique':<{col}} {'Option A':>12} {'Option B':>12}  {'Delta':>6}")
    print(f"  {'-'*col} {'-'*12} {'-'*12}  {'-'*6}")

    for name, va, vb, higher_is_better, fmt in rows:
        arrow, diff = _delta(va, vb, higher_is_better)
        if fmt == ".2%":
            sa = f"{va:.2%}"
            sb = f"{vb:.2%}"
            sd = f"{arrow} {abs(diff):.2%}" if arrow != "=  " else "="
        elif fmt == "d":
            sa = str(int(va))
            sb = str(int(vb))
            sd = f"{arrow} {abs(int(diff))}" if arrow != "=  " else "="
        elif fmt == "":
            sa = str(va)
            sb = str(vb)
            sd = ""
        else:
            sa = format(va, fmt)
            sb = format(vb, fmt)
            sd = f"{arrow} {format(abs(diff), fmt.lstrip('.')[:4])}" if arrow != "=  " else "="
        print(f"  {name:<{col}} {sa:>12} {sb:>12}  {sd}")

    # Detail par type d'attaque
    print(f"\n  {'-'*66}")
    print(f"  DETAIL PAR SCENARIO D'ATTAQUE")
    print(f"  {'-'*66}")
    print(f"  {'Scenario':<22} {'Recall A':>9} {'Recall B':>9} {'F1 A':>8} {'F1 B':>8}  {'Delta F1':>9}")
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*8} {'-'*8}  {'-'*9}")

    for atk_name in res_a["attacks"]:
        ra = res_a["attacks"][atk_name]
        rb = res_b["attacks"][atk_name]
        fp_a = res_a["global"]["fp"]
        fp_b = res_b["global"]["fp"]

        rec_a  = ra["tp"] / max(ra["tp"] + ra["fn"], 1)
        rec_b  = rb["tp"] / max(rb["tp"] + rb["fn"], 1)
        prec_a = ra["tp"] / max(ra["tp"] + fp_a, 1)
        prec_b = rb["tp"] / max(rb["tp"] + fp_b, 1)
        f1_a   = 2 * prec_a * rec_a / max(prec_a + rec_a, 1e-9)
        f1_b   = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-9)

        arrow, df1 = _delta(f1_a, f1_b, higher_is_better=True)
        sd = f"{arrow} {abs(df1):.3f}" if arrow != "=  " else "="
        print(f"  {atk_name:<22} {rec_a:>8.1%} {rec_b:>9.1%} {f1_a:>8.3f} {f1_b:>8.3f}  {sd:>9}")

    # Verdict final
    print(f"\n  {'='*66}")
    f1_a = g_a["f1_score"]
    f1_b = g_b["f1_score"]
    if f1_b > f1_a + 0.001:
        winner = "OPTION B (dataset combine)"
        reason = f"F1 superieur de {f1_b - f1_a:.4f} ({f1_b:.4f} vs {f1_a:.4f})"
    elif f1_a > f1_b + 0.001:
        winner = "OPTION A (dataset reel seul)"
        reason = f"F1 superieur de {f1_a - f1_b:.4f} ({f1_a:.4f} vs {f1_b:.4f})"
    else:
        winner = "EQUIVALENT (difference < 0.001)"
        reason = "Les deux modeles sont statistiquement equivalents"

    fpr_a = res_a["fpr_normal"]
    fpr_b = res_b["fpr_normal"]
    if fpr_b < fpr_a - 0.005:
        fpr_note = f"Option B a un FPR plus bas ({fpr_b:.2%} vs {fpr_a:.2%}) -> moins de fausses alarmes"
    elif fpr_a < fpr_b - 0.005:
        fpr_note = f"Option A a un FPR plus bas ({fpr_a:.2%} vs {fpr_b:.2%}) -> moins de fausses alarmes"
    else:
        fpr_note = f"FPR equivalents ({fpr_a:.2%} vs {fpr_b:.2%})"

    print(f"\n  VERDICT  : {winner}")
    print(f"  Raison   : {reason}")
    print(f"  FPR      : {fpr_note}")
    print()
    print("  RECOMMANDATION SOUTENANCE :")
    if "OPTION A" in winner:
        print("  -> Option A : modele entraine sur 7 096 vecteurs 100% reels,")
        print("     sans aucune augmentation artificielle. Discours academique le plus pur.")
    elif "OPTION B" in winner:
        print("  -> Option B : plus de donnees reelles = meilleure generalisation.")
        print("     Deux captures distinctes reduisent le surapprentissage.")
    else:
        print("  -> Les deux sont defendables. Privilegiez l'Option A (plus simple a justifier).")
    print(f"  {'='*66}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Option A vs Option B d'entrainement SysGuard-AI"
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Nombre d'epochs max pour chaque modele (defaut: 50)")
    parser.add_argument("--test-samples", type=int, default=500,
                        help="Echantillons de test par scenario (defaut: 500)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Sauter l'entrainement si les modeles existent deja")
    args = parser.parse_args()

    for csv_file in [CSV_A, CSV_B1, CSV_B2]:
        if not os.path.exists(csv_file):
            print(f"ERREUR : fichier CSV introuvable : {csv_file}")
            sys.exit(1)

    banner("SysGuard-AI -- COMPARAISON OPTION A vs OPTION B", width=72)
    print(f"\n  Option A : {os.path.basename(CSV_A)}")
    print(f"  Option B : {os.path.basename(CSV_B1)} + {os.path.basename(CSV_B2)}")
    print(f"  Epochs   : {args.epochs}  |  Test samples : {args.test_samples}/scenario")

    train_times = {}

    if not args.skip_train or not os.path.exists(MODEL_A):
        train_times["A"] = train_model(
            label="Option A -- dataset reel seul (231519Z)",
            csv=CSV_A, csv2=None,
            output_model=MODEL_A, output_threshold=THRESHOLD_A,
            epochs=args.epochs,
        )
    else:
        print(f"\n[Skip] Option A : modele existant utilise ({MODEL_A})")
        train_times["A"] = 0

    if not args.skip_train or not os.path.exists(MODEL_B):
        train_times["B"] = train_model(
            label="Option B -- datasets combines (120212Z + 231519Z)",
            csv=CSV_B1, csv2=CSV_B2,
            output_model=MODEL_B, output_threshold=THRESHOLD_B,
            epochs=args.epochs,
        )
    else:
        print(f"\n[Skip] Option B : modele existant utilise ({MODEL_B})")
        train_times["B"] = 0

    res_a = evaluate_model(
        label="Option A",
        model_path=MODEL_A,
        threshold_path=THRESHOLD_A,
        report_path=REPORT_A,
        n_test=args.test_samples,
    )

    res_b = evaluate_model(
        label="Option B",
        model_path=MODEL_B,
        threshold_path=THRESHOLD_B,
        report_path=REPORT_B,
        n_test=args.test_samples,
    )

    print_comparison(res_a, res_b)

    comparison = {
        "config": {
            "epochs":       args.epochs,
            "test_samples": args.test_samples,
            "csv_A":        CSV_A,
            "csv_B":        f"{CSV_B1} + {CSV_B2}",
        },
        "train_time_seconds": train_times,
        "option_A": res_a,
        "option_B": res_b,
        "verdict": {
            "f1_A":   res_a["global"]["f1_score"],
            "f1_B":   res_b["global"]["f1_score"],
            "winner": (
                "A" if res_a["global"]["f1_score"] > res_b["global"]["f1_score"] + 0.001
                else "B" if res_b["global"]["f1_score"] > res_a["global"]["f1_score"] + 0.001
                else "equivalent"
            ),
        },
    }
    with open(COMPARISON_REPORT, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n  Rapport de comparaison sauvegarde : {COMPARISON_REPORT}")


if __name__ == "__main__":
    main()
