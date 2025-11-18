#!/usr/bin/env python
import warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ==== Static paths ====
INPUT_PATH = Path("./data/completed_pokemon_set.parquet")
OUTPUT_DIR = Path("./eval_with_all_groups_png")

# ==== Model constants ====
FEATURES = [
    "base_total",
    "evolution_stages",
    "color",
    "capture_rate",
    "percentage_male",
    "base_experience",
    "base_egg_steps",
    "no_in_generation",
]
EXTRA_COLS_FOR_EVOS = ["name", "evolves_to", "evolves_from"]
TARGET = "experience_growth"
GEN_COL = "generation"

# NOTE: This includes all classes anyways. This is because previously I didn't allow Erratic and Fluctuating, but random foresting already doesn't predict these.
# Turns out that including these actually slightly increases accuracy.
ALLOWED_PREDICT_CLASSES = ["Fast", "Medium Fast", "Medium Slow", "Slow", "Erratic", "Fluctuating"]

RANDOM_STATE = 4
N_ESTIMATORS = 500
N_JOBS = -1


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    return df


def to_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, (np.ndarray, pd.Series)):
        return [str(x).strip() for x in list(v) if str(x).strip()]
    if v is None:
        return []
    try:
        if pd.isna(v):
            return []
    except Exception:
        pass
    s = str(v).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            res = ast.literal_eval(s)
            if isinstance(res, list):
                return [str(x).strip() for x in res if str(x).strip()]
        except Exception:
            pass
    return [tok.strip() for tok in s.replace("~", ",").split(",") if tok.strip()]


def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    needed = FEATURES + EXTRA_COLS_FOR_EVOS + [TARGET, GEN_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    dd = df[needed].copy()

    numeric_cols = [
        "base_total",
        "evolution_stages",
        "capture_rate",
        "percentage_male",
        "base_experience",
        "base_egg_steps",
        "no_in_generation",
        GEN_COL,
    ]
    for c in numeric_cols:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")

    dd["color"] = dd["color"].astype("string").fillna("unknown")

    dd = dd.dropna(subset=[GEN_COL, TARGET])
    dd[GEN_COL] = dd[GEN_COL].astype(int)
    dd[TARGET] = dd[TARGET].astype("string")

    dd["evolves_to"] = dd["evolves_to"].map(to_list)
    dd["name"] = dd["name"].astype(str)

    return dd


def build_final_evolution_map(df: pd.DataFrame) -> dict[str, list[str]]:
    name_to_children = {n: list(kids) for n, kids in zip(df["name"], df["evolves_to"])}

    def reachable_finals(start: str) -> list[str]:
        seen = set()
        q = deque([start])
        found = set()
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            kids = name_to_children.get(cur, [])
            if not kids:
                found.add(cur)
            else:
                for k in kids:
                    q.append(k)
        return sorted(found)

    return {n: reachable_finals(n) for n in name_to_children.keys()}


def build_pipeline(categorical_cols, numeric_cols) -> Pipeline:
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", rf)])
    return pipe


def plot_and_save_confusion_matrix(y_true, y_pred, labels, title, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(8, 7))
    disp.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_and_save_precision_recall(y_true, y_proba, classes, title, out_path: Path,
                                   baseline_proba=None):
    """
    Per-class PR + micro-average; optional baseline overlay (micro).
    """
    Y_bin = label_binarize(y_true, classes=classes)
    if Y_bin.ndim == 1:
        Y_bin = np.column_stack([1 - Y_bin, Y_bin])

    plt.figure(figsize=(8, 7))
    any_plotted = False
    for i, cls in enumerate(classes):
        positives = int(Y_bin[:, i].sum())
        if positives == 0 or positives == Y_bin.shape[0]:
            continue
        precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(Y_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, label=f"{cls} (AP={ap:.3f})")
        any_plotted = True

    prec_micro, rec_micro, _ = precision_recall_curve(Y_bin.ravel(), y_proba.ravel())
    ap_micro = average_precision_score(Y_bin, y_proba, average="micro")
    plt.plot(rec_micro, prec_micro, linewidth=2.2, label=f"micro-average (AP={ap_micro:.3f})")

    if baseline_proba is not None:
        b_prec_micro, b_rec_micro, _ = precision_recall_curve(Y_bin.ravel(), baseline_proba.ravel())
        b_ap_micro = average_precision_score(Y_bin, baseline_proba, average="micro")
        plt.plot(b_rec_micro, b_prec_micro, linestyle="--",
                 label=f"baseline (random pick) — micro (AP={b_ap_micro:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    if any_plotted or baseline_proba is not None:
        plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_and_save_roc(y_true, y_proba, classes, title, out_path: Path,
                      baseline_proba=None):
    """
    Per-class ROC + micro-average; optional baseline overlay (micro).
    """
    Y_bin = label_binarize(y_true, classes=classes)
    if Y_bin.ndim == 1:
        Y_bin = np.column_stack([1 - Y_bin, Y_bin])

    plt.figure(figsize=(8, 7))
    any_plotted = False
    for i, cls in enumerate(classes):
        y_i = Y_bin[:, i]
        pos = int(y_i.sum())
        neg = int((1 - y_i).sum())
        if pos == 0 or neg == 0:
            continue
        fpr, tpr, _ = roc_curve(y_i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
        any_plotted = True

    fpr_micro, tpr_micro, _ = roc_curve(Y_bin.ravel(), y_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linewidth=2.2, label=f"micro-average (AUC={auc_micro:.3f})")

    if baseline_proba is not None:
        bfpr, btpr, _ = roc_curve(Y_bin.ravel(), baseline_proba.ravel())
        bauc = auc(bfpr, btpr)
        plt.plot(bfpr, btpr, linestyle="--",
                 label=f"baseline (random pick) — micro (AUC={bauc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    if any_plotted or baseline_proba is not None:
        plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Model evaluation results\n")
        f.write("========================\n\n")

    df = load_data()
    df = clean_and_filter(df)

    categorical_cols = ["color"]
    numeric_cols = [c for c in FEATURES if c not in categorical_cols]

    all_classes = sorted(df[TARGET].dropna().unique().tolist())  # includes Erratic/Fluctuating
    name_to_finals = build_final_evolution_map(df)

    gens = sorted(df[GEN_COL].unique().tolist())
    print(f"Found generations: {gens}")
    print(f"Classes: {all_classes}")

    all_cls_index = {c: i for i, c in enumerate(all_classes)}

    # --- Accumulators for overall (all-gens) summary & curves ---
    overall_true = []
    overall_pred = []
    overall_model_proba = []
    overall_baseline_proba = []

    for g in gens:
        test_mask = df[GEN_COL] == g
        test_df_full = df.loc[test_mask].copy()

        # TRAIN:
        train_df = df.loc[~test_mask].copy()
        train_df = train_df[train_df[TARGET].isin(ALLOWED_PREDICT_CLASSES)].copy()

        X_train = train_df[FEATURES]
        y_train = train_df[TARGET].astype(str)

        pipe = build_pipeline(categorical_cols, numeric_cols)
        pipe.fit(X_train, y_train)

        # Predict finals only, then backfill to all pre-evos
        is_final = test_df_full["evolves_to"].map(lambda ls: len(ls) == 0)
        finals_df = test_df_full[is_final].copy()

        if len(finals_df) > 0:
            finals_proba = pipe.predict_proba(finals_df[FEATURES])
            trained_classes = list(pipe.named_steps["clf"].classes_)
        else:
            finals_proba = np.zeros((0, len(ALLOWED_PREDICT_CLASSES)), dtype=float)
            trained_classes = list(pipe.named_steps["clf"].classes_)

        trained_idx = {c: i for i, c in enumerate(trained_classes)}
        finals_proba_aligned = np.zeros((finals_proba.shape[0], len(all_classes)), dtype=float)
        for j, cls in enumerate(all_classes):
            if cls in trained_idx:
                finals_proba_aligned[:, j] = finals_proba[:, trained_idx[cls]]
            else:
                finals_proba_aligned[:, j] = 0.0

        final_names = finals_df["name"].tolist()
        name_to_final_proba = {n: p for n, p in zip(final_names, finals_proba_aligned)}

        test_names = test_df_full["name"].tolist()
        averaged_probas = np.zeros((len(test_df_full), len(all_classes)), dtype=float)

        for idx, nm in enumerate(test_names):
            finals = name_to_finals.get(nm, [])
            finals_in_test = [f for f in finals if f in name_to_final_proba]
            if len(finals_in_test) == 0:
                single = pipe.predict_proba(test_df_full.iloc[[idx]][FEATURES])
                aligned = np.zeros((1, len(all_classes)), dtype=float)
                for j, cls in enumerate(all_classes):
                    if cls in trained_idx:
                        aligned[:, j] = single[:, trained_idx[cls]]
                averaged_probas[idx, :] = aligned[0]
            else:
                averaged_probas[idx, :] = np.mean([name_to_final_proba[f] for f in finals_in_test], axis=0)

        y_pred = np.array([all_classes[i] for i in averaged_probas.argmax(axis=1)])
        y_test = test_df_full[TARGET].astype(str).values

        # Accumulate across gens for overall summary & overall curves
        overall_true.append(y_test)
        overall_pred.append(y_pred)

        # -------- Baseline: random pick among ALLOWED_PREDICT_CLASSES --------
        rng = np.random.default_rng(RANDOM_STATE + int(g))
        baseline_labels = rng.choice(ALLOWED_PREDICT_CLASSES, size=len(test_df_full))
        baseline_proba = np.zeros_like(averaged_probas)
        for i, bl in enumerate(baseline_labels):
            baseline_proba[i, all_cls_index[bl]] = 1.0

        overall_model_proba.append(averaged_probas)
        overall_baseline_proba.append(baseline_proba)

        # ----- Per-generation results.txt section -----
        correct = int((y_pred == y_test).sum())
        total = len(y_test)
        incorrect = total - correct
        accuracy = correct / total if total else 0.0

        b_correct = int((baseline_labels == y_test).sum())
        b_accuracy = b_correct / total if total else 0.0

        per_class_lines = []
        for cls in all_classes:
            mask = (y_test == cls)
            support = int(mask.sum())
            correct_c = int((y_pred[mask] == cls).sum())
            incorrect_c = support - correct_c
            acc_c = (correct_c / support) if support else 0.0
            per_class_lines.append(
                f"    {cls:14s}  support={support:3d}  correct={correct_c:3d}  incorrect={incorrect_c:3d}  acc={acc_c:6.3f}"
            )

        pred_counts = {cls: int((y_pred == cls).sum()) for cls in all_classes}
        pred_line = "  Predicted counts: " + ", ".join(f"{k}={v}" for k, v in pred_counts.items())

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(f"=== Generation {g} ===\n")
            f.write(f"Test size: {total}\n")
            f.write(f"Model:    correct={correct}, incorrect={incorrect}, accuracy={accuracy:.3f}\n")
            f.write(f"Baseline: correct={b_correct}, incorrect={total - b_correct}, accuracy={b_accuracy:.3f}\n")
            f.write("Per true experience type (model):\n")
            f.write("\n".join(per_class_lines) + "\n")
            f.write(pred_line + "\n\n")

        print(f"\n=== Generation {g}: test size={total} ===")
        print(classification_report(y_test, y_pred, labels=all_classes, zero_division=0))

        # ----- Per-generation figures -----
        cm_path = OUTPUT_DIR / f"confusion_matrix_gen{g}.png"
        plot_and_save_confusion_matrix(
            y_true=y_test, y_pred=y_pred, labels=all_classes,
            title=f"Confusion Matrix — Test on Generation {g}", out_path=cm_path
        )

        pr_path = OUTPUT_DIR / f"precision_recall_gen{g}.png"
        plot_and_save_precision_recall(
            y_true=y_test, y_proba=averaged_probas, classes=all_classes,
            title=f"Precision–Recall — Test on Generation {g}",
            out_path=pr_path, baseline_proba=baseline_proba
        )

        roc_path = OUTPUT_DIR / f"roc_gen{g}.png"
        plot_and_save_roc(
            y_true=y_test, y_proba=averaged_probas, classes=all_classes,
            title=f"ROC — Test on Generation {g}",
            out_path=roc_path, baseline_proba=baseline_proba
        )

    # ---------- Overall (All Generations) ----------
    y_true_all = np.concatenate(overall_true, axis=0) if overall_true else np.array([])
    y_pred_all = np.concatenate(overall_pred, axis=0) if overall_pred else np.array([])

    total_all = int(y_true_all.size)
    correct_all = int((y_true_all == y_pred_all).sum()) if total_all else 0
    incorrect_all = total_all - correct_all
    accuracy_all = (correct_all / total_all) if total_all else 0.0

    per_class_overall_lines = []
    for cls in all_classes:
        mask = (y_true_all == cls)
        support = int(mask.sum())
        correct_c = int((y_pred_all[mask] == cls).sum())
        incorrect_c = support - correct_c
        acc_c = (correct_c / support) if support else 0.0
        per_class_overall_lines.append(
            f"    {cls:14s}  support={support:4d}  correct={correct_c:4d}  incorrect={incorrect_c:4d}  acc={acc_c:6.3f}"
        )

    pred_counts_all = {cls: int((y_pred_all == cls).sum()) for cls in all_classes}
    pred_line_all = "  Predicted counts (overall): " + ", ".join(f"{k}={v}" for k, v in pred_counts_all.items())

    with open(results_path, "a", encoding="utf-8") as f:
        f.write("=== Overall (All Generations) ===\n")
        f.write(f"Total size: {total_all}\n")
        f.write(f"Model: correct={correct_all}, incorrect={incorrect_all}, accuracy={accuracy_all:.3f}\n")
        f.write("Per true experience type (overall):\n")
        f.write("\n".join(per_class_overall_lines) + "\n")
        f.write(pred_line_all + "\n")

    # Overall PR/ROC across all generations
    if overall_model_proba:
        proba_all = np.concatenate(overall_model_proba, axis=0)
        baseline_all = np.concatenate(overall_baseline_proba, axis=0)

        pr_overall_path = OUTPUT_DIR / "precision_recall_overall.png"
        plot_and_save_precision_recall(
            y_true=y_true_all,
            y_proba=proba_all,
            classes=all_classes,
            title="Precision–Recall — Overall (All Generations)",
            out_path=pr_overall_path,
            baseline_proba=baseline_all
        )

        roc_overall_path = OUTPUT_DIR / "roc_overall.png"
        plot_and_save_roc(
            y_true=y_true_all,
            y_proba=proba_all,
            classes=all_classes,
            title="ROC — Overall (All Generations)",
            out_path=roc_overall_path,
            baseline_proba=baseline_all
        )

    print(f"\nSaved evaluation figures + baseline overlays + results to: {OUTPUT_DIR.resolve()}\n"
          f"- results.txt (includes overall summary)\n"
          f"- confusion_matrix_gen#.png\n- precision_recall_gen#.png\n- roc_gen#.png\n"
          f"- precision_recall_overall.png\n- roc_overall.png")


if __name__ == "__main__":
    main()
