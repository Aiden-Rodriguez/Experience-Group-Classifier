import warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier

_have_xgb = _have_lgbm = _have_cat = True
try:
    from xgboost import XGBClassifier
except Exception:
    _have_xgb = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    _have_lgbm = False
try:
    from catboost import CatBoostClassifier
except Exception:
    _have_cat = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ==== Static paths ====
INPUT_PATH = Path("./data/completed_pokemon_set.parquet")

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

RANDOM_STATE = 4
N_ESTIMATORS = 500
N_JOBS = -1  # use all cores


# -------------------- IO & cleaning --------------------
def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")
    return pd.read_parquet(INPUT_PATH)


def _to_list(v):
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

    dd["evolves_to"] = dd["evolves_to"].map(_to_list)
    dd["name"] = dd["name"].astype(str)
    return dd


# -------------------- Evolution helpers --------------------
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


# -------------------- Preprocessor --------------------
def build_preprocessor(categorical_cols, numeric_cols) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ]
    )


# -------------------- Model builders --------------------
def make_xgb(num_classes: int):
    return XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method="hist",
        num_class=num_classes,  # not strictly required in sklearn API, but fine to set
    )


def make_lgbm(num_classes: int):
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=63,
        objective="multiclass",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def make_catboost(num_classes: int):
    # Silent training (verbose=False) to keep console clean
    return CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def make_extratrees(num_classes: int):
    return ExtraTreesClassifier(
        n_estimators=1000,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


# -------------------- Evaluation core (final-evo backfill) --------------------
def evaluate_model(name: str, estimator_maker, df: pd.DataFrame, all_classes: list[str]) -> float:
    """
    Trains on 6 gens, tests on 1 (loop over gens), predicts finals only then backfills,
    returns overall accuracy across all gens. Uses a LabelEncoder to create a stable class order.
    """
    categorical_cols = ["color"]
    numeric_cols = [c for c in FEATURES if c not in categorical_cols]
    pre = build_preprocessor(categorical_cols, numeric_cols)

    le = LabelEncoder()
    le.fit(all_classes)
    name_to_finals = build_final_evolution_map(df)
    gens = sorted(df[GEN_COL].unique().tolist())

    y_true_all, y_pred_all = [], []

    for g in gens:
        test_mask = df[GEN_COL] == g
        test_df = df.loc[test_mask].copy()
        train_df = df.loc[~test_mask].copy()

        X_train = train_df[FEATURES]
        y_train_enc = le.transform(train_df[TARGET].astype(str))

        # Build pipeline fresh each fold to be fair
        clf = estimator_maker(len(all_classes))
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train_enc)

        # Predict finals only, then backfill across the line
        is_final = test_df["evolves_to"].map(lambda ls: len(ls) == 0)
        finals_df = test_df[is_final].copy()

        if len(finals_df) > 0:
            finals_proba = pipe.predict_proba(finals_df[FEATURES])  # (n_finals, n_classes)
        else:
            finals_proba = np.zeros((0, len(all_classes)), dtype=float)

        final_names = finals_df["name"].tolist()
        name_to_final_proba = {n: p for n, p in zip(final_names, finals_proba)}

        # Average finals probs to every member in the test set
        test_names = test_df["name"].tolist()
        averaged_probas = np.zeros((len(test_df), len(all_classes)), dtype=float)

        for idx, nm in enumerate(test_names):
            finals = name_to_finals.get(nm, [])
            finals_in_test = [f for f in finals if f in name_to_final_proba]
            if len(finals_in_test) == 0:
                single = pipe.predict_proba(test_df.iloc[[idx]][FEATURES])  # fallback
                averaged_probas[idx, :] = single[0]
            else:
                averaged_probas[idx, :] = np.mean([name_to_final_proba[f] for f in finals_in_test], axis=0)

        # Argmax in encoded space, then invert to string labels
        y_pred_enc = averaged_probas.argmax(axis=1)
        y_pred = le.inverse_transform(y_pred_enc)
        y_true = test_df[TARGET].astype(str).values

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    acc = accuracy_score(y_true_all, y_pred_all)
    print(f"{name:12s} accuracy: {acc:.4f}")
    return acc


def main():
    # Load & prep
    df = clean_and_filter(load_data())
    all_classes = sorted(df[TARGET].dropna().unique().tolist())

    print("Model accuracies (overall across all generations):\n")

    results = []

    # ExtraTrees (always available)
    results.append(("ExtraTrees", evaluate_model("ExtraTrees", make_extratrees, df, all_classes)))

    # XGBoost
    if _have_xgb:
        results.append(("XGBoost", evaluate_model("XGBoost", make_xgb, df, all_classes)))
    else:
        print("XGBoost     (skipped) — install with: pip install xgboost")

    # LightGBM
    if _have_lgbm:
        results.append(("LightGBM", evaluate_model("LightGBM", make_lgbm, df, all_classes)))
    else:
        print("LightGBM    (skipped) — install with: pip install lightgbm")

    # CatBoost
    if _have_cat:
        results.append(("CatBoost", evaluate_model("CatBoost", make_catboost, df, all_classes)))
    else:
        print("CatBoost    (skipped) — install with: pip install catboost")

    if results:
        # Print a summary line sorted best-first
        results.sort(key=lambda kv: kv[1], reverse=True)
        best_line = " > ".join(f"{name}:{acc:.4f}" for name, acc in results)
        print("\nSummary (best → worst):")
        print(best_line)


if __name__ == "__main__":
    main()
