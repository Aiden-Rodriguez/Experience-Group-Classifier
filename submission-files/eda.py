# Converts original parquet file into more readable format. Converts the "Experience growth" column into experience type 
# (One of Erratic, Fast, Medium Fast, Medium Slow, Slow, Fluctuating)
# Maps to these respectively: 600,000, 800,000, 1,000,000, 1,059,860, 1,250,000, 1,640,000
# Also creates the number in generation field, which counts how close to the beginning of the generation the pokemon is.
# This is useful because pokemon earlier in a generation are typically 'early game Pokemon', and thus grow easier, but are weaker.

#note to self - if gender ratio is null, is genderless
#note to self again - Had to use pokemon base stats from parquet 2 --- parquet 1 has issues differentiating megas/base forms

#csvs obtained from the following:
#https://www.kaggle.com/datasets/rounakbanik/pokemon
# Used for most of the main data, the base
#https://www.kaggle.com/datasets/kylekohnen/complete-pokemon-data-set
# Used for egg_groups (groups ~ separated), base_experience, egg_cycles, base_happiness, can_evolve, evolves_from, shape, color

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DESIRED_ORDER = [
    "name",
    "pokedex_number",
    "generation",
    "no_in_generation",
    "percentage_in_generation",
    "abilities",
    "type1",
    "type2",
    "weight_kg",
    "is_legendary",
    "Hp",
    "attack",
    "sp_attack",
    "defense",
    "sp_defense",
    "speed",
    "base_total",
    "base_egg_steps",
    "base_happiness",
    "capture_rate",
    "experience_growth",
    "height_m",
    "percentage_male",
    # This is second parquet info below
    "base_experience",
    "egg_cycles",
    "can_evolve",
    "evolves_from",
    "evolves_to",
    "evolution_stages",
    "shape",
    "color",
    "egg_groups",
]

EXP_TOTAL_TO_LABEL = {
    600000:  "Erratic",
    800000:  "Fast",
    1000000: "Medium Fast",
    1059860: "Medium Slow",
    1250000: "Slow",
    1640000: "Fluctuating",
}

ALIASES = {
    "hp": "Hp", "HP": "Hp",
    "sp_atk": "sp_attack", "spatk": "sp_attack", "sp.attack": "sp_attack",
    "special-attack": "sp_attack", "special_attack": "sp_attack",
    "sp_def": "sp_defense", "spdef": "sp_defense", "sp.def": "sp_defense",
    "special-defense": "sp_defense", "special_defense": "sp_defense",
    "type_1": "type1", "type 1": "type1", "type-1": "type1",
    "type_2": "type2", "type 2": "type2", "type-2": "type2",
    "experience_growth_total": "experience_growth",
    "exp_growth": "experience_growth",
    "experiencegroup": "experience_growth",
    "baseexp": "base_experience",
    "egg_cycles_total": "egg_cycles",
    "evolvesfrom": "evolves_from",
    "egg_group": "egg_groups",
    "primary_color": "color",
}

# This is to map names that are not the same between the datasets.
NAME_CANON_MAP = {
    "Mr Mime": "Mr. Mime",
    "Nidoran笙": "Nidoran F",
    "Nidoran笙�": "Nidoran M",
    "Farfetchd": "Farfetch'd",
    "Ho Oh": "Ho-Oh",
    "Deoxys Normal": "Deoxys",
    "Wormadam Plant": "Wormadam",
    "Mime Jr": "Mime Jr.",
    "Porygon Z": "Porygon-Z",
    "Giratina Origin": "Giratina",
    "Shaymin Land": "Shaymin",
    "Basculin Red Striped": "Basculin",
    "Darmanitan Standard": "Darmanitan",
    "Tornadus Incarnate": "Tornadus",
    "Thundurus Incarnate": "Thundurus",
    "Landorus Incarnate": "Landorus",
    "Keldeo Ordinary": "Keldeo",
    "Meloetta Aria": "Meloetta",
    "Flabﾃｩbﾃｩ": "Flabebe",
    "Meowstic Male": "Meowstic",
    "Aegislash Shield": "Aegislash",
    "Pumpkaboo Average": "Pumpkaboo",
    "Gourgeist Average": "Gourgeist",
    "Oricorio Pom Pom": "Oricorio",
    "Lycanroc Midday": "Lycanroc",
    "Wishiwashi Solo": "Wishiwashi",
    "Type Null": "Type: Null",
    "Minior Blue": "Minior",
    "Mimikyu Disguised": "Mimikyu",
    "Jangmo O": "Jangmo-o",
    "Hakamo O": "Hakamo-o",
    "Kommo O": "Kommo-o",
}

STAT_COLS = ["Hp", "attack", "sp_attack", "defense", "sp_defense", "speed"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col in ALIASES:
            rename_map[col] = ALIASES[col]; continue
        low = col.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        if low in ALIASES:
            rename_map[col] = ALIASES[low]
    return df.rename(columns=rename_map)

def normalize_name(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def canonicalize_name(s) -> str:
    if pd.isna(s):
        return s
    ss = str(s).strip()
    return NAME_CANON_MAP.get(ss, ss)

def canonicalize_name_list(vals):
    if not isinstance(vals, list):
        return vals
    out = []
    for v in vals:
        if pd.isna(v):
            continue
        ss = str(v).strip()
        if not ss:
            continue
        out.append(NAME_CANON_MAP.get(ss, ss))
    return list(dict.fromkeys(out).keys())

def to_bool_from_01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    return s.astype(bool)

def coerce_int(x):
    if pd.isna(x):
        return np.nan
    try:
        return int(float(str(x).replace(",", "")))
    except Exception:
        return np.nan

def map_experience_growth(df: pd.DataFrame) -> pd.Series:
    col = df["experience_growth"]
    if pd.api.types.is_numeric_dtype(col):
        totals = col.astype("Int64")
    else:
        totals = col.map(coerce_int)
    labels = []
    valid_lbls = set(EXP_TOTAL_TO_LABEL.values())
    for tot, raw in zip(totals, col):
        if pd.isna(tot):
            s = str(raw).strip()
            labels.append(s if s in valid_lbls else np.nan)
        else:
            labels.append(EXP_TOTAL_TO_LABEL.get(int(tot), np.nan))
    return pd.Series(labels, index=df.index, dtype="string")

def compute_no_in_generation(df: pd.DataFrame) -> pd.Series:
    gen = pd.to_numeric(df["generation"], errors="coerce")
    dex = pd.to_numeric(df["pokedex_number"], errors="coerce")
    tmp = df.assign(_gen=gen, _dex=dex).sort_values(["_gen", "_dex"])
    ranks = tmp.groupby("_gen")["_dex"].rank(method="dense", ascending=True)
    return ranks.astype("Int64").reindex(df.index)

def compute_pct_in_generation(df: pd.DataFrame) -> pd.Series:
    gen = pd.to_numeric(df["generation"], errors="coerce")
    dex = pd.to_numeric(df["pokedex_number"], errors="coerce")
    tmp = df.assign(_gen=gen, _dex=dex).sort_values(["_gen", "_dex"])
    rank = tmp.groupby("_gen")["_dex"].rank(method="dense", ascending=True)
    total = tmp.groupby("_gen")["_dex"].transform("nunique")
    pct = (rank / total).astype(float)
    return pct.reindex(df.index)

def compute_evolution_stages_by_depth(df: pd.DataFrame) -> pd.Series:
    name_to_norm = {str(n): nn for n, nn in zip(df["name"], df["name_norm"])}
    def to_parent_norm(v):
        if pd.isna(v): return ""
        s = str(v).strip()
        if not s: return ""
        return name_to_norm.get(s, s.lower())
    parent_map = dict(zip(df["name_norm"], df["evolves_from"].map(to_parent_norm)))
    def to_child_norms(v):
        if not isinstance(v, list): return []
        out = []
        for nm in v:
            if pd.isna(nm): continue
            s = str(nm).strip()
            if not s: continue
            out.append(name_to_norm.get(s, s.lower()))
        return list(dict.fromkeys(out).keys())
    children_map = dict(zip(df["name_norm"], df["evolves_to"].map(to_child_norms)))
    nodes = list(df["name_norm"])
    roots = [n for n in nodes if not parent_map.get(n, "")]
    from functools import lru_cache
    import sys
    sys.setrecursionlimit(10000)
    @lru_cache(None)
    def max_depth(node: str) -> int:
        kids = children_map.get(node, [])
        if not kids:
            return 0
        return 1 + max(max_depth(k) for k in kids)
    result = {n: 0 for n in nodes}
    visited = set()
    def mark_component(node: str, depth_val: int):
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in visited: continue
            visited.add(cur)
            result[cur] = min(2, depth_val)
            stack.extend(children_map.get(cur, []))
            p = parent_map.get(cur, "")
            if p: stack.append(p)
    for r in roots:
        d = max_depth(r)
        mark_component(r, d)
    for n in nodes:
        if n not in visited:
            d = max_depth(n)
            mark_component(n, d)
    return pd.Series([result[nn] for nn in df["name_norm"]], index=df.index, dtype="Int64")

def parse_egg_groups(val) -> list[str]:
    if isinstance(val, list):
        tokens = val
    elif pd.isna(val):
        tokens = []
    else:
        tokens = str(val).split("~")
    out = []
    for t in tokens:
        tok = t.strip().lower()
        if not tok:
            continue
        if tok == "monter":
            tok = "monster"
        out.append(tok)
    return list(dict.fromkeys(out).keys())

def first_non_null(series: pd.Series):
    for v in series:
        if pd.notna(v):
            return v
    return np.nan

def collapse_sidecar(df2: pd.DataFrame) -> pd.DataFrame:
    if "egg_groups" not in df2.columns:
        df2["egg_groups"] = np.nan
    wanted = ["name", "base_experience", "egg_cycles", "base_happiness",
              "can_evolve", "evolves_from", "shape", "color", "egg_groups",
              "Hp", "attack", "sp_attack", "defense", "sp_defense", "speed"]
    present = [c for c in wanted if c in df2.columns]
    df2 = df2[present].copy()
    df2 = normalize_columns(df2)
    df2["name"] = df2["name"].map(canonicalize_name)
    if "evolves_from" in df2.columns:
        df2["evolves_from"] = df2["evolves_from"].map(canonicalize_name)
    df2["egg_groups"] = df2["egg_groups"].map(lambda v: v if isinstance(v, list) else v)
    df2["name_norm"] = df2["name"].map(normalize_name)
    agg_map = {c: first_non_null for c in present if c not in {"egg_groups"}}
    agg_map["egg_groups"] = lambda s: "~".join(
        [x for x in (str(v) for v in s if pd.notna(v)) if x and x.lower() != "nan"]
    )
    g = df2.groupby("name_norm", as_index=False).agg(agg_map)
    g["egg_groups"] = g["egg_groups"].map(parse_egg_groups)
    if "can_evolve" in g.columns:
        def to_bool(v):
            if isinstance(v, (bool, np.bool_)): return bool(v)
            if pd.isna(v): return np.nan
            s = str(v).strip().lower()
            if s in {"1","true","yes","y"}: return True
            if s in {"0","false","no","n"}: return False
            try: return bool(int(float(s)))
            except Exception: return np.nan
        g["can_evolve"] = g["can_evolve"].map(to_bool)
    for col in ["base_experience", "egg_cycles", "base_happiness"] + STAT_COLS:
        if col in g.columns:
            g[col] = g[col].map(coerce_int)
    return g

def build_evolves_to_map(right_collapsed: pd.DataFrame, allowed_norms: set[str]) -> dict[str, list[str]]:
    if "evolves_from" not in right_collapsed.columns:
        return {}
    rc = right_collapsed.copy()
    rc["evolves_from_norm"] = rc["evolves_from"].map(normalize_name)
    rc["child_norm"] = rc["name_norm"]
    m: dict[str, list[str]] = {}
    for parent_norm, sub in rc.groupby("evolves_from_norm", dropna=False):
        if not parent_norm:
            continue
        valid_children = [(str(n), cn) for n, cn in zip(sub["name"], sub["child_norm"])
                          if pd.notna(n) and isinstance(cn, str) and cn in allowed_norms]
        if not valid_children:
            continue
        dedup_names = list(dict.fromkeys([n for (n, _) in valid_children]).keys())
        m[parent_norm] = dedup_names
    return m

def run(in_path: Path, side_path: Path | None, out_path: Path):
    left = pd.read_parquet(in_path)
    left = normalize_columns(left)
    left["name"] = left["name"].map(canonicalize_name)
    if "evolves_from" in left.columns:
        left["evolves_from"] = left["evolves_from"].map(canonicalize_name)
    if "evolves_to" in left.columns:
        left["evolves_to"] = left["evolves_to"].map(canonicalize_name_list)
    required = ["name","pokedex_number","generation","abilities","type1","type2",
                "weight_kg","is_legendary","base_total",
                "base_egg_steps","base_happiness","capture_rate","experience_growth",
                "height_m","percentage_male"]
    missing = [c for c in required if c not in left.columns]
    if missing:
        raise SystemExit(f"Missing expected columns in left parquet: {missing}")
    left["is_legendary"] = to_bool_from_01(left["is_legendary"])
    left["experience_growth"] = map_experience_growth(left)
    left["no_in_generation"] = compute_no_in_generation(left)
    left["percentage_in_generation"] = compute_pct_in_generation(left)
    left["name_norm"] = left["name"].map(normalize_name)
    if side_path and Path(side_path).exists():
        right = pd.read_parquet(side_path)
        right = normalize_columns(right)
        right["name"] = right["name"].map(canonicalize_name)
        if "evolves_from" in right.columns:
            right["evolves_from"] = right["evolves_from"].map(canonicalize_name)
        right_collapsed = collapse_sidecar(right)
        merged = left.merge(
            right_collapsed,
            on="name_norm",
            how="left",
            suffixes=("", "_side"),
        )
        def attach_fill_from_left(df: pd.DataFrame, col: str) -> pd.DataFrame:
            if col not in df.columns:
                df[col] = np.nan
            sidecol = f"{col}_side"
            if sidecol in df.columns:
                df[col] = df[col].combine_first(df[sidecol])
                df = df.drop(columns=[sidecol])
            return df
        def attach_prefer_side(df: pd.DataFrame, col: str) -> pd.DataFrame:
            if col not in df.columns:
                df[col] = np.nan
            sidecol = f"{col}_side"
            if sidecol in df.columns:
                df[col] = df[sidecol].combine_first(df[col])
                df = df.drop(columns=[sidecol])
            return df
        for c in ["base_experience","egg_cycles","base_happiness",
                  "can_evolve","evolves_from","shape", "color", "egg_groups","name"]:
            merged = attach_fill_from_left(merged, c)
        for c in STAT_COLS:
            merged = attach_prefer_side(merged, c)
        allowed_norms = set(merged["name_norm"].unique())
        evol_to_map = build_evolves_to_map(right_collapsed, allowed_norms)
        merged["evolves_to"] = merged["name_norm"].map(lambda k: evol_to_map.get(k, []))
        merged["evolution_stages"] = compute_evolution_stages_by_depth(merged)
        left = merged.drop(columns=["name_norm"])
    else:
        for c in ["base_experience","egg_cycles","base_happiness",
                  "can_evolve","evolves_from","shape", "color", "egg_groups"]:
            if c not in left.columns:
                left[c] = np.nan
        if "evolves_to" not in left.columns:
            left["evolves_to"] = [[] for _ in range(len(left))]
        left["evolution_stages"] = compute_evolution_stages_by_depth(left)
        left = left.drop(columns=["name_norm"])
    left["egg_groups"] = left["egg_groups"].map(
        lambda v: v if isinstance(v, list) else parse_egg_groups(v)
    )
    for col in ["base_experience","egg_cycles","base_happiness"]:
        left[col] = left[col].map(coerce_int).astype("Int64")
    for col in STAT_COLS:
        if col in left.columns:
            left[col] = left[col].map(coerce_int).astype("Int64")
        else:
            left[col] = pd.Series([np.nan]*len(left), dtype="Int64")
    base_total_series = (
        left["Hp"].fillna(0) +
        left["attack"].fillna(0) +
        left["sp_attack"].fillna(0) +
        left["defense"].fillna(0) +
        left["sp_defense"].fillna(0) +
        left["speed"].fillna(0)
    )
    has_all_stats = left[STAT_COLS].notna().all(axis=1)
    left["base_total"] = base_total_series.where(has_all_stats, np.nan).astype("Int64")
    left["can_evolve"] = left["can_evolve"].astype("boolean")
    left["experience_growth"] = left["experience_growth"].astype("string")
    seen, ordered = set(), []
    for c in DESIRED_ORDER:
        if c not in seen and c in left.columns:
            ordered.append(c); seen.add(c)
    out_df = left[ordered].copy()
    out_path = Path(out_path).with_suffix(".parquet")
    out_df.to_parquet(out_path, index=False, engine="pyarrow")
    side_note = "" if (side_path and Path(side_path).exists()) else " (WARNING: sidecar not found—stats not overridden)"
    print(f"Wrote: {out_path}  (rows={len(out_df):,}, cols={len(out_df.columns)}){side_note}")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="./data/pokemon.parquet",
                    help="Input parquet path (default: %(default)s)")
    ap.add_argument("--side", dest="side_path", default="./data/pokemon_2.parquet",
                    help="Right-side parquet to join by name (default: %(default)s)")
    ap.add_argument("--out", dest="out_path", default="./data/completed_pokemon_set.parquet",
                    help="Output Parquet path (default: %(default)s)")
    args = ap.parse_args(argv)
    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input parquet not found: {in_path}")
    side_path = Path(args.side_path) if args.side_path else None
    out_path = Path(args.out_path) if args.out_path else in_path
    run(in_path, side_path, out_path)

if __name__ == "__main__":
    main()
