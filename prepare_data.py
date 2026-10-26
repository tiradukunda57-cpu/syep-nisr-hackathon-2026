"""Build the SYEP datasets from NISR Seasonal Agricultural Survey (SAS) 2024 *Production* files.
Usage:  python prepare_data.py            (reads data/raw/**/*Production*.dta|.csv|.sav ; trains the loss-risk model)
        python prepare_data.py --demo     (SIMULATED data, only to preview the app; flagged in the UI)
Source: NISR Microdata Portal, RWA-NISR-SAS-2024-v01  https://microdata.statistics.gov.rw/index.php/catalog/113
Unit of analysis = farmer x crop x season (s2q22, s2q26, s2q36 and loss variables repeat across a farmer's plots)."""
import glob, json, os, re, sys
import numpy as np
import pandas as pd

RAW, OUT, MODEL = os.environ.get("SYEP_RAW", "data/raw"), "data/processed", "model"
STAGES = {"harvest": "s2q45", "transport": "s2q46", "storage": "s2q47",
          "processing": "s2q48", "packaging": "s2q49", "sales": "s2q50"}
NEED = ["segment_id", "s1q2", "s1q6", "s2q2", "s2q4", "cropcategory", "s2q22", "s2q26", "s2q28", "s2q36", "s2q37",
        "plot_weight"] + list(STAGES.values())
MAX_YIELD_KG_HA = 60000   # above this = data-entry error (99th pct is ~50,000)
HORT = ["Pepper", "Sweet pepper", "Avocado", "French beans", "Tomato", "Cabbage", "Onion"]  # export-relevant horticulture
# Approximate district centroids (lat, lon) used only for map display
DIST = {"Nyarugenge": (-1.97, 30.04), "Gasabo": (-1.89, 30.13), "Kicukiro": (-1.99, 30.10),
        "Nyanza": (-2.35, 29.75), "Gisagara": (-2.60, 29.85), "Nyaruguru": (-2.72, 29.55), "Huye": (-2.60, 29.74),
        "Nyamagabe": (-2.47, 29.40), "Ruhango": (-2.22, 29.78), "Muhanga": (-2.08, 29.75), "Kamonyi": (-2.00, 29.90),
        "Karongi": (-2.15, 29.38), "Rutsiro": (-1.95, 29.33), "Rubavu": (-1.68, 29.35), "Nyabihu": (-1.65, 29.50),
        "Ngororero": (-1.87, 29.62), "Rusizi": (-2.48, 28.91), "Nyamasheke": (-2.35, 29.10),
        "Rulindo": (-1.75, 29.99), "Gakenke": (-1.69, 29.78), "Musanze": (-1.50, 29.63), "Burera": (-1.45, 29.85),
        "Gicumbi": (-1.58, 30.07), "Rwamagana": (-1.95, 30.43), "Nyagatare": (-1.30, 30.33), "Gatsibo": (-1.58, 30.43),
        "Kayonza": (-1.88, 30.64), "Kirehe": (-2.25, 30.70), "Ngoma": (-2.15, 30.50), "Bugesera": (-2.21, 30.15)}
KEY = lambda s: re.sub(r"[^a-z]", "", str(s).lower())


def match_district(v):
    k = KEY(v)
    return next((n for n in DIST if KEY(n) in k), None)


def read_file(p):
    e = p.lower().rsplit(".", 1)[-1]
    if e == "dta":
        with pd.read_stata(p, iterator=True) as r:
            have = {c.lower(): c for c in r.variable_labels()}
        d = pd.read_stata(p, columns=[have[c] for c in NEED if c in have])  # reading a column subset avoids duplicate-label errors
    elif e == "csv":
        d = pd.read_csv(p, sep="\t", low_memory=False)
    else:
        d = pd.read_spss(p)  # needs pyreadstat
    d.columns = [c.strip().lower() for c in d.columns]
    return d


def load_raw():
    best = {}
    for f in glob.glob(f"{RAW}/**/*", recursive=True):
        b = os.path.basename(f).lower()
        if os.path.isfile(f) and "production" in b and b.endswith((".dta", ".csv", ".sav")):
            stem, rank = b.rsplit(".", 1)[0], {"dta": 0, "csv": 1, "sav": 2}[b.rsplit(".", 1)[1]]
            if stem not in best or rank < best[stem][0]: best[stem] = (rank, f)  # one format per file: never count twice
    if not best: sys.exit(f"No '*Production*' files found in {RAW}/ - unzip Dataset_SAS_2024.zip there (see README).")
    out = []
    for stem, (_, f) in sorted(best.items()):
        m = re.search(r"season([abc])", stem); d = read_file(f); d["season"] = m.group(1).upper() if m else "?"
        print(f"read {os.path.basename(f)}: {len(d)} plot rows, season {d['season'].iloc[0]}"); out.append(d)
    return pd.concat(out, ignore_index=True, sort=False)


def clean(raw):
    num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw else pd.Series(np.nan, index=raw.index)
    d = pd.DataFrame({"season": raw["season"], "seg": raw["segment_id"].astype(str), "farmer": raw["s1q6"].astype(str),
                      "crop": raw["s2q4"].astype(str).str.strip(), "category": raw["cropcategory"].astype(str).str.strip(),
                      "district": raw["s1q2"].map(match_district), "area_sqm": num("s2q2"), "weight": num("plot_weight"),
                      "harvest_kg": num("s2q22"), "stored_kg": num("s2q36"), "sold_kg": num("s2q26"),
                      "price_rwf_kg": num("s2q28"), "storage_type": raw["s2q37"].astype(str).replace({"nan": "none"})})
    for k, c in STAGES.items(): d[f"loss_{k}_kg"] = num(c).where(lambda s: s >= 0).fillna(0)  # negatives = missing code; unreported = 0
    keys = ["season", "seg", "farmer", "crop"]
    f = d.groupby(keys, observed=True).agg(category=("category", "first"), district=("district", "first"), area_sqm=("area_sqm", "sum"),
                                           weight=("weight", "mean"), harvest_kg=("harvest_kg", "first"), stored_kg=("stored_kg", "first"),
                                           sold_kg=("sold_kg", "first"), price_rwf_kg=("price_rwf_kg", "first"),
                                           storage_type=("storage_type", "first"),
                                           **{f"loss_{k}_kg": (f"loss_{k}_kg", "first") for k in STAGES}).reset_index()
    n0 = len(f); f = f[f.district.notna() & (f.harvest_kg > 0)]
    f["total_loss_kg"] = f[[f"loss_{k}_kg" for k in STAGES]].sum(axis=1)
    bad_loss = f.total_loss_kg > f.harvest_kg; bad_yield = f.harvest_kg / (f.area_sqm / 1e4) > MAX_YIELD_KG_HA
    f = f[~bad_loss & ~bad_yield].copy()
    f["price_rwf_kg"] = f.price_rwf_kg.where((f.price_rwf_kg > 0) & (f.price_rwf_kg < 9999))  # 9999+ = missing code
    f["weight"] = f.weight.fillna(1); f["any_loss"] = (f.total_loss_kg > 0).astype(int)
    f["stored_share"] = (f.stored_kg.fillna(0) / f.harvest_kg).clip(0, 1)
    print(f"farmer-crop records: {n0} -> kept {len(f)} (removed {int(bad_loss.sum())} loss>harvest, {int(bad_yield.sum())} implausible yield, rest no district/harvest)")
    return f


def aggregate(f):
    cols = ["harvest_kg", "stored_kg", "total_loss_kg"] + [f"loss_{k}_kg" for k in STAGES]
    w = f.copy(); w[cols] = w[cols].mul(w.weight, axis=0)
    g, k = w.groupby(["district", "crop", "season"], observed=True), ["district", "crop", "season"]
    a = g[cols].sum().join(f.groupby(k, observed=True).agg(n_farmer_crops=("any_loss", "size"), n_any_loss=("any_loss", "sum"),
                                                          price_median_rwf_kg=("price_rwf_kg", "median")))
    return a.reset_index()


def train_model(f):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    top = f.crop.value_counts().index[:60].tolist() or ["?"]
    cats = {"crop": sorted(top), "district": sorted(DIST), "season": sorted(f.season.unique())}
    X = lambda d: np.c_[[pd.Categorical(d[c].where(d[c].isin(cats[c]), cats[c][0]), categories=cats[c]).codes for c in cats]
                        + [np.log1p(d.harvest_kg), np.log1p(d.area_sqm.fillna(d.area_sqm.median())), d.stored_share]].T
    Xtr, Xte, ytr, yte = train_test_split(X(f), f.any_loss, test_size=.2, random_state=0, stratify=f.any_loss)
    kw = dict(n_estimators=200, max_depth=4, learning_rate=.08)
    try:
        from xgboost import XGBClassifier; m, kind = XGBClassifier(**kw, eval_metric="logloss"), "xgboost"
    except Exception:
        m, kind = GradientBoostingClassifier(**kw), "gradient_boosting"
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
    os.makedirs(MODEL, exist_ok=True)
    if kind == "xgboost": m.save_model(f"{MODEL}/loss_model.json")
    else:
        import joblib; joblib.dump(m, f"{MODEL}/loss_model.joblib")
    meta = dict(kind=kind, cats=cats, auc=round(float(roc_auc_score(yte, p)), 3), accuracy=round(float(accuracy_score(yte, p > .5)), 3),
                baseline_accuracy=round(float(1 - yte.mean()), 3), prevalence=round(float(f.any_loss.mean()), 3), n_train=len(ytr), n_test=len(yte))
    json.dump(meta, open(f"{MODEL}/meta.json", "w"), indent=1); print("model:", meta); return meta


def demo():
    rng = np.random.default_rng(1); n = 6000
    crops = ["Maize", "Bush bean", "Tomato", "Avocado", "French beans", "Irish potato", "Pepper"]
    f = pd.DataFrame({"district": rng.choice(list(DIST), n), "crop": rng.choice(crops, n), "season": rng.choice(list("ABC"), n),
                      "area_sqm": rng.lognormal(7, .8, n), "weight": 1.0}); f["category"] = f.crop
    f["harvest_kg"] = f.area_sqm * rng.uniform(.05, .4, n)
    for k in STAGES: f[f"loss_{k}_kg"] = f.harvest_kg * rng.uniform(0, .03, n) * (rng.random(n) < .05)
    f["total_loss_kg"] = f[[f"loss_{k}_kg" for k in STAGES]].sum(axis=1); f["any_loss"] = (f.total_loss_kg > 0).astype(int)
    f["stored_share"] = 0.0; f["stored_kg"] = 0.0; f["sold_kg"] = f.harvest_kg * .5; f["price_rwf_kg"] = rng.uniform(150, 900, n)
    f["storage_type"] = "none"; return f


if __name__ == "__main__":
    sim = "--demo" in sys.argv
    f = demo() if sim else clean(load_raw())
    os.makedirs(OUT, exist_ok=True)
    aggregate(f).to_csv(f"{OUT}/district_crop_season.csv", index=False)  # aggregated, shareable on GitHub
    cs = f.assign(w=f.weight).groupby("crop").apply(lambda x: pd.Series({
        "n_farmer_crops": len(x), "loss_pct": 100 * (x.total_loss_kg * x.w).sum() / (x.harvest_kg * x.w).sum(),
        "share_reporting_loss_pct": 100 * x.any_loss.mean(), "stored_pct": 100 * (x.stored_kg.fillna(0) * x.w).sum() / (x.harvest_kg * x.w).sum(),
        "price_median_rwf_kg": x.price_rwf_kg.median()}), include_groups=False).round(2).reset_index()
    cs.sort_values("n_farmer_crops", ascending=False).to_csv(f"{OUT}/crop_summary.csv", index=False)
    train_model(f)
    flag = f"{OUT}/SIMULATED"; open(flag, "w").write("simulated") if sim else (os.remove(flag) if os.path.exists(flag) else None)
    print(f"{'SIMULATED' if sim else 'REAL'} data written | records={len(f)} districts={f.district.nunique()} crops={f.crop.nunique()}")
    print("weighted loss % of harvest:", round(100 * (f.total_loss_kg * f.weight).sum() / (f.harvest_kg * f.weight).sum(), 2))
