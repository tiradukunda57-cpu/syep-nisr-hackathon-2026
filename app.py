"""SYEP - SmartYield Export & Post-Harvest Planner | NISR 2026 Big Data Hackathon.
Data: NISR Seasonal Agricultural Survey 2024 (seasons A, B, C), prepared by prepare_data.py."""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog

from prepare_data import DIST, HORT, STAGES

st.set_page_config(page_title="SYEP", page_icon="🌾", layout="wide")
P = "data/processed"
L = {"en": dict(title="SmartYield Export & Post-Harvest Planner", persona="User type", farmer="Farmer / Cooperative",
                exporter="Exporter", planner="Planner (MINAGRI/NAEB)", district="District", crop="Crop", season="Season",
                loss="Loss risk", store="Cold storage", price="Prices & margin", map="Priority map",
                harvest="Harvest (t, weighted)", lossp="Post-harvest loss", anyl="Farmers reporting a loss", stored="Share of harvest stored",
                stages="At which stage is the crop lost? (% of harvest)", pred="Chance this farmer reports a post-harvest loss",
                avg="average for all crops", tons="Harvest volume (t)", area="Farm area for this crop (ha)", share="Share you will store (%)",
                frac="Share of harvest to put in cold storage (%)", alloc="Recommended allocation", nostore="Not enough free cold-room capacity.",
                farmp="Farm-gate price, median (RWF/kg, NISR)", expp="Export price (USD/kg)", logi="Logistics cost (USD/kg)",
                fx="Exchange rate (RWF/USD)", margin="Estimated margin per kg (RWF)", n="Survey records",
                small="Small sample - interpret with care.", note="Cold-room list, export price, logistics and exchange rate are editable scenario assumptions, not NISR data.",
                ov="Crop overview (national)"),
     "rw": dict(title="SmartYield: Gutegura Isarura n'Ibyoherezwa Hanze", persona="Ukoresha", farmer="Umuhinzi / Koperative",
                exporter="Uwohereza hanze", planner="Umupanga (MINAGRI/NAEB)", district="Akarere", crop="Igihingwa", season="Igihembwe",
                loss="Ibyago by'igihombo", store="Ububiko bukonje", price="Ibiciro n'inyungu", map="Ikarita y'ibyihutirwa",
                harvest="Umusaruro (t)", lossp="Igihombo nyuma yo gusarura", anyl="Abahinzi bavuze igihombo", stored="Umusaruro ubitswe",
                stages="Igihombera he? (% by'umusaruro)", pred="Amahirwe y'uko umuhinzi avuga igihombo", avg="impuzandengo y'ibihingwa byose",
                tons="Ingano y'umusaruro (t)", area="Ubuso bw'iki gihingwa (ha)", share="Igice uzabika (%)",
                frac="Igice cyo kubika mu bukonje (%)", alloc="Ububiko bwateguwe", nostore="Ubushobozi bw'ububiko bukonje ntibuhagije.",
                farmp="Igiciro cy'umuhinzi, hagati (RWF/kg, NISR)", expp="Igiciro cyo hanze (USD/kg)", logi="Ikiguzi cy'ubwikorezi (USD/kg)",
                fx="Igipimo cy'ivunjisha (RWF/USD)", margin="Inyungu ikekwa kuri kg (RWF)", n="Umubare w'ibisubizo",
                small="Ibisubizo ni bike - sobanura witonze.", note="Urutonde rw'ububiko, igiciro cyo hanze, ubwikorezi n'ivunjisha ni ibigereranyo wakosora, si amakuru ya NISR.",
                ov="Incamake y'ibihingwa (igihugu)")}
lang = st.sidebar.radio("🌐", ["en", "rw"], format_func=lambda x: {"en": "English", "rw": "Kinyarwanda"}[x])
t = L[lang]
if not os.path.exists(f"{P}/district_crop_season.csv"):
    st.error("Processed data missing. Run `python prepare_data.py` (see README)."); st.stop()
SIM = os.path.exists(f"{P}/SIMULATED")


@st.cache_data
def load():
    return (pd.read_csv(f"{P}/district_crop_season.csv"), pd.read_csv(f"{P}/crop_summary.csv"), pd.read_csv("data/cold_storage.csv"))


@st.cache_resource
def load_model():
    try:
        meta = json.load(open("model/meta.json"))
        if meta["kind"] == "xgboost":
            from xgboost import XGBClassifier
            m = XGBClassifier(); m.load_model("model/loss_model.json")
        else:
            import joblib; m = joblib.load("model/loss_model.joblib")
        return m, meta
    except Exception as e:
        return None, str(e)


agg, cropsum, stores = load()
st.title("🌾 " + t["title"])
if SIM:
    st.warning("⚠️ SIMULATED DEMO DATA - run prepare_data.py on the NISR SAS 2024 files.")
persona = st.sidebar.selectbox(t["persona"], ["farmer", "exporter", "planner"], format_func=lambda k: t[k])
hort = [c for c in HORT if c in set(agg.crop)]
crops = hort + [c for c in agg.groupby("crop").harvest_kg.sum().sort_values(ascending=False).index if c not in hort]
crop = st.sidebar.selectbox(t["crop"], crops)
dname = st.sidebar.selectbox(t["district"], sorted(DIST))
season = st.sidebar.selectbox(t["season"], ["All"] + sorted(agg.season.unique()))
st.sidebar.caption("Source: NISR Seasonal Agricultural Survey 2024 (A, B, C)")
A = agg[(agg.crop == crop) & ((agg.season == season) | (season == "All"))]
sel = A[A.district == dname]
if sel.empty:
    sel = A  # no records in this district: fall back to national figures for the crop
    st.caption(f"ℹ️ {dname}: no survey records for {crop} - showing national figures.")


def hav(a, b, c, d):
    r = np.pi / 180
    x = np.sin((c - a) * r / 2) ** 2 + np.cos(a * r) * np.cos(c * r) * np.sin((d - b) * r / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(x))


def tab_loss():
    h, n = sel.harvest_kg.sum(), int(sel.n_farmer_crops.sum())
    if h <= 0:
        st.info("No data."); return
    stg = {k: sel[f"loss_{k}_kg"].sum() / h * 100 for k in STAGES}
    c = st.columns(5)
    c[0].metric(t["n"], n); c[1].metric(t["harvest"], f"{h / 1000:,.0f}"); c[2].metric(t["lossp"], f"{sel.total_loss_kg.sum() / h * 100:.2f}%")
    c[3].metric(t["anyl"], f"{sel.n_any_loss.sum() / max(n, 1) * 100:.1f}%"); c[4].metric(t["stored"], f"{sel.stored_kg.sum() / h * 100:.2f}%")
    if n < 30: st.caption("⚠️ " + t["small"])
    st.plotly_chart(px.bar(x=list(stg), y=list(stg.values()), labels={"x": "", "y": "%"}, title=t["stages"]), width="stretch")
    m, meta = load_model()
    if m is None:
        st.info("Model not found - run `python prepare_data.py` to train it."); return
    cats = meta["cats"]
    a, b, s = st.columns(3)
    vol, ha, shr = a.number_input(t["tons"], .01, 500., 1., key="v1"), b.number_input(t["area"], .01, 100., .3), s.slider(t["share"], 0, 100, 0) / 100
    cc = lambda col, v: cats[col].index(v) if v in cats[col] else 0
    x = [[cc("crop", crop), cc("district", dname), cc("season", season if season != "All" else cats["season"][0]),
          np.log1p(vol * 1000), np.log1p(ha * 1e4), shr]]
    p = float(m.predict_proba(x)[0, 1])
    st.metric(t["pred"], f"{p * 100:.1f}%", f"{meta['prevalence'] * 100:.1f}% {t['avg']}", delta_color="off")
    with st.expander("📊 Model evaluation (20% hold-out, " + ("XGBoost" if meta["kind"] == "xgboost" else "GradientBoosting") + ")"):
        e = st.columns(3)
        e[0].metric("ROC-AUC", meta["auc"]); e[1].metric("Baseline AUC", 0.5); e[2].metric("Base rate", f"{meta['prevalence'] * 100:.1f}%")
        st.caption(f"Target: farmer-crop reports any post-harvest loss. Train {meta['n_train']:,} / test {meta['n_test']:,}. "
                   f"Accuracy {meta['accuracy']} vs {meta['baseline_accuracy']} for always-'no loss' (classes are imbalanced, so AUC is the key metric).")


def tab_store():
    st.caption(f"NISR SAS 2024: {crop} - {cropsum.set_index('crop').stored_pct.get(crop, float('nan')):.2f}% of harvest is stored nationally.")
    vol = st.number_input(t["tons"], .1, 500., 5., key="v2")
    fr = st.slider(t["frac"], 5, 100, 100) / 100 * vol
    lat, lon = DIST[dname]
    dist = hav(lat, lon, stores.lat.values, stores.lon.values); util = stores.used_pct.values / 100
    r = linprog(dist + 60 * util, A_eq=[np.ones(len(stores))], b_eq=[fr], bounds=[(0, f) for f in stores.cap_t.values * (1 - util)], method="highs")
    if not r.success:
        st.error(t["nostore"]); return
    out = stores.assign(km=dist.round(1), assigned_t=r.x.round(2))
    st.subheader(t["alloc"]); st.dataframe(out[out.assigned_t > 0][["name", "assigned_t", "km", "used_pct"]], hide_index=True)
    fig = go.Figure(go.Scattermap(lat=stores.lat, lon=stores.lon, mode="markers+text", text=stores.name, name="Cold rooms",
                                  marker=dict(size=15, color=["green" if u < 60 else "orange" if u < 85 else "red" for u in stores.used_pct])))
    for _, q in out[out.assigned_t > 0].iterrows():
        fig.add_trace(go.Scattermap(lat=[lat, q.lat], lon=[lon, q.lon], mode="lines", line=dict(width=3), showlegend=False))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=-1.95, lon=29.9), zoom=7), height=450, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch"); st.caption(t["note"] + " Straight-line distances.")


def tab_price():
    pr = A.dropna(subset=["price_median_rwf_kg"])
    if pr.empty:
        st.info("No price records for this crop."); return
    local = pr[pr.district == dname].price_median_rwf_kg.median(); nat = pr.price_median_rwf_kg.median()
    farm = float(local if pd.notna(local) else nat)
    st.metric(t["farmp"], f"{farm:,.0f}", f"national {nat:,.0f}", delta_color="off")
    d = pr.groupby("district").price_median_rwf_kg.median().sort_values().reset_index()
    st.plotly_chart(px.bar(d, x="price_median_rwf_kg", y="district", orientation="h", title=t["farmp"]), width="stretch")
    c = st.columns(3)
    e, lg, fx = c[0].number_input(t["expp"], 0., 20., 2.0), c[1].number_input(t["logi"], 0., 10., .6), c[2].number_input(t["fx"], 500., 3000., 1450.)
    st.metric(t["margin"], f"{(e - lg) * fx - farm:,.0f}"); st.caption(t["note"])


def tab_map():
    b = A.groupby("district")[["harvest_kg", "total_loss_kg", "stored_kg"]].sum().reset_index()
    b = b[b.harvest_kg > 0]; b["loss_pct"] = b.total_loss_kg / b.harvest_kg * 100; b["loss_t"] = b.total_loss_kg / 1000
    b["lat"], b["lon"] = b.district.map(lambda x: DIST[x][0]), b.district.map(lambda x: DIST[x][1])
    b["nearest_cold_room_km"] = [hav(r.lat, r.lon, stores.lat.values, stores.lon.values).min().round(1) for r in b.itertuples()]
    st.plotly_chart(px.scatter_map(b, lat="lat", lon="lon", size="loss_t", color="loss_pct", hover_name="district", size_max=30, center=dict(lat=-1.95, lon=29.9), zoom=7, height=450,
                                   color_continuous_scale="RdYlGn_r", map_style="open-street-map"), width="stretch")
    st.dataframe(b.sort_values("loss_t", ascending=False)[["district", "loss_t", "loss_pct", "nearest_cold_room_km"]].round(2), hide_index=True)
    st.subheader(t["ov"]); st.dataframe(cropsum[cropsum.crop.isin(hort + ["Paddy rice", "Maize", "Bush bean"])], hide_index=True)


views = {"farmer": [("loss", tab_loss), ("store", tab_store)], "exporter": [("price", tab_price), ("store", tab_store), ("loss", tab_loss)],
         "planner": [("map", tab_map), ("loss", tab_loss), ("store", tab_store), ("price", tab_price)]}[persona]
for tab, (name, fn) in zip(st.tabs([t[n] for n, _ in views]), views):
    with tab:
        fn()
