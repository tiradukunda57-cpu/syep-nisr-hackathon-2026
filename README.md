# SmartYield Export & Post-Harvest Planner (SYEP)
NISR 2026 Big Data Hackathon · Team: <NAME 1 (Rwandan citizen)>, <NAME 2> · <University>

**Live app:** <Streamlit Cloud link> · **Demo video:** `demo/demo.webm` (under 3 min) · **Repo:** <GitHub link>

## 1. Problem and national alignment
<Paste the exact challenge/track statement you chose.> Rwanda's agricultural transformation (NST2, Vision 2050) depends on
reducing post-harvest losses and growing horticulture exports. Using NISR's own survey we ask: **where and at which stage are crops
lost after harvest, which crops are most exposed, and where is the storage gap?**

**Key findings from the NISR data** (reproduced by `prepare_data.py`, farmer x crop records, survey-weighted):
- Reported post-harvest loss is small overall (**about 0.5% of harvest**) but highly concentrated: **tomato 3.7%** (45% of growers report a loss), paddy rice 2.3%, vegetables about 1%.
- **Horticulture crops are essentially not stored** (tomato, avocado, pepper, French beans, cabbage, onion: about 0% of harvest in storage) - a clear cold-chain gap for export crops.
- Farm-gate prices differ widely by crop and district (see Prices tab), which drives the exporter margin.
- Limitation: losses are self-reported (likely under-reported) and sample sizes for avocado, pepper and French beans are small (n about 60 each); the app flags small samples.

## 2. Data used
| Data | Source | Used for |
|---|---|---|
| **Seasonal Agricultural Survey 2024, Seasons A, B, C - Production files** (RWA-NISR-SAS-2024-v01). Variables: s1q2 district, s2q4 crop, s2q22 harvest (kg), s2q26 sold, s2q28 price (RWF/kg), s2q36 stored, s2q37 storage facility, s2q45-s2q50 losses at harvest/transport/storage/processing/packaging/sales, plot_weight | NISR Microdata Portal, https://microdata.statistics.gov.rw/index.php/catalog/113 (public-use; access request approved) | loss analysis, loss-risk model, prices |
| District locations | approximate centroids typed in `prepare_data.py` | map markers |
| Cold-room list, export price, logistics cost, exchange rate | **scenario assumptions** (`data/cold_storage.csv`, user inputs) - not NISR data | storage optimiser, margin calculator |

**Files in this repo:** `data/processed/district_crop_season.csv` (district x crop x season aggregates - the dataset used by the app) and
`data/processed/crop_summary.csv`. Raw microdata is **not redistributed** (NISR terms); download it from the link above.

**Preprocessing (`prepare_data.py`):** one file format per survey file (no double counting); unit of analysis = farmer x crop x season, because
harvest, sales, storage and loss values repeat across a farmer's plots; negative loss codes treated as missing and unreported losses as 0;
records with loss > harvest (10) or implausible yield above 60,000 kg/ha (492) removed; price codes of 9999 or more treated as missing;
district names matched to the 30 districts; aggregates weighted with `plot_weight`.

## 3. What the app shows and how to use it
Sidebar: language (English/Kinyarwanda), user type, crop, district, season.
- **Farmer / Cooperative:** *Loss risk* - how much of this crop is lost, at which stage, and the model's chance that a farm with this crop, size and storage share reports a loss; *Cold storage* - nearest suitable cold rooms for a given volume.
- **Exporter:** *Prices & margin* - NISR farm-gate prices by district and an editable export margin calculator; plus storage and loss risk.
- **Planner (MINAGRI/NAEB):** *Priority map* of loss by district with distance to the nearest cold room, plus national crop overview.

## 4. Methods
- **Loss-risk model:** Gradient-boosting classifier (scikit-learn). The code uses XGBoost automatically when it is installed; the submitted model was trained with scikit-learn gradient boosting predicting whether a farmer-crop reports any post-harvest loss, from crop, district, season, harvest size, farm area and share stored. 80/20 stratified split; ROC-AUC about 0.86 (exact value in `model/meta.json`), baseline 0.5. Accuracy is close to the always-no-loss baseline because only about 9% report a loss, so AUC is the key metric.
- **Storage optimiser:** linear programming (SciPy HiGHS) allocates volume to cold rooms minimising distance + congestion, within free capacity.
- **Prices:** descriptive NISR medians. We deliberately do **not** forecast prices: the survey has one price per season, not a time series.

## 5. Run locally
```
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# optional: rebuild data/model from raw files unzipped into data/raw/
python prepare_data.py
streamlit run app.py
```

## 6. Limitations and next steps
Self-reported losses; small samples for some export crops; district centroids approximate (official boundaries: NISR Geodata portal);
cold-room capacities and export prices are scenarios; SMS/USSD access for farmers without smartphones; add NAEB export volumes and Meteo Rwanda weather when available.

## 7. Originality declaration and acknowledgements
We declare this submission is our own work and all sources are acknowledged: NISR SAS 2024 (above), open-source libraries (Streamlit, pandas,
NumPy, SciPy, scikit-learn, Plotly). AI-assistance disclosure: we used an AI assistant (Claude, by Anthropic) to help draft and debug code and documentation. The team chose the challenge and the problem framing, obtained the NISR data under its terms, ran the analyses, checked the results and take responsibility for this submission.
Data citation: National Institute of Statistics of Rwanda, Seasonal Agricultural Survey 2024, RWA-NISR-SAS-2024-v01, NISR Microdata Portal.
Per the competition rules, intellectual property in the submitted work is assigned to NISR. We will send NISR a copy of any report based on these data.
