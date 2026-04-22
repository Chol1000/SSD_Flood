# South Sudan Flood Early Warning System

**County-level flood prediction across all 79 administrative counties of South Sudan using satellite-derived climate and terrain data (2011–2025)**

> **Author:** Chol Atem Giet Monykuch · [c.monykuch@alustudent.com](mailto:c.monykuch@alustudent.com) · African Leadership University  
> **Model:** Logistic Regression · AUC-ROC = 0.9601 (95% CI: 0.935–0.982) · F1 = 0.7581 · Precision = 0.7705 · Recall = 0.7460  
> **Live App:** [https://ssd-flood.streamlit.app](https://ssd-flood.streamlit.app)  
> **Source Code:** [https://github.com/Chol1000/SSD_Flood](https://github.com/Chol1000/SSD_Flood)

---

## Overview

South Sudan experiences catastrophic annual flooding driven by the Nile river system and the Sudd — the world's largest tropical wetland. Since 2019, flooding has displaced over one million people annually, yet no systematic county-level predictive early warning system exists.

This project presents the **first machine learning flood prediction framework covering all 79 South Sudan counties**, trained on 14,220 county-month satellite observations spanning 15 years (2011–2025). The system produces calibrated flood risk probabilities at county-month granularity and is deployed as a real-time interactive web application.

---

## Live Demo

**[https://ssd-flood.streamlit.app](https://ssd-flood.streamlit.app)**

The deployed early warning application provides four interactive views:

| Tab | What it does |
|---|---|
| **Flood Risk Prediction** | Select any of the 79 counties — climate inputs auto-fill with historical medians. Adjust sliders to current or forecast conditions and click **Predict** to get a calibrated flood probability and alert level (Critical / High / Moderate / Low). |
| **County Risk Map** | Choropleth bar chart showing historical flood rates across all 79 counties, colour-coded by risk tier. Identifies the most structurally vulnerable counties at a glance. |
| **Historical Analysis** | Per-county monthly flood time series from 2011 to 2025. Select a county to see its full flood history alongside seasonal and annual trend breakdowns. |
| **Model Performance** | Full transparency panel — test-set metrics, confusion matrices, ROC/PR curves, feature importance, ablation results, and the persistence baseline comparison. |

> The app runs on Streamlit Community Cloud (free tier). If it shows a **"This app is sleeping"** screen, click **Wake up** and it will be ready within 30 seconds.

---

## Key Results

| Model | AUC-ROC | 95% CI | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✓ deployed | **0.9601** | **[0.935, 0.982]** | **0.7581** | **0.7705** | **0.7460** | **0.935** |
| Random Forest | 0.9668 | [0.948, 0.983] | 0.5818 | 0.4706 | 0.7619 | 0.506 |
| XGBoost | 0.9581 | [0.935, 0.977] | 0.3881 | 0.2537 | 0.8254 | 0.526 |
| LightGBM | 0.9564 | [0.930, 0.978] | 0.5213 | 0.3920 | 0.7778 | 0.373 |
| Persistence baseline | 0.8692 | — | 0.7581 | 0.7705 | 0.7460 | — |

Bootstrap 95% CIs from 1,000 resample iterations. Logistic Regression is selected for deployment based on precision (77.05%) — the highest among all models. The AUC advantage over persistence (+0.091, DeLong z = 4.15, p < 0.0001) enables continuous risk scoring; the binary confusion matrix at t = 0.935 is identical to persistence, a finding reported transparently as a scientific result.

---

## Dataset

| Property | Value |
|---|---|
| Total observations | 14,220 county-month records |
| Counties | 79 (all South Sudan administrative units) |
| Time period | January 2011 – December 2025 (15 years) |
| Flood events | 649 (4.56%) — severe 1:20.9 class imbalance |
| Missing values | 191 (water_fraction only, 1.34%) |
| Sources | CHIRPS, TerraClimate, MODIS, SRTM, ESA WorldCover, JRC GSW, UNOCHA |

**15 features used (after leakage audit):**

| Category | Features |
|---|---|
| Climate (dynamic) | `rainfall_mm`, `soil_moisture_mm`, `max_temperature_celsius`, `min_temperature_celsius`, `vapor_pressure_deficit_kPa`, `ndvi` |
| Terrain (static) | `wetland_fraction`, `elevation_m`, `slope_deg` |
| Temporal lag | `flood_prev_month` |
| Engineered | `temp_range`, `wetness_index`, `rain_wetland`, `month_sin`, `month_cos` |

**Excluded (label leakage):** `water_fraction` — `water_fraction >= 0.01` perfectly separates all 649 flood events from all 13,571 non-flood events. It encodes the label definition, not an independent predictor.

> The full dataset (14,220 observations, 16 variables) will be made publicly available on Kaggle upon publication.

---

## Methodology

### Core Design Principles

1. **Explicit leakage audit** — every feature tested before modelling; `water_fraction` permanently excluded
2. **SMOTE inside ImbPipeline** — oversampling applied only within each training fold, never to validation or test data
3. **TimeSeriesSplit (5-fold)** — chronological ordering strictly preserved throughout all experiments
4. **Threshold optimisation on CV only** — F1-optimal threshold (0.935) selected on cross-validation folds and applied exactly once to the held-out test set
5. **Persistence baseline comparison** — zero-parameter benchmark with DeLong and McNemar statistical significance tests

### Ablation Study

| Feature Set | N Features | CV AUC | Test AUC | Δ vs Full | Test F1 |
|---|---|---|---|---|---|
| Full (15 features) | 15 | 0.948 ± 0.051 | 0.958 | — | 0.388 |
| No temporal lag | 14 | 0.913 ± 0.077 | 0.880 | −0.078 | 0.241 |
| Climate + engineered only | 11 | 0.836 ± 0.076 | 0.812 | −0.146 | 0.167 |

Climate-only AUC = 0.812 confirms the model learns genuine climate-flood relationships beyond simple persistence.

### Top Feature Importances (Logistic Regression, normalised absolute coefficients)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `slope_deg` | 39.5% |
| 2 | `flood_prev_month` | 18.8% |
| 3 | `vapor_pressure_deficit_kPa` | 9.0% |
| 4 | `elevation_m` | 6.8% |
| 5 | `rainfall_mm` | 6.3% |

Slope dominance reflects geographic discrimination — structurally flood-prone low-lying counties — rather than temporal prediction. The stratified analysis (24 flood-experienced counties only) confirms AUC = 0.9010 on the harder subset, still outperforming persistence by +0.042.

---

## Publication Figures

All 16 figures were generated directly from the fitted pipelines and stored in `figures/` at 200 dpi.

| File | Description |
|---|---|
| `fig01_study_area.png` | County-level choropleth: historical flood rates (2011–2025) |
| `fig02_class_imbalance.png` | Target class distribution (1:20.9 imbalance) |
| `fig03_temporal_patterns.png` | Seasonal and annual flood rate trends |
| `fig04_county_risk.png` | Top-15 highest-risk counties |
| `fig05_feature_distributions.png` | Predictor distributions: flood vs no-flood months |
| `fig06_correlation_10feat.png` | Pearson correlation — 10 honest predictor variables |
| `fig07_correlation_15feat.png` | Pearson correlation — full 15-feature set |
| `fig08_cv_stability.png` | Cross-validation stability across 5 TimeSeriesSplit folds |
| `fig09_cv_model_comparison.png` | CV model comparison (AUC, F1, Precision, Recall) |
| `fig10_roc_pr_curves.png` | ROC and Precision-Recall curves on 2024–2025 test set |
| `fig11_confusion_matrices.png` | Confusion matrices for all 4 models |
| `fig12_ablation_study.png` | Ablation study — feature group contributions |
| `fig13_feature_importance.png` | LR normalised coefficients + ensemble comparison |
| `fig14_cv_thresholds.png` | Distribution of CV-optimal decision thresholds |
| `fig15_baseline_comparison.png` | Persistence baseline vs Logistic Regression |
| `fig16_calibration_curves.png` | Probability calibration curves (5 bins) |

---

## Project Structure

```
ssd_flood/
├── app.py                              # Streamlit early warning application
├── train.py                            # Full training pipeline — reproduces all results
├── flood_prediction_south_sudan.ipynb  # Interactive analysis notebook
├── requirements.txt                    # Python dependencies
├── .streamlit/
│   └── config.toml                     # Streamlit theme and server config
├── figures/                            # 16 publication figures (fig01–fig16)
│   ├── fig01_study_area.png
│   ├── fig02_class_imbalance.png
│   ├── ...
│   └── fig16_calibration_curves.png
└── model/
    ├── best_model.pkl                  # Trained Logistic Regression pipeline (serialised)
    ├── metadata.json                   # All metrics, thresholds, feature importance, significance tests
    ├── counties.json                   # County names and metadata
    ├── county_defaults.json            # Per-county median feature values for app auto-fill
    ├── county_flood_history.csv        # Historical flood rates per county (2011–2025)
    ├── feature_stats.json              # Feature statistics for app display
    └── monthly_flood_data.csv          # Monthly aggregate flood statistics
```

> **Note:** The dataset CSV (`south_sudan_flood_dataset_2011_2025.csv`) is excluded from this repository via `.gitignore` due to file size. It will be made available on Kaggle upon publication.

---

## Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/Chol1000/SSD_Flood.git
cd SSD_Flood

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit early warning app
streamlit run app.py
```

The app runs at `http://localhost:8501` and provides four views:
- **Flood Risk Prediction** — select any county, adjust climate inputs, get a calibrated flood probability
- **County Risk Map** — historical flood rates across all 79 counties
- **Historical Analysis** — per-county flood time series (2011–2025)
- **Model Performance** — metrics, feature importance, ablation results, and methodology transparency

### Reproducing All Results from Scratch

```bash
# Requires the dataset CSV in the project root
python train.py
```

This regenerates all model artefacts in `model/` and all 16 figures in `figures/`.

---

## Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
xgboost>=2.0.0
lightgbm>=4.0.0
plotly>=5.18.0
folium>=0.15.0
streamlit-folium>=0.18.0
```

---

## Limitations

1. **No river discharge** — upstream Nile/Sobat/Bahr el Ghazal forcing not captured; GloFAS integration is the highest-priority extension
2. **No Lake Victoria water levels** — 4–8 week advance warning signal currently absent
3. **Static wetland fraction** — cannot capture seasonal Sudd expansion (factor-of-3 variation between dry and wet years)
4. **No uncertainty quantification** — point probability only; conformal prediction intervals planned
5. **Single 24-month test window** — AUC estimated from 63 flood events; rolling evaluation as post-2025 data accumulates will provide more robust reliability estimates

---

## Citation

```bibtex
@article{monykuch2026ssd_flood,
  title   = {County-Level Flood Prediction in South Sudan Using Satellite-Derived
             Climate and Terrain Features: A Machine Learning Approach (2011--2025)},
  author  = {Monykuch, Chol Atem Giet},
  year    = {2026},
  month   = {April},
  note    = {African Leadership University. Dataset and code: GitHub/Kaggle.}
}
```

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT). Adaptation by humanitarian organisations, government agencies, and academic researchers is explicitly encouraged.

---

*South Sudan Flood Early Warning System · Logistic Regression · AUC = 0.9601 (95% CI: 0.935–0.982) · F1 = 0.7581 · Precision = 0.7705 · Recall = 0.7460 · Decision threshold = 0.935*
