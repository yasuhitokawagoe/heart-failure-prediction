# VT/VF Ensemble Weather Model SHAP Analysis Report

**Analysis Date:** 2025-07-31 10:34:12

**Model Type:** Ensemble (XGBoost, LightGBM, CatBoost)

**Target Variable:** VT/VF High Risk Days (75th percentile)

**Data Shape:** (3562, 65)

**Positive Class Ratio:** 0.326

## Ensemble Models

- **CAT**: AUC = 0.4908

## Feature Importance Summary

| Category | Count |
|----------|-------|
| Total Features | 65 |
| Weather Features | 58 |
| Extreme Weather Features | 1 |
| Time Series Features | 19 |
| Interaction Features | 42 |

## Top 20 Most Important Features (Ensemble)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | min_temp_3day_ago | 0.7870 |
| 2 | avg_humidity_vtvf_ma_14d | 0.3208 |
| 3 | vapor_pressure_vtvf_ma_3d_change_rate | 0.3192 |
| 4 | sunshine_hours_6day_ago | 0.2709 |
| 5 | avg_wind_6day_ago | 0.2466 |
| 6 | avg_humidity_vtvf_std_3d | 0.2335 |
| 7 | vapor_pressure_vtvf_std_7d | 0.2289 |
| 8 | vapor_pressure_vtvf_std_14d | 0.2251 |
| 9 | avg_humidity_vtvf_std_7d | 0.2208 |
| 10 | avg_humidity_2day_ago | 0.2130 |
| 11 | avg_wind_5day_ago | 0.1992 |
| 12 | avg_temp_vtvf_std_3d | 0.1885 |
| 13 | avg_wind_2day_ago | 0.1765 |
| 14 | avg_temp_vtvf_std_7d | 0.1746 |
| 15 | avg_wind_7day_ago | 0.1741 |
| 16 | avg_wind_4day_ago | 0.1685 |
| 17 | avg_humidity_vtvf_ma_3d | 0.1660 |
| 18 | avg_temp_6day_ago | 0.1646 |
| 19 | day_sin | 0.1636 |
| 20 | avg_wind_3day_ago | 0.1632 |
