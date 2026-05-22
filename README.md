# Metaverse Cyberfraud Shielding | Econometric Risk Detection | Python

## Project Overview
Built an econometric pipeline to detect and quantify transactional fraud risk across 78,600 metaverse transactions. OLS regression was used for risk score modelling; PCA for multicollinearity correction; and HC3 robust standard errors for valid inference under heteroskedasticity.

## Tools & Technologies
- **Python** — end-to-end pipeline scripting
- **statsmodels** — OLS regression, White's heteroskedasticity test, Durbin-Watson test
- **scikit-learn** — PCA, StandardScaler, OneHotEncoder, train-test split
- **scipy.stats** — Shapiro-Wilk and Kolmogorov-Smirnov normality tests
- **seaborn / matplotlib** — EDA visualisations and Q-Q plot

## Dataset
A single-table dataset of 78,600 metaverse transactions with 14 variables including `amount`, `transaction_type`, `age_group`, `risk_score`, `anomaly`, and behavioural features like `login_frequency` and `session_duration`. No missing values.

## Key Findings
- **R² = 0.626** after PCA-driven multicollinearity correction (9 principal components retained explaining ≥95% variance)
- **Multicollinearity resolved**: VIF dropped to 1.0 across all components after PCA; pre-PCA VIFs were infinite for `purchase_pattern` and `age_group` dummies
- **Heteroskedasticity confirmed** via White's Test (LM statistic = 55,003.50, p = 0.000) and corrected using HC3 robust standard errors
- **No autocorrelation detected**: Durbin-Watson statistic = 1.9896
- **Veteran cohorts carry ~2× higher risk scores than young users** — suggesting fraud escalation is driven by high-frequency, high-volume accounts rather than inexperience
- Transaction type is a strong predictor: phishing transactions carry significantly higher risk scores than transfers, purchases, and sales

## Methodology Summary
1. **EDA** — distribution analysis, correlation heatmap, risk score by age group boxplot
2. **Preprocessing** — one-hot encoding (`drop='first'`), StandardScaler, log-transform on `risk_score`
3. **Initial OLS** — R² = 0.952 flagged as inflated due to near-singular design matrix
4. **PCA** — reduced to 9 uncorrelated components; re-ran OLS for honest R² = 0.626
5. **Diagnostics** — White's test, Durbin-Watson, Shapiro-Wilk, KS test, Q-Q plot
6. **Robust Inference** — HC3 standard errors applied; significance of key PCs confirmed

## Limitations & Extensions
- Non-normal residuals suggest **quantile regression** may better capture tail-risk fraud events
- PCA sacrifices interpretability — **Ridge/LASSO** on original features is a natural extension
- The `anomaly` column was unused here; a **classification pipeline** on fraud labels is a logical next step
