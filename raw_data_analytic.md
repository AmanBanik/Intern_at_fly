# Exploratory Data Analysis: `content_refresh_anonymized.csv`

## 1. General Overview
- **Number of rows:** 30000
- **Number of columns:** 44

## 2. Column Analysis & Imputation Strategy

For tree-based models (like Random Forest, XGBoost, LightGBM), feature scaling is not strictly necessary. XGBoost and LightGBM handle missing values natively. However, if using standard scikit-learn trees, we need to impute.

| Column Name | Type | Missing (%) | Distribution Info | Imputation Strategy | Action / Feature Engineering |
|-------------|------|-------------|-------------------|---------------------|------------------------------|
| `content_id` | `object` | 0.00% | Unique: 30000, Top: content_0003a66b9fa5 | None needed | Drop (High Cardinality ID) |
| `client_id` | `object` | 0.00% | Unique: 32, Top: client_19581e27de | None needed | One-Hot Encode / Label Encode |
| `search_volume` | `float64` | 8.23% | Mean: 158.88, Median: 10.00, Skew: 26.02 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `competition` | `float64` | 8.23% | Mean: 0.15, Median: 0.00, Skew: 2.04 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `competition_level` | `object` | 8.70% | Unique: 3, Top: LOW | Mode or 'Missing' category | One-Hot Encode / Label Encode |
| `cpc` | `float64` | 8.23% | Mean: 0.49, Median: 0.00, Skew: 13.74 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `content_type` | `object` | 0.00% | Unique: 3, Top: keyword article | None needed | One-Hot Encode / Label Encode |
| `main_intent` | `object` | 7.91% | Unique: 4, Top: informational | Mode or 'Missing' category | One-Hot Encode / Label Encode |
| `word_count` | `float64` | 25.66% | Mean: 3107.76, Median: 2877.00, Skew: 0.94 | Mean (Symmetric) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `char_count` | `float64` | 25.66% | Mean: 20665.28, Median: 19116.00, Skew: 1.55 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `provider_used` | `object` | 71.46% | Unique: 2, Top: google | Mode or 'Missing' category | One-Hot Encode / Label Encode |
| `model_used` | `object` | 19.11% | Unique: 5, Top: gemini-3-flash-preview | Mode or 'Missing' category | One-Hot Encode / Label Encode |
| `impressions_90d` | `int64` | 0.00% | Mean: 5200.37, Median: 731.00, Skew: 11.38 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `clicks_90d` | `int64` | 0.00% | Mean: 16.10, Median: 1.00, Skew: 18.35 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `pageviews_90d` | `int64` | 0.00% | Mean: 49.94, Median: 8.00, Skew: 10.86 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `sessions_90d` | `int64` | 0.00% | Mean: 37.07, Median: 7.00, Skew: 12.13 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `users_90d` | `int64` | 0.00% | Mean: 35.94, Median: 7.00, Skew: 13.10 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `engaged_sessions_90d` | `int64` | 0.00% | Mean: 0.99, Median: 0.00, Skew: 24.54 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `ai_sessions_90d` | `int64` | 0.00% | Mean: 0.20, Median: 0.00, Skew: 19.14 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `scroll_events_90d` | `int64` | 0.00% | Mean: 4.03, Median: 1.00, Skew: 65.29 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `days_with_impressions` | `int64` | 0.00% | Mean: 61.45, Median: 81.00, Skew: -0.82 | Mean (Symmetric) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `days_with_sessions` | `int64` | 0.00% | Mean: 13.10, Median: 6.00, Skew: 2.12 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `impressions_last_30d` | `int64` | 0.00% | Mean: 1429.06, Median: 139.00, Skew: 14.62 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `clicks_last_30d` | `int64` | 0.00% | Mean: 4.93, Median: 0.00, Skew: 17.12 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `sessions_last_30d` | `int64` | 0.00% | Mean: 14.11, Median: 3.00, Skew: 9.09 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `impressions_prev_30d` | `int64` | 0.00% | Mean: 1783.08, Median: 210.00, Skew: 11.73 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `clicks_prev_30d` | `int64` | 0.00% | Mean: 5.44, Median: 0.00, Skew: 21.31 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `sessions_prev_30d` | `int64` | 0.00% | Mean: 10.28, Median: 2.00, Skew: 40.87 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) / Combine highly correlated traffic features (PCA or manual sum) |
| `content_age_days` | `int64` | 0.00% | Mean: 256.17, Median: 236.00, Skew: 0.49 | Mean (Symmetric) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `age_tier` | `object` | 0.00% | Unique: 4, Top: 91-180 | None needed | One-Hot Encode / Label Encode |
| `age_tier_order` | `int64` | 0.00% | Mean: 4.79, Median: 5.00, Skew: 0.20 | Mean (Symmetric) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `days_since_last_update` | `int64` | 0.00% | Mean: 46.10, Median: 20.00, Skew: 1.16 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `freshness_tier` | `object` | 0.00% | Unique: 4, Top: 0-30 | None needed | One-Hot Encode / Label Encode |
| `word_count_tier` | `object` | 25.66% | Unique: 4, Top: 2000-3500 | Mode or 'Missing' category | One-Hot Encode / Label Encode |
| `char_count_tier` | `object` | 25.66% | Unique: 4, Top: 15000-25000 | Mode or 'Missing' category | One-Hot Encode / Label Encode |
| `ctr` | `float64` | 0.00% | Mean: 0.51, Median: 0.07, Skew: 17.44 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `avg_position` | `float64` | 0.00% | Mean: 16.34, Median: 10.80, Skew: 1.98 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `engagement_rate` | `float64` | 0.00% | Mean: 2.53, Median: 0.00, Skew: 7.22 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `scroll_rate` | `float64` | 0.42% | Mean: 18.21, Median: 5.00, Skew: 2.50 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `ai_traffic_pct` | `float64` | 0.00% | Mean: 0.77, Median: 0.00, Skew: 18.43 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |
| `impression_tier` | `object` | 0.00% | Unique: 4, Top: low | None needed | One-Hot Encode / Label Encode |
| `position_tier` | `object` | 0.00% | Unique: 5, Top: page_1 | None needed | One-Hot Encode / Label Encode |
| `trend_direction` | `object` | 0.00% | Unique: 5, Top: down | None needed | One-Hot Encode / Label Encode |
| `trend_pct` | `float64` | 11.29% | Mean: -4.79, Median: -33.50, Skew: 56.76 | Median (Highly Skewed) | Keep. Consider log-transform for skewed vars (optional for trees) |

## 3. Feature Engineering & Selection for Tree Models

### Columns to Drop
- **Identifiers:** `content_id`, `client_id` provide no generalizable information and have very high cardinality. They will lead to severe overfitting if used in trees.
- **Redundant/Categorical Tier Columns:** If numerical columns (like `content_age_days`) already exist, categorical tiers (like `age_tier`, `age_tier_order`) are redundant for trees. Trees are naturally capable of finding the optimal splits in continuous data, making manual binning unnecessary unless it captures non-linear domain knowledge.

### Missing Value Strategy (Summary)
- **Numerical Features:** Since tree models split data based on inequalities, extreme values don't heavily impact them. For heavily skewed variables, impute with the **Median**. For roughly symmetric ones, impute with the **Mean**.
- **Categorical Features:** Tree models often perform well when missing categorical data is treated as its own distinct category (e.g., fill with `'Missing'`). This allows the tree to capture any signal associated with the fact that the data is missing. Otherwise, impute with the **Mode**.

### Feature Creation Ideas (Culmination)
- **Engagement Metrics:** Create aggregate features like `clicks_per_impression` (CTR - might already exist, verify), `sessions_per_click`, or `engaged_sessions_per_total_session`.
- **Temporal Dynamics:** Compare recent 30 days to previous 30 days (e.g., `impressions_last_30d` / `impressions_prev_30d`). This captures the **momentum** of the content, which might be a strong predictor.
- **Content Length Ratios:** `char_count` / `word_count` gives average word length, which could proxy for content complexity or reading level.
- **Provider/Model Features:** If `provider_used` or `model_used` have many missing values, binarizing into `used_ai` (1/0) might provide good information gain.
