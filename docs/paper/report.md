# Capstone Report — Growth / Recovery / Momentum Prediction

- **Author:** Aman Banik
- **Lane:** Freestyle B: Growth / Recovery / Momentum Prediction
- **Repo:** [Link to Repo](https://github.com/AmanBanik/Intern_at_fly)
- **Date:** August 2026

**Abstract:**
Which published articles should content teams update today to prevent organic traffic loss? We engineered a transparent heuristic baseline to identify pages suffering from high-value momentum decay and compared it against two Machine Learning models (Naive Random Forest and Advanced Histogram Gradient Boosting). Evaluated on a strictly grouped holdout split of 77,000 pages, the ML models failed to generalize across varying domain authorities (achieving at best 46% precision), while the business heuristic succeeded with a robust 75% precision. This research proves the dangers of ML generalization traps when dealing with power-law distributed web traffic. The final output is a highly actionable, ranked queue of 4,295 pages ready for immediate editorial triage.

## 1. Problem framing

Content teams face a severe resource allocation problem: out of tens of thousands of published articles, which ones should be updated today? Updating a perfectly stable page wastes editorial hours, while ignoring a decaying high-value page bleeds organic traffic and revenue. 

This work supports the decision of **editorial triage**. The unit of analysis is a single "page" (URL). By evaluating trailing 30-day Google Search Console data, we output a prioritized action queue. The cost of a wrong call is high—either wasted payroll on unnecessary rewrites, or lost visibility from inaction. A data-driven approach removes the guesswork, ensuring human editors focus their limited time strictly on pages where the potential retained value is mathematically the highest.

## 2. Data safety

This research was built on the **FlyRank ML Internship warehouse v20260703**, a robust dataset spanning 17 months of real, pseudonymized production search data. 

To prevent label leakage, we iterated entirely on a mid-panel month partition (March 2026). We deliberately excluded GA4 engagement metrics because we observed a 74% missingness rate across the portfolio; filling these gaps would have injected artificial category signals. We also safely discarded derived "trap" columns (`trend_direction`, `trend_pct`).

All `client_hash_id` and `content_hash_id` fields are strict pseudonyms used solely for grouped cross-validation, never as predictive features. Because we are dealing with severely right-skewed, power-law distributed web traffic (where a few URLs drive massive traffic), we avoided algorithms that assume normal Gaussian distributions or balanced linear boundaries (such as Linear Discriminant Analysis).

## 3. Baseline

The transparent business rule we built flagged pages that were visibly losing momentum. A page was flagged if it held significant historical visibility (>=100 impressions in the past 15 days) but its average Google rank slipped by >= 1.0 positions in the second half of that 15-day window.

This is a fair baseline comparison because it attacks the exact same goal as the ML models, using the same trailing metrics, but operates purely on simple, explainable logic without requiring complex parameter optimization.

## 4. Model / analysis

We attempted multiple model architectures to see if machine learning could identify complex decay patterns better than our baseline rule.

**Target Definition:** `dropped_traffic_next15d`. A page was flagged (1) if its total impressions in the subsequent 15 days dropped below 85% of its prior 15-day baseline.

**Methodology Progression:**
1. **Naive Random Forest:** We first fed the model raw, absolute features (`pos_first_half`, `pos_second_half`, `imp_past15`). 
2. **Advanced Histogram Gradient Boosting (HGB):** To combat severe overfitting observed in the Naive RF, we attempted advanced feature extraction. We engineered `relative_pos_diff`, which measures how far a page's rank deviates from its specific domain's median rank. We then fed these relative features into an advanced HGB model to force it to generalize across domains of wildly different sizes and authorities.

## 5. Evaluation

We utilized a strict `GroupKFold` cross-validation split (grouped by `client_hash_id`). This was critical to prevent **leakage**. Pages on the same domain share authority and seasonality; a random split would have leaked this context and artificially inflated our scores.

**Model vs Baseline:**
When evaluated on the strictly grouped holdout split, the transparent baseline significantly outperformed *both* machine learning models.

![Model Comparison](figures/model_comparison.svg)
*The baseline heuristic achieved a robust 75% Precision@100. The Naive RF collapsed to 42% because it memorized absolute positions. Even with advanced relative feature engineering, the HGB model only reached 46%, failing to beat the baseline's business logic.*

## 6. Interpretation

The results provided a profound lesson in model generalizability. 

The ML models failed because web traffic data is fundamentally unbalanced and domain-specific. The Naive ML model leaned heavily on *absolute* rank positions (`pos_second_half`). This is a classic ML trap: ranking #5 for a small client means something entirely different than ranking #5 for a massive enterprise client. 

Our attempt to fix this via feature extraction (`relative_pos_diff`) and upgrading to a Gradient Boosting model did improve performance slightly, but it still fell woefully short of the baseline. Why? Because the baseline rule (slipping rank + high visibility) perfectly captures the exact domain-specific business intent without having to learn complex, noisy thresholds from scratch. We confidently rejected the ML models in favor of the transparent heuristic.

## 7. Recommendation

Based on the winning baseline heuristic, we mapped the portfolio into a prioritized action playbook with distinct reason codes. These scores are strictly **decision-support** metrics; they identify pages that *look* worth reviewing first. 

![Triage Results](figures/triage_results.svg)  
*By applying the baseline logic, we successfully filtered a noisy 77,000-page dataset into a highly actionable queue of exactly 4,295 R1 targets.*

A FlyRank editor can use this tomorrow by starting at the top of the **[R1] High-Value Drift (4,295 pages)** queue for immediate editorial refreshes. 

## 8. Reproducibility

This research is fully transparent and reproducible. 
*   **Codebase:** The complete sequence of data extraction, baseline scoring, ML modeling, and queue generation is available in `work/scripts/run_all.py`.
*   **Notebooks:** The step-by-step logic and model progressions are documented sequentially in `work/notebooks/`.
*   **Environment:** standard python environment with `pandas`, `duckdb`, `scikit-learn`, and `matplotlib`. All random seeds are locked (`random_state=42`).

