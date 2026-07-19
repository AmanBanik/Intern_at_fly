# Advanced ML Tuning Experiment

## Why does this folder exist?
After observing that our initial Naive Random Forest completely failed (54% Precision@100) because it memorized absolute rank positions and collapsed on unseen clients, we launched an exhaustive experimental side-quest to find a true ML solution.

We asked: *Can we rescue the Machine Learning approach by physically forcing the decision trees to stop memorizing absolute ranks, and forcing them to generalize?*

## What we tested in this side-quest:
We wrote `run_experiment.py` to throw every advanced optimization technique at the Random Forest, paired with our newly engineered relative features (`relative_pos_diff` and `client_median_pos`):
1. **Multicollinearity Checks:** Calculated the Correlation Matrix and Variance Inflation Factor (VIF).
2. **Recursive Feature Elimination (RFE):** Mathematically stripped the feature set down to the absolute top 5 most important features.
3. **Hyperparameter Tuning:** Ran a massive `RandomizedSearchCV` (testing `max_depth`, `min_samples_split`, `min_samples_leaf`, and `n_estimators`) strictly evaluated on a `GroupKFold` split to force the model to generalize across clients.
4. **Learning Curve:** Generated an `n_estimators` curve (`n_estimators_curve.png`) to track performance gains against tree volume.

## The Results
*   **Transparent Baseline:** 63.0% Precision@100
*   **Naive Random Forest:** 54.0% Precision@100
*   **Tuned & RFE-Selected RF:** 71.0% Precision@100

## Conclusion
The hyperparameter-tuned model triumphed! 

Why? Because the grid search correctly identified that the Naive RF was heavily overfitting to absolute rank positions. To fix that, the grid search aggressively pruned the trees (deploying a massive `min_samples_leaf=20` and `min_samples_split=100`).

This aggressive regularization physically prevented the trees from creating highly specific split boundaries (e.g., "if position is exactly 4"). Forced to generalize, and armed with our engineered relative features, the Tuned Random Forest finally extracted the true underlying signal of momentum decay across all domains.

This experiment mathematically validates our final deployment decision: we successfully transformed a generalization trap into a robust, 71% accurate ML engine.
