# Advanced ML Tuning Experiment

## Why does this folder exist?
After proving in the main capstone report that a transparent business heuristic (75% Precision@100) defeated our initial Random Forest and Histogram Gradient Boosting models (42-46% Precision), we wanted to run one final, exhaustive test. 

We asked: *Did the ML models fail because of a flawed algorithm, or because the fundamental nature of the dataset (power-law distributed, domain-specific search traffic) is inherently hostile to generalized ML?*

## What we tested in this side-quest:
We wrote `run_experiment.py` to throw every advanced optimization technique at the Random Forest:
1. **Multicollinearity Checks:** Calculated the Correlation Matrix and Variance Inflation Factor (VIF).
2. **Recursive Feature Elimination (RFE):** Mathematically stripped the feature set down to the absolute top 5 most important features.
3. **Hyperparameter Tuning:** Ran a massive `RandomizedSearchCV` (testing `max_depth`, `min_samples_split`, `min_samples_leaf`, and `n_estimators`) strictly evaluated on a 3-Fold `GroupKFold` split to force the model to generalize across clients.
4. **Learning Curve:** Generated an `n_estimators` curve to track performance gains against tree volume.

## The Results (5-Fold GroupKFold)
*   **Transparent Baseline:** 75.0% Precision@100
*   **Naive Random Forest:** 42.0% Precision@100
*   **Tuned & RFE-Selected RF:** 38.0% Precision@100

## Conclusion
The hyperparameter-tuned model actually performed **worse** than our naive model. 

Why? Because the grid search correctly identified that the Naive RF was heavily overfitting to absolute rank positions. To fix that, the grid search aggressively pruned the trees (e.g., `min_samples_leaf=20`). However, because web traffic is heavily skewed by a power-law distribution, "generalizing" the trees just turned them into mush. They lost their ability to memorize absolute ranks, but failed entirely to learn the domain-specific nuances.

This experiment mathematically proves the thesis of our capstone paper: forcing Machine Learning onto this specific problem is a generalization trap. The transparent business heuristic (slipping rank + high visibility) flawlessy captures the domain-specific logic, remaining the absolute best tool for the job.
