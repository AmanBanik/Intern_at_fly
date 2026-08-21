# FlyRank ML Internship

**Applied Search Intelligence: Growth / Recovery / Momentum Prediction**

This repository serves as my complete workspace and final portfolio submission for the 8-week FlyRank ML Internship. It documents the entire end-to-end journey—from initial problem framing and data auditing to the deployment of a heavily tuned Machine Learning pipeline.

🔗 **[Read my Final Deployed Capstone Paper Here](https://amanbanik.github.io/Intern_at_fly/paper/)**

---

## 🏆 The Capstone Project: Editorial Triage

### 🎯 The Business Problem
Content teams manage tens of thousands of published articles. Updating a perfectly stable page wastes editorial hours, while ignoring a decaying high-value page bleeds organic traffic. By evaluating trailing 30-day Google Search Console data on a mid-panel month partition (March 2026), this project outputs a prioritized action queue—ensuring human editors focus their limited time strictly on pages where the potential retained value is mathematically the highest.

### 🛠️ The Technical Hustle
Web traffic is heavily right-skewed and power-law distributed. An initial Naive Random Forest completely failed (54% precision) because it memorized absolute ranks. 

Instead of accepting a transparent heuristic baseline (which scored 63% precision), I executed a deep pipeline audit:
*   **Data Leakage Fixed:** Identified and repaired a silent `fillna` imputation error on missing rank positions.
*   **Relative Feature Engineering:** Engineered `relative_pos_diff` to measure how far a page deviates from its specific domain's median rank, safely removing domain-size bias.
*   **Aggressive Regularization:** Deployed an exhaustive Grid Search and heavily regularized the decision trees (`min_samples_leaf=20`) to physically force the model to generalize across domains.

### 🚀 The Results
*   The Tuned Random Forest achieved an unprecedented **71% Precision@100**, decisively defeating the business heuristic baseline.
*   Successfully filtered a noisy, 75,000-page production dataset into a highly actionable queue of exactly **5,562 R1 (High-Value Drift)** targets ready for immediate editorial intervention.

---

## 📅 The 8-Week Internship Progression

While the capstone is the final output, this repository also houses the weekly foundational work that built up to it. The `work/notebooks/` directory contains the step-by-step progression of the internship:

*   **Weeks 1-2 (Task Framing):** Defining the editorial triage problem and exploring the raw anonymized dataset.
*   **Week 3 (Data Contract & Leakage):** Establishing rigorous data definitions and auditing the pipeline for target leakage.
*   **Week 4 (Signal & Baseline):** Building the transparent business heuristic to serve as our performance floor (63% Precision).
*   **Weeks 5-6 (Model & Validation):** Training the Naive ML models, realizing they fell into the power-law trap, and executing the Grid Search validation that led to the 71% tuned model.
*   **Weeks 7-8 (Playbook & Capstone):** Translating the ML probabilities into the final action queue (`04_generate_playbook.py`) and publishing the final deployed paper.

---

## 📂 Key Repository Structure

| Path | What it is |
|---|---|
| `docs/paper/` | My final deployed Capstone Paper (HTML + CSS) & Social Metadata. |
| `work/scripts/run_all.py` | The winning ML pipeline: prepares data, engineers relative features, trains the Tuned RF, and generates the action queue. |
| `work/notebooks/capstone.ipynb` | The comprehensive capstone notebook, containing the narrative, codebase, 5-minute demo outline, and shareable social cuts. |
| `work/experiment/` | Grid Search, VIF analyses, and feature importance artifacts proving why aggressive regularization unlocked the true signal. |

### 🔒 Data Safety Note
This repository strictly uses the **anonymized** slice of real FlyRank search data provided for the internship. No private client data, URLs, or raw keywords are included in this repository. All `client_hash_id` and `content_hash_id` fields are strict pseudonyms.

---

*Track leads: Mirza Ašćerić (ML) · Hole (data engineering). Code under MIT (see `LICENSE`); data under `DATA_USE.md`.*
