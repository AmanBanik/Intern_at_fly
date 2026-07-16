import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Setup Logging
log_file = 'experiment.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(labels)[order[:k]].mean()

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vifs = []
    
    for i, col in enumerate(df.columns):
        y = df[col]
        X = df.drop(columns=[col])
        r2 = LinearRegression().fit(X, y).score(X, y)
        if r2 < 1.0:
            vifs.append(1.0 / (1.0 - r2))
        else:
            vifs.append(np.inf)
            
    vif_data["VIF"] = vifs
    return vif_data

def run_experiment():
    logging.info("Starting Advanced Random Forest Experiment...")
    
    # 1. Load Data
    df = pd.read_csv('../outputs/baseline_action_score.csv')
    
    # Feature Engineering
    df['pos_change'] = df['pos_second_half'] - df['pos_first_half']
    df['client_median_pos'] = df.groupby('client_hash_id')['pos_first_half'].transform('median')
    df['relative_pos_diff'] = df['pos_first_half'] - df['client_median_pos']
    df['imp_rank'] = df.groupby('client_hash_id')['imp_past15'].rank(pct=True)
    df['rank_ratio'] = df['pos_second_half'] / (df['pos_first_half'] + 1e-5)
    
    features = ['imp_past15', 'pos_first_half', 'pos_second_half', 'pos_change', 
                'client_median_pos', 'relative_pos_diff', 'imp_rank', 'rank_ratio']
    
    X = df[features].copy()
    y = df['dropped_traffic_next15d']
    groups = df['client_hash_id']
    
    # Clean X for model
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 2. Correlation Matrix
    logging.info("Calculating Correlation Matrix...")
    corr = X.corr()
    corr.to_csv('corr_matrix.csv')
    
    # 3. VIF
    logging.info("Calculating VIF...")
    vif_data = calculate_vif(X)
    vif_data.to_csv('vif.csv', index=False)
    logging.info(f"VIF summary:\n{vif_data}")
    
    # 4. RFE
    logging.info("Running RFE...")
    base_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=3)
    rfe = RFE(estimator=base_rf, n_features_to_select=5, step=1)
    rfe.fit(X, y)
    rfe_res = pd.DataFrame({'Feature': features, 'Ranking': rfe.ranking_, 'Selected': rfe.support_})
    rfe_res = rfe_res.sort_values('Ranking')
    rfe_res.to_csv('rfe_ranking.csv', index=False)
    logging.info(f"RFE summary:\n{rfe_res}")
    
    selected_features = rfe_res[rfe_res['Selected']]['Feature'].tolist()
    logging.info(f"Selected Features for tuning: {selected_features}")
    X_selected = X[selected_features]
    
    # 5. Tuning
    logging.info("Running Hyperparameter Tuning (RandomizedSearchCV)...")
    gkf = GroupKFold(n_splits=3)
    
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 8, 12, None],
        'min_samples_split': [2, 10, 50, 100],
        'min_samples_leaf': [1, 5, 20, 50]
    }
    
    rf = RandomForestClassifier(random_state=42)
    # limit n_jobs to prevent freezes
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=15, 
                                       cv=gkf, scoring='roc_auc', n_jobs=3, random_state=42, verbose=1)
    
    random_search.fit(X_selected, y, groups=groups)
    best_rf = random_search.best_estimator_
    logging.info(f"Best Hyperparameters: {random_search.best_params_}")
    
    # 6. n_estimators curve
    logging.info("Generating n_estimators curve...")
    n_estimators_range = [10, 50, 100, 200, 300]
    precisions = []
    
    train_idx, val_idx = next(gkf.split(X_selected, y, groups=groups))
    X_train, y_train = X_selected.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X_selected.iloc[val_idx], y.iloc[val_idx]
    
    for n in n_estimators_range:
        model = RandomForestClassifier(n_estimators=n, 
                                       max_depth=random_search.best_params_['max_depth'], 
                                       min_samples_split=random_search.best_params_['min_samples_split'],
                                       min_samples_leaf=random_search.best_params_['min_samples_leaf'],
                                       random_state=42, n_jobs=3)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        p100 = precision_at_k(preds, y_val, 100)
        precisions.append(p100)
        logging.info(f"n_estimators={n} -> Precision@100 = {p100:.4f}")
        
    plt.figure()
    plt.plot(n_estimators_range, precisions, marker='o')
    plt.title("n_estimators vs Precision@100")
    plt.xlabel("n_estimators")
    plt.ylabel("Precision@100")
    plt.grid(True)
    plt.savefig('n_estimators_curve.png')
    
    # 7. Final Inference
    logging.info("Running final inference with Best Model on 5-Fold GroupKFold...")
    gkf_eval = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df))
    
    for train_idx, val_idx in gkf_eval.split(X_selected, y, groups):
        best_rf.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
        oof_preds[val_idx] = best_rf.predict_proba(X_selected.iloc[val_idx])[:, 1]
        
    df['tuned_rf_prob'] = oof_preds
    base_p100 = precision_at_k(df['score'], y, 100)
    tuned_p100 = precision_at_k(df['tuned_rf_prob'], y, 100)
    
    logging.info(f"--- FINAL RESULTS ---")
    logging.info(f"Baseline Precision@100: {base_p100:.4f}")
    logging.info(f"Tuned RF Precision@100: {tuned_p100:.4f}")
    
    if tuned_p100 > base_p100:
        logging.info("WIN! The Tuned Random Forest finally beat the baseline!")
    else:
        logging.info("LOSS! The Baseline still wins. The leakage trap is incredibly strong.")

if __name__ == '__main__':
    run_experiment()
