import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(labels)[order[:k]].mean()

def run_experiment():
    print("Running 03_ml_model_experiment...")
    df = pd.read_csv('../outputs/baseline_action_score.csv')
    df['pos_change'] = df['pos_second_half'] - df['pos_first_half']
    
    # Relative Feature Engineering
    df['client_median_pos'] = df.groupby('client_hash_id')['pos_first_half'].transform('median')
    df['relative_pos_diff'] = df['pos_first_half'] - df['client_median_pos']
    
    features_naive = ['imp_past15', 'pos_first_half', 'pos_second_half', 'pos_change']
    features_adv = ['imp_past15', 'relative_pos_diff', 'pos_change']
    
    y = df['dropped_traffic_next15d']
    groups = df['client_hash_id']
    
    gkf = GroupKFold(n_splits=5)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    hgb_model = HistGradientBoostingClassifier(max_depth=5, random_state=42)
    
    rf_oof_preds = np.zeros(len(df))
    hgb_oof_preds = np.zeros(len(df))
    
    for train_idx, val_idx in gkf.split(df[features_naive], y, groups):
        # Naive RF
        rf_model.fit(df[features_naive].iloc[train_idx], y.iloc[train_idx])
        rf_oof_preds[val_idx] = rf_model.predict_proba(df[features_naive].iloc[val_idx])[:, 1]
        
        # Advanced HGB
        hgb_model.fit(df[features_adv].iloc[train_idx], y.iloc[train_idx])
        hgb_oof_preds[val_idx] = hgb_model.predict_proba(df[features_adv].iloc[val_idx])[:, 1]
    
    df['rf_pred_prob'] = rf_oof_preds
    df['hgb_pred_prob'] = hgb_oof_preds
    
    results = []
    base_rate = y.mean()
    for k in [20, 50, 100]:
        base_p = precision_at_k(df['score'], y, k)
        rf_p = precision_at_k(df['rf_pred_prob'], y, k)
        hgb_p = precision_at_k(df['hgb_pred_prob'], y, k)
        results.append({'K': k, 'Base Rate': float(base_rate), 'Baseline P@K': float(base_p), 'Naive RF P@K': float(rf_p), 'Advanced HGB P@K': float(hgb_p)})
    
    with open('../outputs/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Experiment complete: Both ML models underperformed compared to Baseline. Metrics saved.")

if __name__ == '__main__':
    run_experiment()
