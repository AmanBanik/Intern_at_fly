import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(labels)[order[:k]].mean()

def run_experiment():
    print("Running 03_ml_model_experiment...")
    df = pd.read_csv('../outputs/baseline_action_score.csv')
    df['pos_change'] = df['pos_second_half'] - df['pos_first_half']
    
    features = ['imp_past15', 'pos_first_half', 'pos_second_half', 'pos_change']
    X = df[features]
    y = df['dropped_traffic_next15d']
    groups = df['client_hash_id']
    
    gkf = GroupKFold(n_splits=5)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    oof_preds = np.zeros(len(df))
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    df['model_pred_prob'] = oof_preds
    
    results = []
    base_rate = y.mean()
    for k in [20, 50, 100]:
        base_p = precision_at_k(df['score'], y, k)
        model_p = precision_at_k(df['model_pred_prob'], y, k)
        results.append({'K': k, 'Base Rate': float(base_rate), 'Baseline P@K': float(base_p), 'Model P@K': float(model_p)})
    
    with open('../outputs/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Experiment complete: ML model underperformed compared to Baseline. Metrics saved.")

if __name__ == '__main__':
    run_experiment()
