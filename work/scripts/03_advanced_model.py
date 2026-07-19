import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_advanced_model():
    logging.info("Loading baseline features...")
    df = pd.read_csv('../outputs/baseline_action_score.csv')
    
    logging.info("Engineering relative features...")
    df['pos_change'] = df['pos_second_half'] - df['pos_first_half']
    df['client_median_pos'] = df.groupby('client_hash_id')['pos_first_half'].transform('median')
    df['relative_pos_diff'] = df['pos_first_half'] - df['client_median_pos']
    
    features_adv = ['imp_past15', 'pos_first_half', 'pos_second_half', 'client_median_pos', 'relative_pos_diff']
    
    X_adv = df[features_adv]
    y = df['dropped_traffic_next15d']
    groups = df['client_hash_id']
    
    # Fill Nans (should be none due to dropna in 02, but safe)
    X_adv = X_adv.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    logging.info("Training Tuned Random Forest with GroupKFold...")
    gkf = GroupKFold(n_splits=5)
    tuned_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=100, min_samples_leaf=20, random_state=42, n_jobs=-1)
    
    tuned_oof_preds = np.zeros(len(df))
    
    for train_idx, val_idx in gkf.split(X_adv, y, groups):
        tuned_model.fit(X_adv.iloc[train_idx], y.iloc[train_idx])
        tuned_oof_preds[val_idx] = tuned_model.predict_proba(X_adv.iloc[val_idx])[:, 1]
        
    df['tuned_rf_prob'] = tuned_oof_preds
    
    # Save predictions
    output_path = '../outputs/advanced_ml_scores.csv'
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully exported ML scores to {output_path}")

if __name__ == '__main__':
    run_advanced_model()
