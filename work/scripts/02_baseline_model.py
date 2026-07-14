import os
import pandas as pd
import numpy as np

def run_baseline():
    print("Running 02_baseline_model...")
    df = pd.read_csv('../outputs/raw_features.csv')
    
    df['pos_first_half'] = df['pos_first_half'].fillna(0)
    df['pos_second_half'] = df['pos_second_half'].fillna(0)
    
    df['stale'] = (df['pos_second_half'] - df['pos_first_half'] >= 1.0).astype(int)
    df['visible'] = (df['imp_past15'] >= 100).astype(int)
    df['score'] = df['stale'] * df['visible'] * df['imp_past15']
    
    ranked_queue = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    ranked_queue.to_csv('../outputs/baseline_action_score.csv', index=False)
    print(f"Baseline complete: saved baseline_action_score.csv with {len(ranked_queue)} rows.")

if __name__ == '__main__':
    run_baseline()
