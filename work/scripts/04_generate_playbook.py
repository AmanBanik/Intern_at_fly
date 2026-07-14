import os
import json
import pandas as pd

def assign_priority(row):
    if row['imp_past15'] > 1000 and row['score'] > 0:
        return 'R1 - High-Value Drift'
    elif row['score'] > 0:
        return 'R2 - Stale Warning'
    elif row['imp_past15'] > 500 and row['score'] <= 0:
        return 'S1 - Stable/Safe'
    else:
        return 'S2 - Low-Value Ghost'

def run_playbook():
    print("Running 04_generate_playbook...")
    df = pd.read_csv('../outputs/baseline_action_score.csv')
    df['action_priority'] = df.apply(assign_priority, axis=1)
    
    queue_df = df[df['action_priority'].str.startswith('R')].sort_values(by='imp_past15', ascending=False)
    queue_df.to_csv('../outputs/action_queue.csv', index=False)
    
    triage_stats = df['action_priority'].value_counts().to_dict()
    with open('../outputs/triage_stats.json', 'w') as f:
        json.dump(triage_stats, f, indent=2)
        
    print("Playbook complete: action_queue.csv and triage_stats.json generated.")

if __name__ == '__main__':
    run_playbook()
