import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('../../outputs/model_comparison.json', 'r') as f:
    data = json.load(f)

k_values = [str(d['K']) for d in data]
baseline_pk = [d['Baseline P@K'] for d in data]
rf_pk = [d['Naive RF P@K'] for d in data]
tuned_pk = [d.get('Tuned RF P@K', d.get('Advanced HGB P@K', 0)) for d in data]
base_rate = data[0]['Base Rate']

x = np.arange(len(k_values))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, baseline_pk, width, label='Transparent Baseline', color='#2196F3')
rects2 = ax.bar(x, rf_pk, width, label='Naive RF', color='#FF9800')
rects3 = ax.bar(x + width, tuned_pk, width, label='Tuned RF (Winner)', color='#4CAF50')

# Add base rate line
ax.axhline(y=base_rate, color='#E53935', linestyle='--', label=f'Base Rate ({base_rate:.2f})')

ax.set_ylabel('Precision@K')
ax.set_title('GroupKFold Honest Validation: Baseline vs ML Models')
ax.set_xticks(x)
ax.set_xticklabels([f"Top {k}" for k in k_values])
ax.legend()
ax.set_ylim(0, 1.0)

fig.tight_layout()
os.makedirs('..', exist_ok=True)
plt.savefig('../model_comparison.svg')
plt.savefig('../model_comparison.png', dpi=300)
print("Generated model_comparison charts.")
