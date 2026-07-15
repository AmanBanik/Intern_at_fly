import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('../../outputs/model_comparison.json', 'r') as f:
    data = json.load(f)

k_values = [str(d['K']) for d in data]
baseline_pk = [d['Baseline P@K'] for d in data]
model_pk = [d['Model P@K'] for d in data]
base_rate = data[0]['Base Rate']

x = np.arange(len(k_values))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, baseline_pk, width, label='Transparent Baseline (Generalizes)', color='#2196F3')
rects2 = ax.bar(x + width/2, model_pk, width, label='Random Forest (Overfits)', color='#FF9800')

# Add base rate line
ax.axhline(y=base_rate, color='#E53935', linestyle='--', label=f'Base Rate ({base_rate:.2f})')

ax.set_ylabel('Precision@K')
ax.set_title('GroupKFold Honest Validation: Baseline vs ML Model')
ax.set_xticks(x)
ax.set_xticklabels([f"Top {k}" for k in k_values])
ax.legend()
ax.set_ylim(0, 1.0)

fig.tight_layout()
os.makedirs('..', exist_ok=True)
plt.savefig('../model_comparison.svg')
plt.savefig('../model_comparison.png', dpi=300)
print("Generated model_comparison charts.")
