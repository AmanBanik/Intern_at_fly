import os
import json
import matplotlib.pyplot as plt

with open('../../outputs/triage_stats.json', 'r') as f:
    data = json.load(f)

# Data is {"S1 - Stable/Safe": 26035, "S2 - Low-Value Ghost": 25801, "R2 - Stale Warning": 20892, "R1 - High-Value Drift": 4295}
order = ['R1 - High-Value Drift', 'R2 - Stale Warning', 'S1 - Stable/Safe', 'S2 - Low-Value Ghost']
ordered_sizes = [data.get(k, 0) for k in order]
colors = ['#E53935', '#FFB74D', '#8BC34A', '#E0E0E0']

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    ordered_sizes, labels=order, colors=colors, autopct='%1.1f%%',
    startangle=140, pctdistance=0.85, textprops=dict(color="black", weight="bold")
)

# Draw a circle at the center of pie to make it a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(centre_circle)

# Adjust label text
for text in texts:
    text.set_color('black')

ax.axis('equal')  
plt.title('Content Action Playbook: 77k Pages Triaged', y=1.05, weight='bold')

fig.tight_layout()
os.makedirs('..', exist_ok=True)
plt.savefig('../triage_results.svg')
plt.savefig('../triage_results.png', dpi=300)
print("Generated triage_results charts.")
