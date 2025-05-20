import matplotlib.pyplot as plt
import pandas as pd

# Load all sheets from the Excel file
xls = pd.ExcelFile('old/multitask_elastic_net_results_6.xlsx')
results = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}

# Feature names
strip_columns = [
    'ZeroCouponBond', 'DividendValue', 'DividendREIT',
    'DividendInfra', 'DividendMarket', 'DividendSmall',
    'DividendGrowth', 'DividendNR', 'GainValue',
    'GainREIT', 'GainInfra', 'GainMarket',
    'GainSmall', 'GainGrowth', 'GainNR'
]

# Plotting
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14))  # Vertical layout
quarters = list(range(1, 65))  # 64 time points

# Feature style setup
strip_pairs = [
    ('DividendValue', 'GainValue'),
    ('DividendREIT', 'GainREIT'),
    ('DividendInfra', 'GainInfra'),
    ('DividendMarket', 'GainMarket'),
    ('DividendSmall', 'GainSmall'),
    ('DividendGrowth', 'GainGrowth'),
    ('DividendNR', 'GainNR')
]

# Select 7 visually distinct colors
selected_colors = [
    plt.cm.tab20.colors[0],   # blue
    plt.cm.tab20.colors[2],   # green
    plt.cm.tab20.colors[4],   # red
    plt.cm.tab20.colors[6],   # purple
    plt.cm.tab20.colors[8],   # brown
    plt.cm.tab20.colors[10],  # pink
    plt.cm.tab20.colors[12]   # gray
]

# Build style dict
styles = {
    'ZeroCouponBond': {'color': 'black', 'linestyle': '-'}
}
for (dividend, gain), color in zip(strip_pairs, selected_colors):
    styles[dividend] = {'color': color, 'linestyle': ':'}
    styles[gain] = {'color': color, 'linestyle': '-'}

# Plotting loop
for ax, (asset_class, df) in zip(axes, results.items()):
    B = df.to_numpy()

    for i, name in enumerate(strip_columns):
        if name not in styles:
            raise ValueError(f"Style for feature '{name}' is not defined.")

        ax.plot(
            quarters,
            B[i],
            label=name,
            color=styles[name]['color'],
            linestyle=styles[name]['linestyle'],
            linewidth=2.5  # slightly thicker for visibility
        )

    ax.set_title(asset_class)
    ax.set_xlabel("Quarter")
    ax.grid(True)


axes[0].set_ylabel("Coefficient value")
axes[1].set_ylabel("Coefficient value")
axes[2].set_ylabel("Coefficient value")

# Unified legend outside the plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Make space for the legend
plt.show()
