import matplotlib.pyplot as plt
import pandas as pd

sheets_to_load = 3
group_size = 4  # 4 quarters per year

# Load only the first two sheets from the Excel file
xls = pd.ExcelFile('multitask_elastic_net_results.xlsx')
sheet_names = xls.sheet_names[:sheets_to_load]
results = {sheet: xls.parse(sheet) for sheet in sheet_names}

# Feature names
strip_columns = [
    'ZeroCouponBond', 'DividendValue', 'DividendREIT',
    'DividendInfra', 'DividendMarket', 'DividendSmall',
    'DividendGrowth', 'DividendNR', 'GainValue',
    'GainREIT', 'GainInfra', 'GainMarket',
    'GainSmall', 'GainGrowth', 'GainNR'
]

# Plotting
fig, axes = plt.subplots(nrows=sheets_to_load, ncols=1, figsize=(12, 14))  # Vertical layout
quarters = list(range(1, 65))  # 64 time points
if sheets_to_load == 1:
    axes = [axes]

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
    'blue', 'green', 'red', 'orange', 'brown', 'purple', 'deeppink'
]

# Build style dict
styles = {
    'ZeroCouponBond': {'color': 'black', 'linestyle': '-', 'marker':'o'}
}
for (dividend, gain), color in zip(strip_pairs, selected_colors):
    styles[dividend] = {'color': color, 'linestyle': '-', 'marker':'o'}
    styles[gain] = {'color': color, 'linestyle': '--','marker':'+'}


# Plotting loop
for ax, (asset_class, df) in zip(axes, results.items()) :
    print(df)
    B = df.iloc[:, 1:].to_numpy()
    n_years = B.shape[1] // group_size  # 64 // 4 = 16 years

    years = [1 + i for i in range(n_years)]

    for i, name in enumerate(strip_columns):
        if name not in styles:
            raise ValueError(f"Style for feature '{name}' is not defined.")

        yearly_means = B[i].reshape(n_years, group_size).mean(axis=1)

        ax.plot(
            years,
            yearly_means,
            label=name,
            color=styles[name]['color'],
            linestyle=styles[name]['linestyle'],
            marker=styles[name]['marker'],
            linewidth=.5  # slightly thicker for visibility
        )
    ax.set_xticks(years)  # show a tick for each year in your years list
    ax.set_xlim(years[0], years[-1])  # set x-axis limits from first to last year
    ax.set_title(asset_class)
    ax.set_xlabel("Years")
    ax.grid(True)

for i in range(sheets_to_load):
    axes[i].set_ylabel("Coefficient value")

# Unified legend outside the plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, ncols = 5,
    loc='lower center',      # position at the bottom center
    fontsize='large'         # make legend font size larger (you can also use an int like 12)
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # Make space for the legend
plt.show()
