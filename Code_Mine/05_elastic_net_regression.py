import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multitask_elastic import cross_validate_elastic_net, multitask_elastic_net, generate_alpha_grid

pd.set_option('display.max_columns', None)

start = time.time()

print("Loading data...")
# Load data

df = pd.read_csv("dataset.csv")

# Find FundsIDs with at least one Quarter > 2019
ids_to_drop = df.loc[df["Quarter"] > 2019, "FundID"].unique()

# Drop all rows with those FundsIDs
df = df[~df["FundID"].isin(ids_to_drop)].reset_index(drop=True)

print("Data loaded")
print("Elapsed time:", time.time() - start, "seconds")

# Filter to only relevant asset classes
asset_classes = ['Private Equity', 'Venture Capital', 'Real Estate'] #

strip_columns = [
    'ZeroCouponBond', 'DividendValue', 'DividendREIT',
    'DividendInfra', 'DividendMarket', 'DividendSmall',
    'DividendGrowth', 'DividendNR', 'GainValue',
    'GainREIT', 'GainInfra', 'GainMarket',
    'GainSmall', 'GainGrowth', 'GainNR'
]

horizon = 64
strips = 15

print("Starting elastic net regression...")
# To store results
results = {}

for asset_class in asset_classes:
    print("Processing asset class", asset_class)
    df_ac = df[df['AssetClass'] == asset_class]

    funds = df_ac['FundID'].unique()
    n_funds = len(df_ac['FundID'].unique())

    F = np.zeros((n_funds, strips, horizon))
    X = np.zeros((n_funds, horizon))

    for i in range(n_funds):
        fund = funds[i]
        X[i] = df_ac[df_ac['FundID'] == fund]['ScaledCashflow'].values
        F[i] = df_ac[df_ac['FundID'] == fund][strip_columns].values.T

        # Zero out rows in X[i] where y[i] is zero
        # zero_indices = np.where(X[i] == 0)[0]
        # F[i][:, zero_indices] = 0

    F = (F - np.mean(F, axis=0))
    X = (X - np.mean(X, axis=0))

    alphas = generate_alpha_grid(X=F,y=X,n_alphas=25)
    best_alpha, best_l1_ratio = cross_validate_elastic_net(X=F, y=X, alphas=alphas, l1_ratios=[10**(-6),10**(-7),10**(-8), 10**(-9), 0])
    B_final = multitask_elastic_net(X=F, Y=X, alpha=best_alpha, l1_ratio=best_l1_ratio)
    results[asset_class] = B_final

# optionally: index = feature names, columns = task names
with pd.ExcelWriter('old/multitask_elastic_net_results_6.xlsx') as writer:
    for asset_class, B in results.items():
        df = pd.DataFrame(B)
        df.to_excel(writer, sheet_name=asset_class[:31], index=False)

# Assuming B has shape (K=15, H=64)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)
quarters = list(range(1, 65))  # x-axis: 1 to 64

for ax, (asset_class, B) in zip(axes, results.items()):
    B = np.array(B)  # Ensure it's a NumPy array
    for i in range(B.shape[0]):  # K=15 rows (features)
        ax.plot(quarters, B[i], label=f'{strip_columns[i]}')

    ax.set_title(asset_class)
    ax.set_xlabel("Quarter")
    ax.grid(True)

axes[0].set_ylabel("Coefficient value")
axes[0].legend(loc='upper right', fontsize='small', ncol=2)

plt.tight_layout()
plt.show()