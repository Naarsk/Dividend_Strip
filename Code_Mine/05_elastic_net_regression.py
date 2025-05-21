import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ADMM_elastic import cross_validate_elastic_net, ADMM_elastic_net

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pd.set_option('display.max_columns', None)

start = time.time()
logging.info("Loading data...")

# Load data
df = pd.read_csv("dataset.csv")

# Drop FundsIDs with any Quarter > 2019
ids_to_drop = df.loc[df["Quarter"] > 2019, "FundID"].unique()
df = df[~df["FundID"].isin(ids_to_drop)].reset_index(drop=True)

logging.info("Data loaded")
logging.info("Elapsed time: %.2f seconds", time.time() - start)

# Filter to only relevant asset classes
asset_classes = ['Private Equity', 'Venture Capital', 'Real Estate']

strip_columns = [
    'ZeroCouponBond', 'DividendValue', 'DividendREIT',
    'DividendInfra', 'DividendMarket', 'DividendSmall',
    'DividendGrowth', 'DividendNR', 'GainValue',
    'GainREIT', 'GainInfra', 'GainMarket',
    'GainSmall', 'GainGrowth', 'GainNR'
]

horizon = 64
strips = 15

logging.info("Starting elastic net regression...")

results = {}
max_iter = 15000

for asset_class in asset_classes:
    logging.info("Processing asset class: %s", asset_class)
    df_ac = df[df['AssetClass'] == asset_class]
    funds = df_ac['FundID'].unique()
    n_funds = len(funds)

    F = np.zeros((n_funds, strips, horizon))
    X = np.zeros((n_funds, horizon))

    for i in range(n_funds):
        fund = funds[i]
        X[i] = df_ac[df_ac['FundID'] == fund]['ScaledCashflow'].values
        F[i] = df_ac[df_ac['FundID'] == fund][strip_columns].values.T

    for h in range(horizon):
        F[:, :, h] = StandardScaler().fit_transform(F[:, :, h])
        X[:, h] -= X[:, h].mean()

    l1_ratios = [1, 0.99, 0.7, 0.5, 0.3, 0.1, 1e-3, 1e-5, 1e-7]
    best_l1_ratio, best_alpha = cross_validate_elastic_net(x=F, y=X, l1_ratios=l1_ratios, n_alphas=5, cv_splits=5, max_iter=max_iter)

    logging.info("Best l1_ratio: %.5f, alpha: %.5f for asset class: %s", best_l1_ratio, best_alpha, asset_class)

    B_final = ADMM_elastic_net(x=F, y=X, alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=max_iter)
    results[asset_class] = {'B': B_final, 'l1': best_l1_ratio, 'alpha': best_alpha}

    logging.info("Finished processing %s", asset_class)
    logging.info("Elapsed time: %.2f seconds", time.time() - start)

# Export results
logging.info("Exporting results to Excel...")

with pd.ExcelWriter('old/multitask_elastic_net_results_6.xlsx') as writer:
    summary_data = []

    for asset_class in results:
        B = results[asset_class]['B']
        l1 = results[asset_class]['l1']
        alpha = results[asset_class]['alpha']

        df = pd.DataFrame(B, index=strip_columns)
        df.to_excel(writer, sheet_name=asset_class[:31], index=True)

        summary_data.append({'Asset Class': asset_class, 'l1_ratio': l1, 'alpha': alpha})

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

logging.info("Results exported successfully.")
logging.info("Total elapsed time: %.2f seconds", time.time() - start)
