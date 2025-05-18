import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

horizon = 64

pd.set_option('display.max_columns', None)

start = time.time()

print("Loading data...")
# Load data
cf_df = pd.read_csv("pe_cashflows_scaled.csv")
strips_df = pd.read_csv("strips.csv")

print("Data loaded")
print("Elapsed time:", time.time() - start, "seconds")
print("Preparing data...")
start = time.time()

# Merge cashflows and strip realizations on Quarter and Vintage
merged = pd.merge(cf_df, strips_df, on=['Quarter', 'Vintage'], how='inner')

# Define the QuartersSinceInception
merged['QuartersSinceInception'] = ((merged['Quarter'] - merged['Vintage']) / 0.25).astype(int)

# Drop missing values
merged.dropna(subset=['ScaledCashflow'], inplace=True)

# Define strip factor columns
strip_columns = [
    'ZeroCouponBond',
    'Dividend_Value', 'Dividend_REIT', 'Dividend_Infra', 'Dividend_Market',
    'Dividend_Small', 'Dividend_Growth', 'Dividend_NR',
    'Gain_Value', 'Gain_REIT', 'Gain_Infra', 'Gain_Market',
    'Gain_Small', 'Gain_Growth', 'Gain_NR'
]


# Get unique asset classes
# asset_classes = merged['AssetClass'].dropna().unique()
# Filter to only relevant asset classes
asset_classes = ['Private Equity', 'Venture Capital', 'Real Estate']


print("Data prepared for elastic net regression.")
print("Elapsed time:", time.time() - start, "seconds")
start = time.time()
print("Starting elastic net regression...")

# To store results
results = {}

for asset_class in asset_classes:
    print("Processing asset class", asset_class)

    for n_quarter in range(horizon):
        print("\tProcessing quarter", n_quarter)
        df_q = merged[[merged['QuartersSinceInception']==n_quarter, merged['AssetClass']==asset_class]]

        if df_q.shape[0] < 15:
            continue

        class_results = {}
        X = df_q[strip_columns]
        y = df_q['ScaledCashflow']

        # Drop rows with NaNs in X or y
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        # Standardize X (like the R glmnet package does)
        #scaler = StandardScaler()
        #X_scaled = scaler.fit_transform(X)

        # ElasticNet with 10-fold CV over a grid of alphas
        model = ElasticNetCV(
            #l1_ratio=[.1, .3, .7, 1],  # alpha in R (mixing param)
            #alphas=np.logspace(-4, 2, 10),    # lambda in R (penalty)
            #cv=10,
            max_iter=5000,
            fit_intercept=False,
            positive=True
        )

        #model.fit(X_scaled, y)
        model.fit(X, y)

        # Store results
        class_results[n_quarter] = {
            'AssetClass': asset_class,
            'BestAlpha': model.l1_ratio_,
            'BestLambda': model.alpha_,
            'Coefficients': model.coef_,
            'Intercept': model.intercept_,
            'R2': model.score(X_scaled, y),
            'NumObs': len(y)
        }
        print(model.coef_)

    results[asset_class] = class_results

print(f"Completed Elastic Net regression on {len(results)} asset classes.")
print("Elapsed time:", time.time() - start, "seconds")
print(f"Exporting...")

for asset_class in asset_classes:
    pd.DataFrame(results[asset_class]).to_excel(f"{asset_class}_results.xlsx")

print(f"Export done")
print("Elapsed time:", time.time() - start, "seconds")


