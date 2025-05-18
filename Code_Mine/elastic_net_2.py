import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

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

# Initialize nested dictionary: {AssetClass: {FundID: {'X': [...], 'y': [...]}}}
structured_data = defaultdict(lambda: defaultdict(dict))

# Get unique asset classes
# asset_classes = merged['AssetClass'].dropna().unique()
# Filter to only relevant asset classes
asset_classes = ['Private Equity', 'Venture Capital', 'Real Estate']

for asset_class in asset_classes:
    df_fund = merged[merged['AssetClass'] == asset_class]

    # Ensure fund has enough data points
    if df_fund.shape[0] < 15:
        continue

    X = df_fund[strip_columns].values
    y = df_fund['ScaledCashflow'].values

    # Store data
    structured_data[asset_class]['X'] = X
    structured_data[asset_class]['y'] = y

print("Data prepared for elastic net regression.")
print("Elapsed time:", time.time() - start, "seconds")
start = time.time()
print("Starting elastic net regression...")

# To store results
results = []

for asset_class in structured_data:
    X = structured_data[asset_class]['X']
    y = structured_data[asset_class]['y']

    # Drop rows with NaNs in X or y
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    # Standardize X (like the R glmnet package does)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ElasticNet with 10-fold CV over a grid of alphas
    model = ElasticNetCV(
        l1_ratio=[.1, .3, .7, 1],  # alpha in R (mixing param)
        alphas=np.logspace(-4, 2, 10),    # lambda in R (penalty)
        cv=10,
        max_iter=5000,
        fit_intercept=True
    )

    model.fit(X_scaled, y)

    # Store results
    results.append({
        'AssetClass': asset_class,
        'BestAlpha': model.l1_ratio_,
        'BestLambda': model.alpha_,
        'Coefficients': model.coef_,
        'Intercept': model.intercept_,
        'R2': model.score(X_scaled, y),
        'NumObs': len(y)
    })

print(f"Completed Elastic Net regression on {len(results)} funds.")
print("Elapsed time:", time.time() - start, "seconds")

# Convert to DataFrame for export or analysis
results_df = pd.DataFrame(results)
print(results_df.head())
results_df.to_pickle("elastic_net_results.pkl")

