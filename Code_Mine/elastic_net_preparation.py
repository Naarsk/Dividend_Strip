import time

import pandas as pd

H = 64

start = time.time()

print("Loading data...")
# Load data
cf_df = pd.read_csv("pe_cashflows_scaled.csv")
strips_df = pd.read_csv("strips.csv")

print("Data loaded")
print("Elapsed time:", time.time() - start, "seconds")
print("Exporting data...")
start = time.time()

# Define Quarters since inception
cf_df['QuartersSinceInception'] = ((cf_df['Quarter'] - cf_df['Vintage']) / 0.25).astype(int)

# Merge cashflows and strip realizations on Quarter and Vintage
merged = pd.merge(cf_df, strips_df, on=['Quarter', 'Vintage'], how='inner')


merged.to_excel('merged.xlsx', index=False)




# If Quarter > H-1=63, distribute ScaledCashflow between the last 3:
exceeding = merged[merged['QuartersSinceInception'] > 63]

for idx, row in exceeding.iterrows():
    fund_id = row['FundID']
    vintage = row['Vintage']
    asset_class = row['AssetClass']
    cashflow_share = row['ScaledCashflow'] / 3

    for q in [61, 62, 63]:
        # Boolean mask for matching row
        mask = (
            (merged['FundID'] == fund_id) &
            (merged['Vintage'] == vintage) &
            (merged['AssetClass'] == asset_class) &
            (merged['QuartersSinceInception'] == q)
        )

        if merged.loc[mask].shape[0] == 1:
            # Row exists, add the value
            merged.loc[mask, 'ScaledCashflow'] += cashflow_share
        elif merged.loc[mask].shape[0] == 0:
            # Optional: create row if missing (comment this block to skip creation)
            new_row = row.copy()
            new_row['QuartersSinceInception'] = q
            new_row['ScaledCashflow'] = cashflow_share
            merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)
        else:
            raise ValueError("Unexpected duplicate rows found during redistribution. Row:", q)


# Drop missing values
merged.dropna(subset=['ScaledCashflow','ZeroCouponBond',
    'Dividend_Value', 'Dividend_REIT', 'Dividend_Infra', 'Dividend_Market',
    'Dividend_Small', 'Dividend_Growth', 'Dividend_NR',
    'Gain_Value', 'Gain_REIT', 'Gain_Infra', 'Gain_Market',
    'Gain_Small', 'Gain_Growth', 'Gain_NR'], inplace=True)

merged.to_excel('merged.xlsx', index=False)

print("Data exported")
print("Elapsed time:", time.time() - start, "seconds")