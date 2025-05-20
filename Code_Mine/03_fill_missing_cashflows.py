import pandas as pd

pd.set_option('display.max_columns', None)

H=64

# Load data
df = pd.read_csv("pe_discounted_cashflows.csv")

# Step 1: Create full quarter index per fund
all_qsi = pd.DataFrame({'QuartersSinceInception': range(1,H+1)})

all_combinations = (
    df[['FundID']].drop_duplicates()
    .merge(all_qsi, how='cross')
)

# Step 2: Add back fund metadata
all_combinations = pd.merge(
    all_combinations,
    df[['FundID', 'Vintage', 'AssetClass']].drop_duplicates('FundID'),
    on='FundID',
    how='left'
)

# Step 3: Compute 'Quarter' from Vintage and QSI
all_combinations['Quarter'] = all_combinations['Vintage'] + 0.25 * all_combinations['QuartersSinceInception']

# Step 4: Merge actual cashflows
filled_df = pd.merge(
    all_combinations,
    df[['FundID', 'Quarter', 'ScaledCashflow']],
    on=['FundID', 'Quarter'],
    how='left'
)

# Step 5: Fill missing ScaledCashflow with 0
filled_df['ScaledCashflow'] = filled_df['ScaledCashflow'].fillna(0)

# Step 6: Sort
filled_df.sort_values(by=['FundID', 'QuartersSinceInception'], inplace=True)

# Step 7: Export
filled_df[['FundID', 'QuartersSinceInception', 'ScaledCashflow', 'AssetClass', 'Quarter', 'Vintage']].to_csv("pe_cashflows_filled.csv", index=False)

print("Filled dataset exported to pe_cashflows_filled.csv")
