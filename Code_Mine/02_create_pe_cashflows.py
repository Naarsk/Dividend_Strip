import pandas as pd
import pyreadr

pd.set_option('display.max_columns', None)

cashflow_file_path = "C:/Users/leocr/Projects/Economics/Finance/PE_dividend_strip/Data_Preqin_Raw/Cashflows.xlsx"
fund_file_path = "C:/Users/leocr/Projects/Economics/Finance/PE_dividend_strip/Data_Preqin_Raw/Funds.xlsx"
discount_rates_file_path = "C:/Users/leocr/Projects/Economics/Finance/PE_dividend_strip/Data/DiscountRatesOct20.Rda"

H=64

# Load data
df = pd.read_excel(cashflow_file_path)
df.columns = df.columns.str.strip()

cf_df = df[['FUND ID', 'ASSET CLASS', 'TRANSACTION DATE', 'NET CASHFLOW']].copy()
cf_df.columns = ['FundID', 'AssetClass', 'Date', 'Cashflow']
cf_df['Date'] = pd.to_datetime(cf_df['Date'], dayfirst=True, errors='coerce')

funds_df = pd.read_excel(fund_file_path)
funds_df.columns = funds_df.columns.str.strip()
metadata = funds_df[['FUND ID', 'FUND SIZE (USD MN)', 'VINTAGE / INCEPTION YEAR']].copy()
metadata.columns = ['FundID', 'CommittedCapital', 'VintageYear']

cf_df = pd.merge(cf_df, metadata, on='FundID', how='left')
cf_df.dropna(subset=['Date', 'Cashflow', 'CommittedCapital', 'VintageYear'], inplace=True)

cf_df['ScaledCashflow'] = cf_df['Cashflow'] / (cf_df['CommittedCapital'] * 1e6)

def convert_to_quarter_fraction(date):
    q = (date.month - 1) // 3
    return date.year + 0.25 * q

cf_df['Quarter'] = cf_df['Date'].apply(convert_to_quarter_fraction)
cf_df['Vintage'] = cf_df['VintageYear'].astype(float)
cf_df['QuartersSinceInception'] = ((cf_df['Quarter'] - cf_df['Vintage']) / 0.25).astype(int)

rda_data = pyreadr.read_r(discount_rates_file_path)
discount_rates_df = rda_data['late.discount.rates.quarterly']
discount_rates_df.columns = ['Quarter', 'ForwardRate', 'DiscountRate', 'Vintage']

## SPLIT CASHFLOWS AFTER 16y TO LAST £ QUARTERS

# Separate rows exceeding quarter 64
remaining_rows = cf_df[cf_df['QuartersSinceInception'] <= H].copy()
exceeding_rows = cf_df[cf_df['QuartersSinceInception'] > H].copy()

exceeding_rows = pd.merge(exceeding_rows, discount_rates_df, on=['Quarter','Vintage'], how='left')

# Create new redistributed rows
redistributed_rows = []

for _, row in exceeding_rows.iterrows():
    third_cashflow = row['ScaledCashflow'] / (3*row['DiscountRate'])
    for qsi in [H-2, H-1, H]:
        new_row = row.copy()
        new_row['QuartersSinceInception'] = qsi
        new_row['Quarter'] = row['Vintage'] + 0.25 * qsi
        new_row['ScaledCashflow'] = third_cashflow
        redistributed_rows.append(new_row)

# Create final DataFrame
redistributed_df = pd.DataFrame(redistributed_rows)
multiline_df = pd.concat([remaining_rows, redistributed_df], ignore_index=True)

#multiline_df =cf_df[cf_df['QuartersSinceInception'] <= H].copy()

## AGGREGATE

# Define columns that should be identical within each group (sanity check)
identity_columns = ['AssetClass', 'QuartersSinceInception']  # Add more if needed

# Group by FundID, Quarter, Vintage and perform the aggregation
group_cols = ['FundID', 'Quarter', 'Vintage']

# Check if identity columns are equal within each group
def check_unique_values(group):
    for col in identity_columns:
        if group[col].nunique() > 1:
            raise ValueError(f"Inconsistent values in column '{col}' for group {group.name}")
    return group.iloc[0]  # return one row with the preserved values

# Aggregate cashflows
summed_cf = multiline_df.groupby(group_cols, group_keys=False).agg({
    'ScaledCashflow': 'sum'}).reset_index()

# Reattach static fields (after verifying consistency)
metadata = multiline_df.groupby(group_cols).apply(check_unique_values).reset_index(drop=True)

# Merge back the summed ScaledCashflow
final_df = pd.merge(summed_cf, metadata.drop(columns=['ScaledCashflow']), on=group_cols, how='left')


## EXPORT

# Sort
final_df.sort_values(by=['FundID', 'QuartersSinceInception'], inplace=True)

# Save to file
final_df[['FundID', 'QuartersSinceInception', 'ScaledCashflow', 'AssetClass', 'Quarter', 'Vintage']].to_csv("pe_discounted_cashflows.csv", index=False)

