import pandas as pd
import pyreadr

pd.set_option('display.max_columns', None)

# Load the RDA file
path = r"C:/Users/leocr/Projects/Economics/Finance/PE_dividend_strip/Data/DividendStripOct20.Rda"
rda_data = pyreadr.read_r(path)

# Extract the relevant DataFrame containing the strip realizations
dividend_strips = rda_data['cohort.series.quarterly']
gain_series = rda_data['dividend.strip.capital.gains']

# Define the dividend and gain columns
dividend_columns = dividend_strips[
    ['Transaction.Quarter', 'Vintage.Quarter', 'cohort.value',
     'cohort.reit', 'cohort.infra', 'cohort.stock',
     'cohort.small', 'cohort.growth', 'cohort.nr']
]

gain_columns = gain_series[
    ['Transaction.Quarter', 'Vintage.Quarter', 'gain.cohort.value',
     'gain.cohort.reit', 'gain.cohort.infra', 'gain.cohort.stock',
     'gain.cohort.small', 'gain.cohort.growth', 'gain.cohort.nr']
]


merged = pd.merge(
    dividend_columns,
    gain_columns,
    on=['Transaction.Quarter', 'Vintage.Quarter'],
    how='left'
)

# Add the zero-coupon bond factor as a column of ones
merged.insert(2, 'ZeroCouponBond', 1.0)

# Rename columns for consistency
merged.columns = [
    'Quarter', 'Vintage', 'ZeroCouponBond',
    'Dividend_Value', 'Dividend_REIT', 'Dividend_Infra', 'Dividend_Market',
    'Dividend_Small', 'Dividend_Growth', 'Dividend_NR',
    'Gain_Value', 'Gain_REIT', 'Gain_Infra', 'Gain_Market',
    'Gain_Small', 'Gain_Growth', 'Gain_NR'
]

# Final check
print(merged.head())

# Export to CSV for use in regression step
merged.to_csv("strips.csv", index=False)
