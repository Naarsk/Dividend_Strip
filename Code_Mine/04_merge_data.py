import time

import pandas as pd

H = 64

start = time.time()

print("Loading data...")
# Load data
cf_df = pd.read_csv("pe_cashflows_filled.csv")
strips_df = pd.read_csv("strips.csv")

print("Data loaded")
print("Elapsed time:", time.time() - start, "seconds")
print("Exporting data...")
start = time.time()

# Merge cashflows and strip realizations on Quarter and Vintage
merged = pd.merge(cf_df, strips_df, on=['Quarter', 'Vintage'], how='inner')

merged.to_excel('merged.xlsx', index=False)

merged.to_csv('dataset.csv', index=False)

print("Data exported")
print("Elapsed time:", time.time() - start, "seconds")