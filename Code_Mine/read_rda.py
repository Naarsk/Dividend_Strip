import pandas as pd
import pyreadr

pd.set_option('display.max_columns', None)

# Load the .Rda file
result = pyreadr.read_r("C:/Users/leocr/Projects/Economics/Finance/PE_dividend_strip/Data/DiscountRatesOct20.Rda")

# Check what objects are in the file
print(result.keys())  # e.g., dict_keys(['DividendStripOct20'])

# Extract each as a pandas DataFrame
for name, obj in result.items():
    if isinstance(obj, pd.DataFrame):
        print(f"\nObject: {name}")
        print(obj.head())  # View top rows
        print(obj.tail())  # View top rows

        # Optionally save or work with the DataFrame:
        # obj.to_csv(f"{name}.csv", index=False)
