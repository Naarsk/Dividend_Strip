from pathlib import Path
import pandas as pd
"""
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
"""

# Optional: for RDA export
"""
try:
    pandas2ri.activate()
    base = importr('base')
    saveRDS = ro.r['saveRDS']
    to_r = pandas2ri.py2rpy
except ImportError:
    saveRDS = None
"""

saveRDS = False

# --- Load Excel files ---
data_path = Path("./")
cashflows = pd.read_excel(data_path / "cashflows.xlsx", parse_dates=["TRANSACTION DATE"])
funds = pd.read_excel(data_path / "funds.xlsx", parse_dates=["FINAL CLOSE DATE"])

# --- Rename columns to match R expectations ---
cashflows.rename(columns={
    "FUND ID": "FUND_ID",
    "FIRM ID": "FIRM_ID",
    "TRANSACTION DATE": "TRANSACTION_DATE",
    "TRANSACTION TYPE": "TRANSACTION_TYPE",
    "TRANSACTION AMOUNT": "TRANSACTION_AMOUNT",
    "STRATEGY": "STRATEGY",
    "VINTAGE / INCEPTION YEAR": "VINTAGE_YEAR"
}, inplace=True)

funds.rename(columns={
    "FUND ID": "FUND_ID",
    "FIRM ID": "FIRM_ID",
    "STRATEGY": "STRATEGY",
    "VINTAGE / INCEPTION YEAR": "VINTAGE_YEAR",
    "FUND SIZE (USD MN)": "FUND_SIZE_USD_MN"
}, inplace=True)

# --- Merge funds info into cashflows ---
merged_cf = pd.merge(cashflows, funds, on="FUND_ID", how="left", suffixes=("", "_fund"))

# --- Create MASTER_Preqin_Returns with placeholder IRR/TVPI ---
returns = funds[["FUND_ID", "FIRM_ID", "STRATEGY", "VINTAGE_YEAR"]].drop_duplicates()
returns["Net.IRR...."] = 0.15  # Placeholder IRR
returns["Net.Multiple..X."] = 1.6  # Placeholder TVPI

# --- Create Preqin_PEPerf with fund size and static data ---
perf = funds[["FUND_ID", "FIRM_ID", "FUND_SIZE_USD_MN", "STRATEGY", "VINTAGE_YEAR"]].drop_duplicates()
perf["Net.IRR...."] = 0.15
perf["Net.Multiple..X."] = 1.6

# --- Export to CSV (exact filenames expected by loadPreqin.R) ---
merged_cf.to_csv("Preqin_PE_CashFlow_202042800_20200428031312.csv", index=False)
returns.to_csv("MASTER_Preqin_Returns.csv", index=False)
perf.to_csv("Preqin_PEPerf_20170919143905AllAlt.csv", index=False)

"""
# --- Optionally export to .rds (RDA equivalent) ---
if saveRDS:
    saveRDS(to_r(merged_cf), file="Preqin_PE_CashFlow_202042800_20200428031312.rds")
    saveRDS(to_r(returns), file="MASTER_Preqin_Returns.rds")
    saveRDS(to_r(perf), file="Preqin_PEPerf_20170919143905AllAlt.rds")
else:
    print("rpy2 not installed, RDS export skipped.")
"""