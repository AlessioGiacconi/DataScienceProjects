import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent / 'Dataset' / 'Online_Retail.csv'

df_csv = pd.read_csv(file_path)
df_csv.info()

new = df_csv[df_csv['StockCode']=='22145']
print(new)
df_csv = df_csv.dropna(subset=['CustomerID'])

#df_csv.info()

unit_price_not_zero = df_csv[df_csv['UnitPrice'] > 0].copy()

unit_price_not_zero['TotalPrice'] = (unit_price_not_zero['UnitPrice'] * unit_price_not_zero['Quantity']).round(2)

unit_price_not_zero['InvoiceDate'] = pd.to_datetime(unit_price_not_zero['InvoiceDate'])

unit_price_not_zero['Date'] = unit_price_not_zero['InvoiceDate'].dt.date
unit_price_not_zero['Time'] = unit_price_not_zero['InvoiceDate'].dt.time
unit_price_not_zero.info()

unique_descriptions_count = df_csv['Description'].nunique()
print(unique_descriptions_count)
