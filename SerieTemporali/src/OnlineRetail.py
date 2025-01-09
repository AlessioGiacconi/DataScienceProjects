import pandas as pd

df_csv = pd.read_csv('C:\\Users\\falco\\PycharmProjects\\ProgettoSerieTemporali\\DataScienceProjects\\SerieTemporali\\Dataset\\Online_Retail.csv')
df_csv.info()

df_csv = df_csv.dropna(subset=['CustomerID'])

df_csv.info()

unit_price_not_zero = df_csv[df_csv['UnitPrice'] > 0]

unit_price_not_zero['TotalPrice'] = (unit_price_not_zero['UnitPrice'] * unit_price_not_zero['Quantity']).round(2)
df_csv.info()