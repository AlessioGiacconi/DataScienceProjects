import pandas as pd

df_csv = pd.read_csv('C:\\Users\\falco\\PycharmProjects\\ProgettoSerieTemporali\\Dataset\\Online_Retail.csv')
df_csv.info()

# Controllo preliminare dei dati mancanti
missing_values = df_csv.isnull().sum()

# Controlliamo eventuali valori unici in alcune colonne chiave per individuare anomalie
unique_values_summary = {
    "InvoiceNo": df_csv["InvoiceNo"].nunique(),
    "StockCode": df_csv["StockCode"].nunique(),
    "CustomerID": df_csv["CustomerID"].nunique(),
    "Country": df_csv["Country"].nunique(),
}

print(missing_values)

print(unique_values_summary)

stockcode_total_count = df_csv["StockCode"].count()
print(stockcode_total_count)

unitprice_zero = df_csv[df_csv["UnitPrice"] == 0]