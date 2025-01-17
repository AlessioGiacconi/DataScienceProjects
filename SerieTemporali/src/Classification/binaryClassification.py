import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent

file_path = 'C:\\Users\\falco\\PycharmProjects\\ProgettoSerieTemporali\\DataScienceProjects\\SerieTemporali\\Dataset\\clean_dataset.csv'

df_csv = pd.read_csv(file_path)

df_csv['IsCancelled'] = df_csv['InvoiceNo'].astype(str).str.startswith('C').astype(int)

# Numero totale di ordini per prodotto
total_orders_by_product = df_csv.groupby('Description')['InvoiceNo'].nunique()

# Numero di cancellazioni per prodotto
cancelled_orders_by_product = df_csv[df_csv['IsCancelled'] == 1].groupby('Description')['InvoiceNo'].nunique()

# Creare un DataFrame con i dati
cancellation_stats = pd.DataFrame({
    'TotalOrders': total_orders_by_product,
    'CancelledOrders': cancelled_orders_by_product
})

# Sostituire i NaN (prodotti senza cancellazioni) con 0
cancellation_stats['CancelledOrders'] = cancellation_stats['CancelledOrders'].fillna(0)

# Calcolare la percentuale di cancellazioni
cancellation_stats['CancellationRate'] = (cancellation_stats['CancelledOrders'] / cancellation_stats['TotalOrders']) * 100

# Ordinare i prodotti per percentuale di cancellazioni
cancellation_stats = cancellation_stats.sort_values(by='CancellationRate', ascending=False)

# Calcolare la media di Quantity e UnitPrice per prodotto
product_stats = df_csv.groupby('Description').agg({
    'Quantity': 'mean',
    'UnitPrice': 'mean',
    'Country': lambda x: x.mode()[0]  # Prendere il paese più frequente
}).reset_index()

# Unire con il dataset delle cancellazioni
cancellation_stats = cancellation_stats.reset_index()  # Assicurarsi che l'indice sia un campo
classification_data = pd.merge(cancellation_stats, product_stats, on='Description', how='left')
classification_data['AtRisk'] = (classification_data['CancellationRate'] > 20).astype(int)
classification_data['Country'] = classification_data['Country'].astype('category').cat.codes

# Selezionare solo le colonne numeriche
numerical_data = classification_data.select_dtypes(include=['float64', 'int64'])

# Calcolare la matrice di correlazione
correlation_matrix = numerical_data.corr()

# Stampare le correlazioni con la variabile target "AtRisk"
print(correlation_matrix['AtRisk'].sort_values(ascending=False))

# Creare una heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice di Correlazione')
plt.show()

