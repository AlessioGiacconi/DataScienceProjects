import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
import seaborn as sns
import numpy as np

base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent / 'Dataset' / 'clean_dataset.csv'

df_csv = pd.read_csv(file_path)
df_csv['Hour'] = pd.to_datetime(df_csv['Time'], format='%H:%M:%S').dt.hour
max_unit_price = df_csv['UnitPrice'].max()
print(f"Il prezzo unitario più alto è: {max_unit_price}")
num_products_high_price = df_csv[df_csv['UnitPrice'] > 100].shape[0]
print(f"cacca: {num_products_high_price}")
# Filtrare le righe con Quantity negativa
df_csv_filtered = df_csv[df_csv['Quantity'] > 0]


#Grafico sulla distribuzione delle vendite per fascia oraria
plt.figure(figsize=(12, 6))
sns.histplot(data=df_csv_filtered, x='Hour', weights='Quantity', discrete=True, kde=True, color='orange', alpha=0.7)
plt.title("Distribuzione della quantità di prodotti acquistati per fascia oraria (£)")
plt.xlabel("Ora del giorno")
plt.ylabel("Quantità totale acquistata")
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Filtrare i prezzi per rimuovere valori estremamente alti
df_filtered = df_csv[df_csv['UnitPrice'] < 100].copy()  # Imposta un limite massimo ragionevole


#Grafico sulla distribuzione delle vendite in base al prezzo di vendita
plt.figure(figsize=(12, 6))
sns.histplot(df_filtered['UnitPrice'], kde=True, bins=30, color='skyblue', alpha=0.7, label='Distribuzione')
plt.title("Distribuzione del Prezzo di Acquisto")
plt.xlabel("Prezzo Unitario (£)")
plt.ylabel("Frequenza")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim(0, 50)
plt.xticks(ticks=range(0, 50, 5))
plt.show()


df_filtered['TotalRevenue'] = df_filtered['UnitPrice'] * df_filtered['Quantity']

# Creazione fasce di prezzo
bins = [0, 5, 10, 20, 50, 100]
labels = ['0-5', '5-10', '10-20', '20-50', '>50']
df_filtered['PriceRange'] = pd.cut(df_filtered['UnitPrice'], bins=bins, labels=labels, right=False)


revenue_by_price_range = df_filtered.groupby('PriceRange', observed=False)['TotalRevenue'].sum()
colors = colormaps['Pastel1'](np.linspace(0, 1, len(revenue_by_price_range)))
explode = [0.1, 0.05, 0.05, 0.05, 0.01]  # Esplodi la prima sezione più delle altre

#Grafico a torta che rappresenta i prodotti più venduti in base alle fasce di prezzo
plt.figure(figsize=(8, 8))
plt.pie(revenue_by_price_range.values, labels=revenue_by_price_range.index, autopct='%1.1f%%',
        startangle=140, colors=colors, explode=explode, wedgeprops={'edgecolor': 'black', 'linewidth': 1},
        labeldistance=1.1)  # Aumenta la distanza di tutte le etichette
plt.title("Percentuale degli Introiti per Fascia di Prezzo (Labeldistance)")
plt.show()

