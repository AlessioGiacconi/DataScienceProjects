import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
import seaborn as sns
import numpy as np
import textwrap

base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent / 'Dataset' / 'clean_dataset.csv'

df_csv = pd.read_csv(file_path)
df_csv['Hour'] = pd.to_datetime(df_csv['Time'], format='%H:%M:%S').dt.hour

# Filtrare le righe che rappresentano cancellazioni (InvoiceNo inizia con "C")
cancellations = df_csv[df_csv['InvoiceNo'].str.startswith('C')]
total_cancellations = cancellations.shape[0]
total_orders = df_csv.shape[0]
print(f"Totale Cancellazioni: {total_cancellations}")
print(f"Percentuale di Cancellazioni: {total_cancellations / total_orders * 100:.2f}%")

cancellations = df_csv[df_csv['InvoiceNo'].str.startswith('C')].copy()
cancellations['Quantity'] = cancellations['Quantity'].abs()
cancellations_by_product = cancellations.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
wrapped_labels = [textwrap.fill(label, width=20) for label in cancellations_by_product.index]  # Regola `width` per il numero di caratteri

#Grafico a barre orizzontali sui prodotti più restituiti
plt.figure(figsize=(12, 8))
sns.barplot(y=wrapped_labels, x=cancellations_by_product.values, palette="coolwarm_r")
plt.title("Top 10 Prodotti Più Cancellati")
plt.xlabel("Quantità Totale Cancellata")
plt.ylabel("Descrizione Prodotto")
plt.yticks(fontsize=8)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Filtrare le righe con Quantity negativa
df_csv_filtered = df_csv[df_csv['Quantity'] > 0]


country_sales = df_csv.groupby('Country')['TotalPrice'].sum()
country_sales_sqrt = np.sqrt(country_sales)
country_sales_sqrt = country_sales_sqrt.sort_values(ascending=False)

#Grafico a barre orizzontali sui costi sostenuti per Stato
plt.figure(figsize=(12, 10))
sns.barplot(y=country_sales_sqrt.index, x=country_sales_sqrt.values, palette="coolwarm_r", alpha=0.8)
plt.title("Distribuzione dei costi sostenuti per Stato (scala radice quadrata)")
plt.xlabel("√ Totale fatturato (£)")
plt.ylabel("Paese")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

country_quantity = df_csv.groupby('Country')['Quantity'].sum()
country_quantity_sqrt = np.sqrt(country_quantity)
country_quantity_sqrt = country_quantity_sqrt.sort_values(ascending=False)

#Grafico a barre orizzontali sulle quantità vendute per Stato
plt.figure(figsize=(12, 10))
sns.barplot(y=country_quantity_sqrt.index, x=country_quantity_sqrt.values, palette="coolwarm_r", alpha=0.8)
plt.title("Distribuzione delle quantità di prodotti venduti per Stato")
plt.xlabel("√ Totale venduto")
plt.ylabel("Paese")
plt.grid(axis='x', linestyle='--', alpha=0.7)

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

