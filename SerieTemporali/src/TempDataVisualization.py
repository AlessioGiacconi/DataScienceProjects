import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
import seaborn as sns
import numpy as np
import textwrap

base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent / 'Dataset' / 'clean_dataset_customerID.csv'

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
sns.barplot(y=wrapped_labels, x=cancellations_by_product.values, palette="viridis")
plt.title("Top 10 Prodotti Più Cancellati", fontsize=18)
plt.xlabel("Quantità Totale Cancellata", fontsize=15)
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
plt.title("Costi sostenuti per Stato (scala radice quadrata)", fontsize=17)
plt.xlabel("√ Totale fatturato (£)", fontsize=14)
plt.ylabel("Paese", fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

country_quantity = df_csv.groupby('Country')['Quantity'].sum()
country_quantity_sqrt = np.sqrt(country_quantity)
country_quantity_sqrt = country_quantity_sqrt.sort_values(ascending=False)

#Grafico a barre orizzontali sulle quantità vendute per Stato
plt.figure(figsize=(12, 10))
sns.barplot(y=country_quantity_sqrt.index, x=country_quantity_sqrt.values, palette="coolwarm_r", alpha=0.8)
plt.title("Quantità di prodotti venduti per Stato", fontsize=17)
plt.xlabel("√ Totale venduto", fontsize=14)
plt.ylabel("Paese", fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Grafico sulla distribuzione delle vendite per fascia oraria
plt.figure(figsize=(12, 6))
sns.histplot(data=df_csv_filtered, x='Hour', weights='Quantity', discrete=True, kde=True, color='orange', alpha=0.7)
plt.title("Distribuzione della quantità di prodotti acquistati per fascia oraria (£)", fontsize=17)
plt.xlabel("Ora del giorno", fontsize=14)
plt.ylabel("Quantità totale acquistata", fontsize=14)
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Filtrare i prezzi per rimuovere valori estremamente alti
df_filtered = df_csv[df_csv['UnitPrice'] < 100].copy()  # Imposta un limite massimo ragionevole


#Grafico sulla distribuzione delle vendite in base al prezzo di vendita
plt.figure(figsize=(12, 6))
sns.histplot(df_filtered['UnitPrice'], kde=True, bins=30, color='skyblue', alpha=0.7, label='Distribuzione')
plt.title("Distribuzione del Prezzo di Acquisto", fontsize=17)
plt.xlabel("Prezzo Unitario (£)", fontsize=14)
plt.ylabel("Frequenza", fontsize=14)
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
colors = ['red', 'green', 'orange', 'lightblue', 'yellow']
explode = [0.1, 0.1, 0.1, 0.1, 0.25]  # Esplodi la prima sezione più delle altre

# Grafico a torta con ombra e spazio tra le fette
# Grafico a torta con ombra e spazio tra le fette
plt.figure(figsize=(13, 13))

# Crea il grafico a torta con percentuali sugli spicchi
wedges, texts, autotexts = plt.pie(revenue_by_price_range.values,
                                   labels=None,  # Rimuove le etichette dei nomi dagli spicchi
                                   autopct='%1.2f%%',  # Mostra le percentuali sugli spicchi
                                   startangle=140,
                                   colors=colors,
                                   explode=explode,
                                   shadow=True,  # Aggiungi ombra
                                   wedgeprops={'edgecolor': 'black', 'linewidth': 1})  # Bordo delle fette

# Personalizza la posizione e lo stile delle percentuali (autotexts)
for autotext in autotexts:
    autotext.set_color('black')  # Cambia il colore delle percentuali
    autotext.set_fontsize(18)    # Cambia la dimensione del font

# Aggiungi una legenda
plt.legend(wedges, revenue_by_price_range.index,  # Fasce di prezzo come etichette
           title="Fasce di Prezzo",
           title_fontsize=20,  # Aumenta la dimensione del titolo della legenda
           loc="center left",  # Posizione della legenda
           bbox_to_anchor=(-0.15, 0.5),
           prop={'size': 18})  # Aumenta la dimensione del font della legenda

# Aggiungi un titolo
plt.title("Percentuale degli Introiti per Fascia di Prezzo", fontsize=23)

# Mostra il grafico
plt.show()

