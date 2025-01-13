import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import squarify as sq

base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent / 'Dataset' / 'clean_dataset.csv'

df_csv = pd.read_csv(file_path)

'''spesa per paese'''

# Calcolare il totale delle vendite per ogni paese
country_sales = df_csv.groupby('Country')['TotalPrice'].sum()

# Applicare la trasformazione radice quadrata ai dati
country_sales_sqrt = country_sales.apply(lambda x: np.sqrt(x))

plt.figure(figsize=(14, 7))

# Grafico di distribuzione con istogramma e curva di densità
sns.histplot(country_sales_sqrt, kde=True, bins=100, color="skyblue", alpha=0.6, label="Distribuzione (Radice Quadrata)")

# Personalizzazione del grafico
plt.title("Distribuzione dei Costi per Stato (Scala Radice Quadrata - Tutti gli Stati)")
plt.xlabel("√ Totale Fatturato (€)")
plt.ylabel("Densità")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

'''numero  acquisti per paese'''

# Group by 'Country' and count the number of unique 'InvoiceNo'
country_invoice_counts = df_csv.groupby('Country')['InvoiceNo'].nunique()

# Sort by the number of invoices in descending order
country_invoice_counts = country_invoice_counts.sort_values(ascending=False)

# Apply square root to the values for square scale
sqrt_values = np.sqrt(country_invoice_counts)

# Plot the data
plt.figure(figsize=(14,8))
sns.barplot(x=country_invoice_counts.index, y=sqrt_values, color="skyblue", edgecolor="black")

plt.title("Numero di acquisti (Invoices) per Stato - Scala Quadrata", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Numero di acquisti", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)

yticks = plt.gca().get_yticks()
plt.gca().set_yticklabels([int(y**2) for y in yticks])

plt.tight_layout()
plt.show()

'''prodotti maggiormente acquistati'''

# Raggruppa per 'Description' e calcola il numero totale di unità vendute
product_sales = df_csv.groupby('Description')['Quantity'].sum().sort_values(ascending=False)

top_products = product_sales.head(25)

plt.figure(figsize=(14, 10))

sq.plot(
    sizes=top_products.values,
    label=[f"{product}\n{int(sales)} unità" for product, sales in zip(top_products.index, top_products.values)],
    alpha=0.8,
    color = sns.color_palette("viridis", len(top_products))
)

plt.title("Distribuzione delle Vendite per Prodotto (Top 25)", fontsize=16)
plt.axis('off')
plt.tight_layout()

plt.show()