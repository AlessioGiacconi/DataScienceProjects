import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.pyplot import yticks

base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent / 'Dataset' / 'clean_dataset.csv'

df_csv = pd.read_csv(file_path)

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