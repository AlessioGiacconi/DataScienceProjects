import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import squarify as sq

base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent / 'Dataset' / 'clean_dataset_customerID.csv'

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

# Plot the data (scala quadrata)
plt.figure(figsize=(14,8))
sns.barplot(x=country_invoice_counts.index, y=sqrt_values, color="skyblue", edgecolor="black")

plt.title("Numero di acquisti per Stato - Scala Quadrata", fontsize=16)
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


''' numero di transazioni e spese totali per mese '''

df_csv['InvoiceDate'] = pd.to_datetime(df_csv['InvoiceDate'])

# Group by month and sum the total amount
df_csv['Month'] = df_csv['InvoiceDate'].dt.to_period('M')  # Extract year and month
monthly_revenue = df_csv.groupby('Month')['TotalPrice'].sum()  # Group by month and sum the Total_Amount

monthly_transactions = df_csv.groupby('Month')['InvoiceNo'].nunique()

# Create a plot with two y-axes to handle different scales for the two datasets

fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot the number of transactions on the left y-axis
ax1.bar(monthly_transactions.index.astype(str), monthly_transactions, color='coral', alpha=0.7, width=0.4, label='Numero di Transazioni', align='center')
ax1.set_ylabel('Numero di Transazioni', fontsize=12, color='coral')
ax1.tick_params(axis='y', labelcolor='coral')
ax1.set_xlabel('Mese', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.set_title('Numero di Transazioni e Spese Totali per Mese', fontsize=16)

# Create a second y-axis for the total revenue
ax2 = ax1.twinx()
ax2.bar(monthly_revenue.index.astype(str), monthly_revenue, color='lightgreen', alpha=0.7, width=0.4, label='Spese Totali (€)', align='edge')
ax2.set_ylabel('Spese Totali (€)', fontsize=12, color='lightgreen')
ax2.tick_params(axis='y', labelcolor='lightgreen')
ax2.get_yaxis().get_major_formatter().set_scientific(False)

# Add a legend to identify the two metrics
fig.legend(['Numero di Transazioni', 'Spese Totali (€)'], loc='upper left', fontsize=10)

# Adjust layout for better clarity
plt.tight_layout()

# Show the plot
plt.show()

''' quanti articoli vengono acquistati per ogni transazione'''

# Group by 'InvoiceNo' and calculate the total quantity of items for each invoice
invoice_quantities = df_csv.groupby('InvoiceNo')['Quantity'].sum()

# Define the new bins and labels for categorizing the quantity ranges
bins = [0, 10, 25, 50, 100, 200, 500, float('inf')]
labels = ['1-10', '10-25', '25-50', '50-100', '100-200', '200-500', '>500']

# Categorize the data into the new defined bins
invoice_quantities_binned = pd.cut(invoice_quantities,bins=bins, labels=labels, right=False)

# Count the number of invoice in each bin
quantity_distribution = invoice_quantities_binned.value_counts().sort_index()

# Plot the data
plt.figure(figsize=(12,8))
quantity_distribution.plot(kind='bar', color='sandybrown', edgecolor='black')

# Add title and lables
plt.title('Distribuzione delle Fatture per Fasce di Quantità di Articoli Acquistati', fontsize=16)
plt.xlabel('Fasce di Quantità di Articoli', fontsize=12)
plt.ylabel('Numero di Fatture', fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.tight_layout()

plt.show()
