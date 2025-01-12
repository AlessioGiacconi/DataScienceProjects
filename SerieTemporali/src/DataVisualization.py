import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.pyplot import yticks

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

# Prendi i primi 20 prodotti più venduti
top_products = product_sales.head(10)
'''
# Shuffle the top products to create a random order
shuffled_top_products = top_products.sample(frac=1, random_state=42)

# Generate a new sequential index for the shuffled x-axis
x_shuffled = range(len(shuffled_top_products))

# Palette di colore da seaborn
colors = sns.color_palette("viridis", len(top_products))

# Crea il grafico a bolle
plt.figure(figsize=(16, 9))
plt.scatter(x=x_shuffled, y=shuffled_top_products.values, s=shuffled_top_products.values*0.3, alpha=0.8, c=colors, edgecolor="black")

# Aggiungi etichette ai punti
for i, (product,quantity) in enumerate(zip(shuffled_top_products.index, shuffled_top_products.values)):
    plt.text(x_shuffled[i], quantity + 1000, product, ha='center', va='center',fontsize=9, color='black', weight='bold')

# Titolo e etichette degli assi
plt.title('Top 20 Prodotti Maggiormente Venduti - Grafico a Bolle', fontsize = 18)
plt.xlabel('Prodotti', fontsize=12)
plt.ylabel('Unità Vendute', fontsize=12)

plt.xticks([])

plt.tight_layout()
plt.show()'''

plt.figure(figsize=(14, 14))
wedges, texts, autotexts = plt.pie(top_products.values, labels=top_products.index, autopct=lambda p: f'{int(p * sum(top_products.values) / 100)}', startangle=140, colors=sns.color_palette("viridis", len(top_products)))

# Style the text
for text in texts:
    text.set_color('black')
    text.set_fontsize(10)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

# Reduce distance of labels from the chart
for text in texts:
    text.set_position((text.get_position()[0] * 0.93, text.get_position()[1] * 0.93))

# Add a title
plt.title("Distribuzione Percentuale dei Top 10 Prodotti Maggiormente Venduti", fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

# DA DISCUTERE

# Calculate the total quantities for the top 10 products
top_products = df_csv.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# Calculate the "Altro" category (sum of all other products)
other_quantity = df_csv.groupby('Description')['Quantity'].sum().sort_values(ascending=False).iloc[10:].sum()

# Add "Altro" to the top_products series
top_products_with_other = pd.concat([top_products, pd.Series({'Altro': other_quantity})])

# Create a pie chart with the "Altro" category
plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(
    top_products_with_other.values,
    labels=top_products_with_other.index,  # Product names + "Altro"
    autopct=lambda p: f'{int(p * sum(top_products_with_other.values) / 100)}',  # Show number of articles sold
    startangle=140,
    colors=sns.color_palette("viridis", len(top_products_with_other))
)

# Style the text
for text in texts:
    text.set_color('black')  # Labels in black
    text.set_fontweight('bold')
    text.set_fontsize(8)  # Smaller font size for product names
for autotext in autotexts:
    autotext.set_color('white')  # Numbers inside sections in white
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)  # Adjust size for numbers

# Reduce distance of labels from the chart
for text in texts:
    text.set_position((text.get_position()[0] * 0.9, text.get_position()[1] * 0.9))  # Move labels closer

# Add a title
plt.title("Distribuzione Percentuale dei Prodotti Venduti (Top 10 + Altro)", fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()