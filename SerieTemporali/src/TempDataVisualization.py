import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent / 'Dataset' / 'clean_dataset.csv'

df_csv = pd.read_csv(file_path)

df_csv['Hour'] = pd.to_datetime(df_csv['Time'], format='%H:%M:%S').dt.hour

# Filtrare le righe con Quantity negativa
df_csv_filtered = df_csv[df_csv['Quantity'] > 0]

plt.figure(figsize=(12, 6))

# Usare bin discreti per ogni ora (senza centrare le barre)
sns.histplot(data=df_csv_filtered, x='Hour', weights='Quantity', discrete=True, kde=True, color='orange', alpha=0.7)

# Personalizzare il grafico
plt.title("Distribuzione della quantità di prodotti acquistati per fascia oraria")
plt.xlabel("Ora del giorno")
plt.ylabel("Quantità totale acquistata")
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Filtrare i prezzi per rimuovere valori estremamente alti
df_filtered = df_csv[df_csv['UnitPrice'] < 100]  # Imposta un limite massimo ragionevole

plt.figure(figsize=(12, 6))
sns.histplot(df_filtered['UnitPrice'], kde=True, bins=30, color='skyblue', alpha=0.7, label='Distribuzione')
plt.title("Distribuzione del Prezzo di Acquisto (UnitPrice, Dati Filtrati)")
plt.xlabel("Prezzo Unitario (€)")
plt.ylabel("Frequenza")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

df_csv['Log_UnitPrice'] = np.log1p(df_csv['UnitPrice'])

plt.figure(figsize=(12, 6))
sns.histplot(df_csv['Log_UnitPrice'], kde=True, bins=30, color='orange', alpha=0.7, label='Distribuzione (Log)')
plt.title("Distribuzione del Prezzo di Acquisto (Logaritmica)")
plt.xlabel("Log Prezzo Unitario (€)")
plt.ylabel("Frequenza")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()