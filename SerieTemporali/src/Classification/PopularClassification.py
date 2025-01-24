import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Caricamento del Dataset
# =========================
file_path = 'C:\\Users\\falco\\PycharmProjects\\ProgettoSerieTemporali\\DataScienceProjects\\SerieTemporali\\Dataset\\clean_dataset.csv'
df = pd.read_csv(file_path)

# =========================
# Creazione del target binario
# =========================
df['IsInternational'] = (df['Country'] != 'United Kingdom').astype(int)

# =========================
# Preprocessing delle feature
# =========================
# Feature numeriche principali
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceMonth'] = df['InvoiceDate'].dt.month  # Mese come variabile numerica

# Aggiungiamo anche il totale dell'ordine (opzionale)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['DescriptionFrequency'] = df['Description'].map(df['Description'].value_counts())

# Variabili candidate per la matrice di correlazione
features = ['Quantity', 'UnitPrice', 'InvoiceMonth', 'TotalPrice', 'IsInternational', 'DescriptionFrequency']

# =========================
# Matrice di correlazione
# =========================
correlation_matrix = df[features].corr()

# Visualizzazione della heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice di Correlazione tra le Variabili')
plt.show()
pass