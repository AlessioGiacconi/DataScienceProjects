import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent
print(base_dir)
file_path = 'C:\\Users\\falco\\PycharmProjects\\ProgettoSerieTemporali\\DataScienceProjects\\SerieTemporali\\Dataset\\clean_dataset.csv'

df_csv = pd.read_csv(file_path)

# Caricare il dataset
df_csv['InvoiceDate'] = pd.to_datetime(df_csv['InvoiceDate'])
reference_date = df_csv['InvoiceDate'].max()

# Calcolo Recency, Frequency, Monetary
rfm = df_csv.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalPrice': 'sum'  # Monetary
}).reset_index()

# Rinominare le colonne
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Normalizzare i dati RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Metodo Elbow
inertia = []
range_n_clusters = range(1, 10)

for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

# Grafico del Metodo Elbow
plt.figure(figsize=(8, 6))
plt.plot(range_n_clusters, inertia, marker='o')
plt.title('Metodo Elbow per Determinare il Numero di Cluster')
plt.xlabel('Numero di Cluster')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Applicare K-Means
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Calcolare le statistiche medie per cluster
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'  # Numero di clienti per cluster
}).rename(columns={'CustomerID': 'Num_Customers'})

print(cluster_summary)

# Visualizzare il numero di clienti per cluster
plt.figure(figsize=(8, 6))
cluster_summary['Num_Customers'].plot(kind='bar', color='skyblue')
plt.title('Distribuzione dei Clienti per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Numero di Clienti')
plt.show()

