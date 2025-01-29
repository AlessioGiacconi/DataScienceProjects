import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent.parent / 'Dataset' / 'clean_dataset.csv'

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

# Calcolo delle distanze k più vicine
neighbors = NearestNeighbors(n_neighbors=5)  # min_samples
neighbors_fit = neighbors.fit(rfm_scaled)
distances, indices = neighbors_fit.kneighbors(rfm_scaled)

# Ordinamento delle distanze
distances = np.sort(distances[:, -1])  # Considerare la distanza k-esima
plt.figure(figsize=(8, 6))
plt.plot(distances)
plt.title('K-Distance Plot')
plt.xlabel('Punti Ordinati')
plt.ylabel('Distanza K-esima')
plt.grid(True)
plt.show()

# Applicazione di DBSCAN
dbscan = DBSCAN(eps=0.6, min_samples=5)  # Puoi regolare eps e min_samples
rfm['Cluster'] = dbscan.fit_predict(rfm_scaled)

# Visualizzazione dei cluster generati
print(rfm['Cluster'].value_counts())


#pca = PCA(n_components=3)
#rfm_pca_3d = pca.fit_transform(rfm_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot utilizzando Recency, Frequency, e Monetary
scatter = ax.scatter(
    rfm_scaled[:, 0],  # Recency normalizzato
    rfm_scaled[:, 1],  # Frequency normalizzato
    rfm_scaled[:, 2],  # Monetary normalizzato
    c=rfm['Cluster'], cmap='viridis', s=50, alpha=0.7
)

# Titoli e assi
ax.set_title('DBSCAN Clustering')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

# Aggiungere la legenda
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Cluster')

# Mostrare il grafico
plt.show()

# Statistiche per cluster
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Num_Customers'})

print(cluster_summary)
'''
best_min_samples = None
best_score = -1

for min_samples in range(3, 10):
    dbscan = DBSCAN(eps=0.6, min_samples=min_samples)
    labels = dbscan.fit_predict(rfm_scaled)

    # Ignorare il calcolo del Silhouette Score se tutti i punti sono outlier
    if len(set(labels)) > 1:
        score = silhouette_score(rfm_scaled, labels)
        print(f"Silhouette Score per min_samples = {min_samples}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_min_samples = min_samples

print(f"Il miglior valore di min_samples è {best_min_samples} con Silhouette Score = {best_score:.4f}")
'''