import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples


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
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot con i clustercolors = ['blue', 'orange', 'green', 'red']
for cluster in rfm['Cluster'].unique():
    cluster_data = rfm[rfm['Cluster'] == cluster]
    ax.scatter(
        cluster_data['Recency'],
        cluster_data['Frequency'],
        cluster_data['Monetary'],
        label=f'Cluster {cluster}',
        s=40,
        alpha=0.6
    )

centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], s=20, c='black', label='Centroidi', marker='o')

ax.view_init(elev=20, azim=220)

ax.set_title('K-Means Clustering', fontsize=16)
ax.set_xlabel('Recency', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_zlabel('Monetary', fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

'''plt.figure(figsize=(8, 6))
cluster_summary['Num_Customers'].plot(kind='bar', color='skyblue')
plt.title('Distribuzione dei Clienti per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Numero di Clienti')
plt.show()'''

# Calcolare i silhouette scores
# Calcolare i silhouette scores
silhouette_vals = silhouette_samples(rfm_scaled, kmeans.labels_)

# Creare il Silhouette Plot senza sovrapposizioni
plt.figure(figsize=(10, 6))

y_lower = 0
tick_positions = []  # Per memorizzare le posizioni dei tick

for i in range(optimal_k):
    # Selezionare i silhouette scores del cluster corrente
    cluster_silhouette_vals = silhouette_vals[kmeans.labels_ == i]
    cluster_silhouette_vals.sort()
    cluster_size = len(cluster_silhouette_vals)
    y_upper = y_lower + cluster_size

    # Disegnare la banda per il cluster corrente
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7, label=f"Cluster {i}")

    # Calcolare la posizione centrale per l'etichetta del cluster
    tick_positions.append((y_lower + y_upper) / 2)
    y_lower = y_upper + 10  # Aggiungere uno spazio tra i cluster

# Aggiungere le etichette dei cluster all'asse Y
plt.yticks(tick_positions, [f"Cluster {i}" for i in range(optimal_k)])

# Linea del Silhouette Score Medio
plt.axvline(np.mean(silhouette_vals), color="red", linestyle="--", label="Silhouette Score Medio")

# Titoli e legende
plt.title("Silhouette Plot con Bande Separate per Cluster")
plt.xlabel("Valore del Silhouette Coefficient")
plt.ylabel("Cluster")
plt.legend()
plt.show()
