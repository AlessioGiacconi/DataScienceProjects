import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset_path: str = '../DataScienceProjects/SerieTemporali/Dataset/urban_mobility_dataset.csv'
raw_dataset = pd.DataFrame(pd.read_csv(dataset_path))
description = raw_dataset.describe()

raw_dataset['timestamp'] = pd.to_datetime(raw_dataset['timestamp'])

raw_dataset["hour"] = raw_dataset["timestamp"].dt.hour
raw_dataset["day_of_week"] = raw_dataset["timestamp"].dt.day_name()
raw_dataset["month"] = raw_dataset["timestamp"].dt.month
raw_dataset['year'] = raw_dataset['timestamp'].dt.year
raw_dataset['weekday_type'] = raw_dataset['day_of_week'].apply(lambda x: 'weekend' if x in ['Saturday', 'Sunday'] else 'feriale')

data_2023 = raw_dataset[raw_dataset['year'] == 2023]
pass
weekday_analysis = data_2023.groupby('weekday_type')[['public_transport_usage', 'bike_sharing_usage']].mean()


# Visualizzare i risultati tramite un grafico a barre
plt.figure(figsize=(10, 6))
weekday_analysis.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange'])
plt.title('Utilizzo di trasporti pubblici e bike-sharing: feriali vs weekend')
plt.ylabel('Numero medio di utenti')
plt.xlabel('Tipo di giorno')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Tipologia di utilizzo')
plt.tight_layout()
plt.show()

"""
data_2023 = raw_dataset[raw_dataset['year'] == 2023]

# Raggruppare i dati del 2023 per mese e calcolare la media
data_2023_monthly_trends = data_2023.groupby('month')[['public_transport_usage', 'bike_sharing_usage']].mean()

# Visualizzare le tendenze mensili per l'anno 2023
plt.figure(figsize=(12, 6))
data_2023_monthly_trends.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('Tendenze mensili di utilizzo dei trasporti pubblici e bike-sharing (Anno 2023)')
plt.xlabel('Mese')
plt.ylabel('Numero medio di utenti')
plt.xticks(ticks=range(1, 13), labels=['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Tipologia di utilizzo')
plt.tight_layout()
plt.show()

data_2024 = raw_dataset[raw_dataset['year'] == 2024]

data_2024_monthly_trends = data_2024.groupby('month')[['public_transport_usage', 'bike_sharing_usage']].mean()

plt.figure(figsize=(12, 6))
data_2024_monthly_trends.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('Tendenze mensili di utilizzo dei trasporti pubblici e bike-sharing (Anno 2024)')
plt.xlabel('Mese')
plt.ylabel('Numero medio di utenti')
plt.xticks(ticks=range(1, 13), labels=['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Tipologia di utilizzo')
plt.tight_layout()
plt.show()
"""


