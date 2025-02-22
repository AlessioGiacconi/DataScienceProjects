import pandas as pd

# Carica il dataset
df = pd.read_csv("TMDB_movie_dataset_v11.csv")

# Selezione delle colonne di interesse
df_clean = df[['id', 'title', 'vote_average', 'vote_count', 'status', 'release_date', 'revenue', 'runtime', 'adult', 
               'original_language', 'original_title', 'overview', 'poster_path', 'genres', 
               'production_companies', 'production_countries', 'spoken_languages', 'keywords']]

# Rimozione duplicati e film senza titolo
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.dropna(subset=['title'])

# Conversione della data
df_clean['release_date'] = pd.to_datetime(df_clean['release_date'], errors='coerce')

# Conversione dei tipi di dato per le colonne di testo
text_columns = ['title', 'status', 'original_language', 'original_title', 'overview', 'genres', 
                'production_companies', 'production_countries', 'spoken_languages', 'keywords']
df_clean[text_columns] = df_clean[text_columns].fillna("").astype(str)

# **Filtraggio dei film con valutazioni più affidabili**
SOGLIA_VOTE_COUNT = 50  # Minimo numero di recensioni
SOGLIA_VOTE_AVG_ALTA = 9  # Soglia per film con rating sospetto
SOGLIA_VOTE_COUNT_ALTA = 100  # Minimo voto_count per film con rating molto alto

df_clean = df_clean[df_clean["vote_count"] >= SOGLIA_VOTE_COUNT]
df_clean = df_clean[~((df_clean["vote_average"] >= SOGLIA_VOTE_AVG_ALTA) & 
                      (df_clean["vote_count"] < SOGLIA_VOTE_COUNT_ALTA))]

# **Calcolo del Voto Ponderato**
media_globale = df_clean["vote_average"].mean()
m = df_clean["vote_count"].quantile(0.90)  # Top 10% dei film per numero di voti

def weighted_rating(row, m=m, C=media_globale):
    v = row["vote_count"]
    R = row["vote_average"]
    return (v / (v + m) * R) + (m / (m + v) * C)

df_clean["weighted_vote"] = df_clean.apply(weighted_rating, axis=1)
df_clean = df_clean.drop(columns=['vote_count', 'weighted_vote'])

# Salvataggio del dataset pulito
df_clean.to_csv("tmdb_movies.csv", index=False)

# Stampa informazioni finali
print(f"Dataset ridotto da {len(df)} a {len(df_clean)} film dopo la pulizia.")
df_clean.info()