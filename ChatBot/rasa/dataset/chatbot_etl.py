import pandas as pd
import ast

df = pd.read_csv("TMDB_movie_dataset_v11.csv")

df_clean = df[['id', 'title', 'vote_average', 'status', 'release_date', 'revenue', 'runtime', 'adult', 'original_language', 
               'original_title', 'overview', 'genres', 'production_companies', 'production_countries', 'spoken_languages',
                'keywords']]

df_clean = df_clean.drop_duplicates()

df_clean = df_clean.dropna(subset=['title'])

df_clean['release_date'] = pd.to_datetime(df_clean['release_date'])

# def clean_list_column(value):
#     if isinstance(value, str):
#         try:
#             return ', '.join(ast.literal_eval(value))  # Converte stringhe di liste in testo
#         except:
#             return value  # Se già in formato corretto, lascia invariato
#     return ""

# df_clean['genres'] = df_clean['genres'].apply(clean_list_column)
# df_clean['production_companies'] = df_clean['production_companies'].apply(clean_list_column)
# df_clean['production_countries'] = df_clean['production_countries'].apply(clean_list_column)
# df_clean['spoken_languages'] = df_clean['spoken_languages'].apply(clean_list_column)
# df_clean['keywords'] = df_clean['keywords'].apply(clean_list_column)

df_clean['title'] = df_clean['title'].astype(str)
df_clean['status'] = df_clean['status'].astype(str)
df_clean['original_language'] = df_clean['original_language'].astype(str)
df_clean['original_title'] = df_clean['original_title'].astype(str)
df_clean['overview'] = df_clean['overview'].astype(str)

text_columns = ['title', 'status', 'original_language', 'original_title', 'overview', 'genres', 
                'production_companies', 'production_countries', 'spoken_languages', 'keywords']
df_clean[text_columns] = df_clean[text_columns].fillna("")

df_clean.to_csv("tmdb_movies.csv", index=False)

df_clean.info()