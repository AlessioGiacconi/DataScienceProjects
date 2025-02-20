import pandas as pd

df = pd.read_csv("TMDB_movie_dataset_v11.csv")

df_clean = df[['id', 'title', 'vote_average', 'status', 'release_date', 'revenue', 'runtime', 'adult', 'original_language', 
               'original_title', 'overview', 'poster_path', 'genres', 'production_companies', 'production_countries', 'spoken_languages',
                'keywords']]

df_clean = df_clean.drop_duplicates()

df_clean = df_clean.dropna(subset=['title'])

df_clean['release_date'] = pd.to_datetime(df_clean['release_date'])

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