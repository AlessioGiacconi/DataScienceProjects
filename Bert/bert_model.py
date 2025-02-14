import pandas as pd
from pathlib import Path



base_dir = Path(__file__).resolve().parent

file_path = base_dir / 'Dataset' / 'cleaned_tweets.csv'

df_csv = pd.read_csv(file_path)
df_csv.info()

label_mapping = {
    "Not Suicide post": 0,
    "Potential Suicide post ": 1
}

df_csv['Suicide'] = df_csv['Suicide'].map(label_mapping)







