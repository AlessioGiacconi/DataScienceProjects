import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

base_dir = Path(__file__).resolve().parent

file_path = base_dir / 'Dataset' / 'cleaned_tweets.csv'

df_csv = pd.read_csv(file_path)
df_csv.info()

# Suddividi il dataset in training set (80%) e test set (20%)
train_set, test_set = train_test_split(df_csv, test_size=0.2, random_state=42, shuffle=True)



