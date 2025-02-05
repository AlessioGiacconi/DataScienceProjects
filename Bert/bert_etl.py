import pandas as pd
import re
from pathlib import Path
from nltk.corpus import stopwords
import nltk


base_dir = Path(__file__).resolve().parent

file_path = base_dir / 'Dataset' / 'Suicide_Dataset.csv'

df_csv = pd.read_csv(file_path)
df_csv.info()

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
important_terms = {"myself", "yourself", "me", "i", "my", "mine"}
stop_words = stop_words - important_terms

# Funzione per rimuovere le emoji
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # simboli e pittogrammi
        "\U0001F680-\U0001F6FF"  # trasporti e simboli mappa
        "\U0001F1E0-\U0001F1FF"  # bandiere
        "\U00002500-\U00002BEF"  # simboli vari
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)


# Funzione per pulire i tweet, rimuovendo la punteggiatura escluso '#'
def clean_tweet(text):
    if isinstance(text, str):
        # Converti tutto in minuscolo
        text = text.lower()
        # Normalizzazione apostrofi
        text = text.replace('’', "'").replace('‘', "'").replace('`', "'")
        # Rimuovi tag HTML
        text = re.sub(r'<.*?>', '', text)
        # Rimuove gli URL
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Rimuovi emoji
        text = remove_emojis(text)
        # Rimuove la punteggiatura mantenendo solo '#'
        cleaned_text = ''.join(char for char in text if char.isalpha() or char.isspace() or char == '#')
        # Filtra e rimuovi le stopwords
        filtered_words = [word for word in cleaned_text.split() if word not in stop_words]
        return ' '.join(filtered_words).strip()
    return text  # Se non è una stringa, ritorna il valore originale

# Applica la pulizia ai tweet
df_csv['Cleaned_Tweet'] = df_csv['Tweet'].apply(clean_tweet)

# Dataset ripulito
cleaned_df = df_csv[['Cleaned_Tweet', 'Suicide']].to_csv(base_dir / 'Dataset' / 'cleaned_tweets.csv', index=False)


