from deep_translator import GoogleTranslator

def translate_to_italian(text):
    """Traduce un testo dall'inglese all'italiano."""
    try:
        translator = GoogleTranslator(source='en', target='it')
        return translator.translate(text)
    except Exception as e:
        return f"Errore di traduzione: {e}"

# Mappatura statica per la traduzione dei generi
genre_translation = {
    "Action": "Azione",
    "Adventure": "Avventura",
    "Animation": "Animazione",
    "Comedy": "Commedia",
    "Crime": "Crimine",
    "Documentary": "Documentario",
    "Drama": "Drammatico",
    "Family": "Famiglia",
    "Fantasy": "Fantasy",
    "History": "Storico",
    "Horror": "Horror",
    "Music": "Musica",
    "Mystery": "Mistero",
    "Romance": "Romantico",
    "Science Fiction": "Fantascienza",
    "TV Movie": "Film TV",
    "Thriller": "Thriller",
    "War": "Guerra",
    "Western": "Western"
}

def translate_genre(genre_list):
    genres = genre_list.split(',')
    return ', '.join([genre_translation.get(g.strip(), g.strip()) for g in genres])