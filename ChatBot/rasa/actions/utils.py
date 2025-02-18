from deep_translator import GoogleTranslator

def translate_to_italian(text):
    try:
        if len(text) > 500:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            translator = GoogleTranslator(source='en', target='it')
            translated_chunks = [translator.translate(chunk) for chunk in chunks]
            return ' '.join(translated_chunks)
        else:
            return GoogleTranslator(source='en', target='it').translate(text)
    except Exception as e:
        return f"Errore di traduzione: {e}"

# Mappatura statica per la traduzione dei generi
genre_translation = {
    "azione": "Action",
    "avventura": "Adventure",
    "animazione": "Animation",
    "commedia": "Comedy",
    "crimine": "Crime",
    "documentario": "Documentary",
    "drammatico": "Drama",
    "famiglia": "Family",
    "fantasy": "Fantasy",
    "storico": "History",
    "horror": "Horror",
    "musica": "Music",
    "mistero": "Mystery",
    "romantico": "Romance",
    "fantascienza": "Science Fiction",
    "film tv": "TV Movie",
    "thriller": "Thriller",
    "guerra": "War",
    "western": "Western"
}

# Funzione per tradurre i codici lingua (IT → EN)
language_translation = {
    "inglese": "en",
    "italiano": "it",
    "francese": "fr",
    "spagnolo": "es",
    "tedesco": "de",
    "giapponese": "ja",
    "cinese": "zh",
    "russo": "ru",
    "portoghese": "pt",
    "olandese": "nl",
    "arabo": "ar",
    "coreano": "ko",
    "hindi": "hi"
}

# Funzione per tradurre la lingua dall'italiano al codice ISO
def translate_language_ita_to_iso(language):
    return language_translation.get(language.lower().strip(), language)

#Creiamo la mappatura inversa (IT → EN)
reverse_genre_translation = {v: k for k, v in genre_translation.items()}

def translate_genre(genre_list):
    genres = genre_list.split(',')
    return ', '.join([reverse_genre_translation.get(g.strip(), g.strip()) for g in genres])

#Funzione per tradurre un genere dall'italiano all'inglese
def translate_genre_ita_to_eng(genre):
    return genre_translation.get(genre.lower().strip(), genre)