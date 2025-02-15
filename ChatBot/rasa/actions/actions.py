# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from actions.utils import translate_to_italian, translate_genre, translate_genre_ita_to_eng, translate_language_ita_to_iso

df = pd.read_csv("./dataset/tmdb_movies.csv").fillna("Dato non disponibile")

class ActionCercaFilm(Action):
    def name(self):
        return "action_cerca_film"

    def run(self, dispatcher, tracker, domain):
        titolo = tracker.get_slot("title")
        genere = tracker.get_slot("genres")

        if titolo:
            film = df[df["title"].str.lower() == titolo.lower()]
        elif genere:
            film = df[df["genres"].str.lower().str.contains(genere.lower())]
        else:
            dispatcher.utter_message("Per favore, dimmi un titolo di un film. 🎬")
            return []

        if not film.empty:
            film_info = film.iloc[0]
            overview_it = translate_to_italian(film_info['overview'])
            genres_it = translate_genre(str(film_info['genres']))
            runtime = film.iloc[0]['runtime']
            ore = int(runtime // 60)
            minuti = int(runtime % 60)

            risposta = (
                f"🎬 {film_info['title']} è uscito il {film_info['release_date']}."
                f"\nGenere: {genres_it}"
                f"\nDurata: {ore} ore e {minuti} minuti"
                f"\nTrama: {overview_it}"
            )
            dispatcher.utter_message(risposta)
        else:
            dispatcher.utter_message("Non ho trovato film corrispondenti 😢")

        return []

class ActionCercaPerGenere(Action):
    def name(self):
        return "action_cerca_per_genere"

    def run(self, dispatcher, tracker, domain):
        genere_ita = tracker.get_slot("genres")  # Slot del genere in italiano

        if genere_ita:
            genere_eng = translate_genre_ita_to_eng(genere_ita)  # Traduciamo il genere

            # Filtriamo i film con il genere richiesto
            film_trovati = df[df["genres"].apply(lambda x: genere_eng.lower() in [g.strip().lower() for g in x.split(',')])]

            if not film_trovati.empty:
                film_list = film_trovati["title"].tolist()[:5]
                dispatcher.utter_message(f"Ecco alcuni film di genere {genere_ita}:\n" + "\n".join(film_list))
            else:
                dispatcher.utter_message(f"Non ho trovato film di genere {genere_ita}. 😢")
        else:
            dispatcher.utter_message("Per favore, dimmi un genere per cercare film. 🎬")

        return []

class ActionCercaPerRating(Action):
    def name(self):
        return "action_cerca_per_rating"

    def run(self, dispatcher, tracker, domain):
        rating = tracker.get_slot("rating")

        if rating:
            rating = float(rating)
            film_trovati = df[df["vote_average"] >= rating]
            if not film_trovati.empty:
                film_list = film_trovati["title"].tolist()[:5]
                dispatcher.utter_message(f"Ecco alcuni film con voto superiore a {rating}:\n" + "\n".join(film_list))
            else:
                dispatcher.utter_message(f"Non ho trovato film con un voto superiore a {rating}.")
        else:
            dispatcher.utter_message("Per favore, dimmi una valutazione per cercare film.")

        return []
    

class ActionCercaPerLingua(Action):
    def name(self):
        return "action_cerca_per_lingua"

    def run(self, dispatcher, tracker, domain):
        lingua_ita = tracker.get_slot("language")  # Slot della lingua in italiano

        if lingua_ita:
            lingua_eng = translate_language_ita_to_iso(lingua_ita)

            # Filtra i film con la lingua richiesta
            film_trovati = df[df["original_language"].str.lower() == lingua_eng.lower()]

            if not film_trovati.empty:
                film_list = [f"{row['original_title']} (titolo in inglese: {row['title']})" 
                                if row['title'].lower() != row['original_title'].lower() 
                                else row['original_title'] for _, row in film_trovati.iterrows()][:5]
                dispatcher.utter_message(f"Ecco alcuni film in {lingua_ita}:")
                dispatcher.utter_message("\n".join(film_list))
            else:
                dispatcher.utter_message(f"Non ho trovato film in {lingua_ita}. 😢")
        else:
            dispatcher.utter_message("Per favore, dimmi una lingua per cercare film. 🎬")

        return []

class ActionCercaFilmRandom(Action):
    def name(self):
        return "action_cerca_film_random"

    def run(self, dispatcher, tracker, domain):
        if not df.empty:
            film_random = df.sample(n=1).iloc[0]
            overview_it = translate_to_italian(film_random['overview'])
            genres_it = translate_genre(str(film_random['genres']))
            risposta = (
                f"🎲 Ecco un film randomico scelto per te!\n"
                f"🎬 {film_random['title']} è uscito il {film_random['release_date']}.\n"
                f"Genere: {genres_it}\n"
                f"Trama: {overview_it}"
            )
            dispatcher.utter_message(risposta)
        else:
            dispatcher.utter_message("Non ho trovato film disponibili al momento 😢")

        return []

class ActionCercaPerDurata(Action):
    def name(self):
        return "action_cerca_per_durata"

    def run(self, dispatcher, tracker, domain):
        durata_min = tracker.get_slot("runtime")

        if durata_min:
            film_trovati = df[df['runtime'] <= float(durata_min)]

            if not film_trovati.empty:
                film_list = [f"{row['title']} ({int(row['runtime']//60)}h {int(row['runtime']%60)}m)" for _, row in film_trovati.iterrows()][:5]
                dispatcher.utter_message(f"Ecco alcuni film con durata inferiore a {durata_min} minuti:\n" + "\n".join(film_list))
            else:
                dispatcher.utter_message(f"Non ho trovato film con durata inferiore a {durata_min} minuti. 😢")
        else:
            dispatcher.utter_message("Dimmi una durata massima per cercare film 🎬")

        return []

class ActionCercaFilmRecenti(Action):
    def name(self):
        return "action_cerca_film_recenti"

    def run(self, dispatcher, tracker, domain):
        # Calcola la data di 3 mesi fa
        data_massima = datetime.strptime("2025-02-12", "%Y-%m-%d") # Calcola la data di 3 mesi fa        
        data_limite = data_massima - timedelta(days=90)

        # Filtra i film usciti negli ultimi 3 mesi e ordina per voto decrescente
        film_recenti = df[(pd.to_datetime(df['release_date'], errors='coerce') >= data_limite)].sort_values(by='vote_average', ascending=False).head(10)

        if not film_recenti.empty:
            film_list = [f"{row['title']} (⭐ {row['vote_average']}, uscita: {row['release_date']})" for _, row in film_recenti.iterrows()]
            dispatcher.utter_message("🎞️ Ecco i 10 film più recenti e famosi degli ultimi 3 mesi:\n" + "\n".join(film_list))
        else:
            dispatcher.utter_message("Non ho trovato film recenti negli ultimi 3 mesi 😢")

        return []
    

class ActionMostraOverview(Action):
    def name(self):
        return "action_mostra_overview"

    def run(self, dispatcher, tracker, domain):
        # Ottieni il titolo dallo slot (ultimo titolo cercato)
        titolo_slot = tracker.get_slot("title")

        # Controlla se l'utente ha fornito un titolo nel messaggio
        titolo_messaggio = next(tracker.get_latest_entity_values("title"), None)

        # Se l'utente ha fornito un nuovo titolo, lo usiamo
        if titolo_messaggio:
            titolo_film = titolo_messaggio
        else:
            # Se l'utente NON ha fornito un nuovo titolo, usiamo l'ultimo memorizzato
            titolo_film = titolo_slot

        # Se il titolo è vuoto o errato, chiediamo di specificarlo
        if not titolo_film or titolo_film.lower() in ["di cosa parla questo film?", "raccontami la trama", "qual è la sinossi?"]:
            dispatcher.utter_message("Non so a quale film ti riferisci. Puoi dirmi il titolo? 🎬")
            return []

        # Cerchiamo il film nel dataset
        film = df[df["title"].str.lower() == titolo_film.lower()]

        if not film.empty:
            overview = film.iloc[0]["overview"]  # Otteniamo la trama
            dispatcher.utter_message(f"Ecco la trama di **{titolo_film}**:\n\n_{overview}_")
        else:
            dispatcher.utter_message(f"Non ho trovato la trama di **{titolo_film}**. Assicurati che il titolo sia corretto! 😢")

        return []