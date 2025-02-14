# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
import pandas as pd
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

            risposta = f"🎬 {film_info['title']} è uscito il {film_info['release_date']}.\nGenere: {genres_it}\nTrama: {overview_it}"
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
                film_list = film_trovati["title"].tolist()[:5]
                dispatcher.utter_message(f"Ecco alcuni film in {lingua_ita}:\n" + "\n".join(film_list))
            else:
                dispatcher.utter_message(f"Non ho trovato film in {lingua_ita}. 😢")
        else:
            dispatcher.utter_message("Per favore, dimmi una lingua per cercare film. 🎬")

        return []

