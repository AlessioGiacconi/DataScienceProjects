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
from actions.utils import translate_to_italian, translate_genre

df = pd.read_csv("./dataset/tmdb_movies.csv")

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
        genere = tracker.get_slot("genres")

        if genere:
            genere_eng = translate_genre(genere)
            film = df[df["genres"].fillna("").str.lower().apply(lambda x: any(g.strip() == genere_eng.lower() for g in x.split(',')))]
            if not film.empty:
                film_list = film["title"].tolist()[:5]
                film_list_it = [translate_to_italian(title) for title in film_list]
                dispatcher.utter_message(f"Ecco alcuni film di genere {genere}: \n" + "\n".join(film_list_it))
            else:
                dispatcher.utter_message(f"Non ho trovato film di genere {genere}. 😢")
        else:
            dispatcher.utter_message("Per favore, dimmi un genere per cercare film. 🎬")

        return []