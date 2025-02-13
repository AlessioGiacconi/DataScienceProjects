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

df = pd.read_csv("./dataset/tmdb_movies.csv")

class ActionCercaFilm(Action):
    def name(self):
        return "action_cerca_film"

    def run(self, dispatcher, tracker, domain):
        titolo = tracker.get_slot("title")

        if titolo:
            film = df[df["title"].str.lower() == titolo.lower()]
            if not film.empty:
                risposta = f"🎬 {film.iloc[0]['title']} è uscito il {film.iloc[0]['release_date']}."
                dispatcher.utter_message(risposta)
            else:
                dispatcher.utter_message("Non ho trovato questo film 😢")
        else:
            dispatcher.utter_message("Dimmi il titolo del film che cerchi 🎬")

        return []

