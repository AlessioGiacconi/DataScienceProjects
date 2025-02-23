# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from actions.utils import translate_to_italian, translate_genre, translate_genre_ita_to_eng, translate_language_ita_to_iso
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import SlotSet, AllSlotsReset

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
            runtime = film_info['runtime']
            ore = int(runtime // 60)
            minuti = int(runtime % 60)

            # Creazione del link dell'immagine
            poster_path = film_info['poster_path']
            image_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if pd.notna(poster_path) else None

            # Estrazione e formattazione informazioni aggiuntive
            status = translate_to_italian(film_info['status'])
            revenue = f"{film_info['revenue']:,}".replace(',', '.') + " USD"
            original_lang = translate_to_italian(film_info['original_language'])
            original_title = translate_to_italian(film_info['original_title'])
            companies = translate_to_italian(film_info['production_companies'])
            countries = translate_to_italian(film_info['production_countries'])
            spoken_languages = translate_to_italian(film_info['spoken_languages'])
            vote_avg = film_info['vote_average']

            risposta = (
                f"🎬 Titolo: {film_info['title']} ({original_title})",
                f"\n 🌍 Lingua originale: {original_lang}",
                f"\n 📅 Data di uscita: {film_info['release_date']}",
                f"\n 📝 Stato: {status}",
                f"\n 🎯 Voto medio: {vote_avg}/10",
                f"\n 💵 Incasso: {revenue}",
                f"\n 🕒 Durata: {ore} ore e {minuti} minuti",
                f"\n 🎞️ Genere: {genres_it}",
                f"\n 🏢 Case di produzione: {companies}",
                f"\n 🌎 Paesi di produzione: {countries}",
                f"\n 🗣️ Lingue parlate: {spoken_languages}",
                f"\n 📖 Trama: {overview_it}"
            )
            dispatcher.utter_message(" ".join(risposta))

            # Invia l'immagine se disponibile
            if image_url:
                dispatcher.utter_message(image=image_url)
                
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
        rating = tracker.get_slot("vote_average")

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
                f"🎞️ Genere: {genres_it}\n"
                f"📖 Trama: {overview_it}"
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
            # Pulizia dei dati: rimuove NaN e converte in float
            df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(9999)

            # Filtra i film con durata tra 60 minuti e il valore massimo richiesto
            film_trovati = df[(df['runtime'] >= 50) & (df['runtime'] <= float(durata_min))]

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
            overview_en = film.iloc[0]["overview"]  # Otteniamo la trama
            overview_it = translate_to_italian(overview_en)
            dispatcher.utter_message(f"Ecco la trama di **{titolo_film}**:\n\n{overview_it}")
        else:
            dispatcher.utter_message(f"Non ho trovato la trama di **{titolo_film}**. Assicurati che il titolo sia corretto! 😢")

        return []

'''class ActionCercaFilmCombinato(Action):
    def name(self):
        return "action_cerca_film_combinato"

    def run(self, dispatcher, tracker, domain):
        # Recupera i parametri dai rispettivi slot
        genere = tracker.get_slot("genres")
        durata_max = tracker.get_slot("runtime")
        lingua = tracker.get_slot("language")
        voto_min = tracker.get_slot("rating")
        data_min = tracker.get_slot("release_date")

        # Applica i filtri in base ai parametri forniti
        film_filtrati = df

        # Filtro per genere
        if genere:
            genere_eng = translate_genre_ita_to_eng(genere)
            film_filtrati = film_filtrati[film_filtrati["genres"].str.contains(genere_eng, case=False, na=False)]

        # Filtro per durata massima
        if durata_max:
            film_filtrati = film_filtrati[film_filtrati["runtime"] <= float(durata_max)]

        # Filtro per lingua
        if lingua:
            lingua_eng = translate_language_ita_to_iso(lingua)
            film_filtrati = film_filtrati[film_filtrati["original_language"] == lingua_eng]

        # Filtro per voto minimo
        if voto_min:
            film_filtrati = film_filtrati[film_filtrati["vote_average"] >= float(voto_min)]

        # Filtro per data di rilascio minima
        if data_min:
            try:
                data_min = datetime.strptime(data_min, "%Y-%m-%d")
                film_filtrati["release_date"] = pd.to_datetime(film_filtrati["release_date"], errors="coerce")
                film_filtrati = film_filtrati[film_filtrati["release_date"] >= data_min]
            except Exception as e:
                dispatcher.utter_message(f"Errore nella gestione della data: {e}")
                return []

        # Mostra i risultati
        if not film_filtrati.empty:
            film_list = film_filtrati.sort_values(by="vote_average", ascending=False).head(5)
            messaggio = "🎬 Ecco i film che corrispondono ai criteri richiesti:\n"
            for _, row in film_list.iterrows():
                overview_it = translate_to_italian(row["overview"])
                messaggio += (
                    f"\n🎞️ {row['title']}\n"
                    f"⭐ Voto medio: {row['vote_average']}/10\n"
                    f"🕒 Durata: {int(row['runtime']//60)}h {int(row['runtime']%60)}m\n"
                    f"🗓️ Rilasciato il: {row['release_date']}\n"
                    f"📝 Trama: {overview_it}\n"
                    f"---------------------------------\n"
                )
            dispatcher.utter_message(messaggio)
        else:
            dispatcher.utter_message("Non ho trovato film che corrispondano ai criteri richiesti 😢")

        return []'''

    
class ActionCercaFilmSimile(Action):
    def name(self):
        return "action_cerca_film_simile"

    def run(self, dispatcher, tracker, domain):
        titolo = tracker.get_slot("title")

        if not titolo:
            dispatcher.utter_message("Per favore, dimmi il titolo di un film per cercarne di simili. 🎬")
            return []

        # Cerchiamo il film nel dataset
        film_base = df[df["title"].str.lower() == titolo.lower()]
        if film_base.empty:
            dispatcher.utter_message(f"Non ho trovato il film '{titolo}'. Assicurati che il titolo sia corretto! 😢")
            return []

        # Estraiamo informazioni del film di riferimento
        film_info = film_base.iloc[0]
        keywords_base = set(str(film_info['keywords']).lower().split(', ')) if pd.notna(film_info['keywords']) else set()
        generi_base = set(str(film_info['genres']).lower().split(', ')) if pd.notna(film_info['genres']) else set()

        # Calcolo vettorializzato del punteggio di somiglianza
        def calcola_punteggio(row):
            kw = set(str(row['keywords']).lower().split(', ')) if pd.notna(row['keywords']) else set()
            gn = set(str(row['genres']).lower().split(', ')) if pd.notna(row['genres']) else set()
            return 2 * len(generi_base.intersection(gn)) + len(keywords_base.intersection(kw))

        # Calcoliamo i punteggi di somiglianza in modo vettorializzato
        df['match_score'] = df.apply(calcola_punteggio, axis=1)

        # Filtriamo i film simili (escludendo quello cercato) e otteniamo i primi 5
        film_simili = df[df['title'].str.lower() != titolo.lower()].nlargest(5, ['match_score', 'vote_average'])

        if film_simili.empty:
            dispatcher.utter_message("Non sono riuscito a trovare film simili. Prova con un altro titolo! 🎬")
            return []

        # Creiamo il messaggio di risposta
        messaggi = [f"🎬 Ecco alcuni film simili a '{titolo}':\n"]
        for _, row in film_simili.iterrows():
            genres_it = translate_genre(str(row["genres"]))
            overview_it = translate_to_italian(row["overview"]) if row["overview"] else "Trama non disponibile"
            data_rilascio = pd.to_datetime(row["release_date"], errors='coerce').strftime('%Y-%m-%d') if pd.notna(row["release_date"]) else "Data non disponibile"

            messaggi.append(
                f"🎞️ {row['title']} ({data_rilascio})\n"
                f"⭐ Voto medio: {row['vote_average']}/10\n"
                f"🎞️ Genere: {genres_it}\n"
                f"📖 Trama: {overview_it}\n"
                f"---------------------------------\n"
            )

        dispatcher.utter_message("\n".join(messaggi))
        return []

class ActionSubmitFilmCombinato(Action):
    def name(self):
        return "action_submit_film_combinato"

    def run(self, dispatcher, tracker, domain):
        genere = tracker.get_slot("genres")
        durata_max = tracker.get_slot("runtime")
        lingua = tracker.get_slot("language")
        voto_min = tracker.get_slot("vote_average")
        data_min = tracker.get_slot("release_date")

        film_filtrati = df

        if genere:
            genere_eng = translate_genre_ita_to_eng(genere)
            film_filtrati = film_filtrati[film_filtrati["genres"].str.contains(genere_eng, case=False, na=False)]
        
        if durata_max:
            film_filtrati = film_filtrati[(film_filtrati["runtime"] >= 50) & (film_filtrati["runtime"] <= float(durata_max))]

        if lingua:
            lingua_eng = translate_language_ita_to_iso(lingua)
            film_filtrati = film_filtrati[film_filtrati["original_language"] == lingua_eng]

        if voto_min:
            film_filtrati = film_filtrati[film_filtrati["vote_average"] >= float(voto_min)]

        if data_min:
            try:
                data_min = datetime.strptime(data_min, "%Y-%m-%d")
                film_filtrati["release_date"] = pd.to_datetime(film_filtrati["release_date"], errors="coerce")
                film_filtrati = film_filtrati[film_filtrati["release_date"] >= data_min]
            except Exception as e:
                dispatcher.utter_message(f"Errore nella gestione della data: {e}")
                return []

        if not film_filtrati.empty:
            film_list = film_filtrati.sort_values(by="vote_average", ascending=False).head(5)
            messaggio = "🎬 Ecco i film che corrispondono ai criteri richiesti:\n"
            for _, row in film_list.iterrows():
                overview_it = translate_to_italian(row["overview"])
                messaggio += (
                    f"\n🎞️ {row['title']}\n"
                    f"⭐ Voto medio: {row['vote_average']}/10\n"
                    f"🕒 Durata: {int(row['runtime']//60)}h {int(row['runtime']%60)}m\n"
                    f"🗓️ Rilasciato il: {row['release_date']}\n"
                    f"📝 Trama: {overview_it}\n"
                    f"---------------------------------\n"
                )
            dispatcher.utter_message(messaggio)
        else:
            dispatcher.utter_message("Non ho trovato film che corrispondano ai criteri richiesti 😢")

        # 🔥 Resetta gli slot alla fine della ricerca
        return [
            SlotSet("genres", None),
            SlotSet("runtime", None),
            SlotSet("language", None),
            SlotSet("vote_average", None),
            SlotSet("release_date", None)
        ]

class ValidateFilmCombinatoForm(FormValidationAction):
    def name(self) -> str:
        return "validate_movie_search_form"

    def validate_genres(self, slot_value, dispatcher, tracker, domain):
        """Valida il genere del film. Se l'utente dice 'no', lo slot viene ignorato."""
        if slot_value and slot_value.lower() in ["no", "nessuna preferenza", "non importa"]:
            dispatcher.utter_message("Va bene, non considererò il genere! 🎭")
            return {"genres": None}
        return {"genres": slot_value}

    def validate_runtime(self, slot_value, dispatcher, tracker, domain):
        """Valida la durata del film. Se l'utente dice 'no', lo slot viene ignorato."""
        if slot_value is None or isinstance(slot_value, str) and slot_value.lower() in ["no", "nessuna preferenza", "non importa"]:
            dispatcher.utter_message("Va bene, ignorerò la durata! ⏳")
            return {"runtime": None}
        try:
            runtime_value = int(slot_value)
            return {"runtime": runtime_value}
        except (ValueError, TypeError):
            dispatcher.utter_message("Per favore, inserisci un numero valido per la durata.")
            return {"runtime": None}

    def validate_language(self, slot_value, dispatcher, tracker, domain):
        """Valida la lingua del film. Se l'utente dice 'no', lo slot viene ignorato."""
        if slot_value is None or isinstance(slot_value, str) and slot_value.lower() in ["no", "nessuna preferenza", "qualsiasi lingua", "non importa"]:
            dispatcher.utter_message("Va bene, ignorerò la lingua! 😊")
            return {"language": None}
        return {"language": slot_value}

    def validate_vote_average(self, slot_value, dispatcher, tracker, domain):
        """Valida il voto minimo. Se l'utente dice 'no', lo slot viene ignorato."""
        if slot_value is None or isinstance(slot_value, str) and slot_value.lower() in ["no", "nessuna preferenza", "non importa"]:
            dispatcher.utter_message("Va bene, non terrò conto del voto! ⭐")
            return {"vote_average": None}
        try:
            vote_value = float(slot_value)
            if 0 <= vote_value <= 10:
                return {"vote_average": vote_value}
            else:
                dispatcher.utter_message("Il voto deve essere tra 0 e 10. Riprova! 🎥")
                return {"vote_average": None}
        except (ValueError, TypeError):
            dispatcher.utter_message("Per favore, inserisci un numero valido per il voto.")
            return {"vote_average": None}

