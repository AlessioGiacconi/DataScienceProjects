# Progetti Data Science A.A. 2024/2025

## 🤖 Progetto 4: ChatBot - CineGuru AI

### Prerequisiti

- Installare Conda (sia versione Miniconda che Anaconda vanno bene).

### Step di preparazione

1. Clone della repository

```bash
git clone https://github.com/AlessioGiacconi/DataScienceProjects
cd DataScienceProjects/ChatBot
```

2. Creazione e attivazione ambiente Conda

```bash
conda create --name chatbot_env python=3.9  # Basta eseguirlo una volta sola, dopodiché solo activate
conda activate chatbot_env
```

3. Installazione dipendenze

```bash
pip install -r requirements.txt
```

### Train del modello

```bash
cd rasa
rasa train
```

### Run del server

```bash
rasa run actions
```

### Start del ChatBot

In un altro terminale:

```bash
rasa shell
```

### Connessione del ChatBot a Telegram

1. Creazione del Bot Telegram

    - Aprire Telegram e cercare il **BotFather**,
    - Scrivere ```\newbot``` e seguire le istruzioni per la creazione del bot.
    - BotFather fornirà un TOKEN, necessario per il collegamento a Rasa.

2. Configurare Telegram in RASA

    Aprire il file ```credentials.yml``` del progetto Rasa e modificare la configurazione per Telegram.

    ```yaml
    telegram:
        access_token: "TOKEN"
        verify: "USERNAME_BOT"  # username del bot senza @
        webhook_url: "https://<NGROK_URL>/webhooks/telegram/webhook"
    ```
    > Sostituire i valori placeholder, rispettivamente, con l'access token di BotFather, l'username inserito durante la creazione del Bot e l'URL fornita da ngrok (vedere Passaggio 3)

3. Avviare Ngrok

    - (Se non si ha già) Installare, configurare ed estrarre [ngrok](https://ngrok.com/download) all'interno della sua cartella di lavoro.
    - Avviare ngrok con la porta di default Rasa:
        ```bash
        ngrok http 5005
        ```
    - L'output fornirà un URL HTTPS, da inserire nel file ```credentials.yml```, come mostrato prima.

4. Avviare Rasa

    Avviare il server Rasa con:

    ```bash
    rasa run --enable-api
    ```

    E abilitare le azioni perrsonalizzate da un altro terminale:

    ```bash
    rasa run actions
    ```

5. Impostare il Webhooh di Telegram

    - Comunicare a Telegram il nuovo URL del webhook da linea di comando (o sulla barra di ricerca passando solo l'URL intera):
        ```bash
        curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://<NGROK_URL>/webhooks/telegram/webhook"
        ```
      Sostituendo con i valori corretti.
    - Se tutto è andato correttamente, si otterrà una risposta simile a questa:
        ```json
        {
            "ok": true,
            "result": true,
            "description": "Webhook was set"
        }
        ```

6. Ora è possibile aprire Telegram, cercare il bot e iniziare a inviare messaggi, per verificare la correttezza delle risposte.

## 👥 Autori 

|Nome | GitHub |
|-----------|--------|
| 👩 `Agresta Arianna` | [Click here](https://github.com/Arianna6400) |
| 👨 `Camplese Francesco` | [Click here](https://github.com/FrancescoCamplese00) |
| 👨 `Giacconi Alessio` | [Click here](https://github.com/AlessioGiacconi) |
| 👨 `Iasenzaniro Andrea` | [Click here](https://github.com/AndreaIasenzaniro) |
