# Progetti Data Science A.A. 2024/2025

## 🤖 Progetto 4: ChatBot

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

## 👥 Autori 

|Nome | GitHub |
|-----------|--------|
| 👩 `Agresta Arianna` | [Click here](https://github.com/Arianna6400) |
| 👨 `Camplese Francesco` | [Click here](https://github.com/FrancescoCamplese00) |
| 👨 `Giacconi Alessio` | [Click here](https://github.com/AlessioGiacconi) |
| 👨 `Iasenzaniro Andrea` | [Click here](https://github.com/AndreaIasenzaniro) |
