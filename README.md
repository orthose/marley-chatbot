# Introduction
Projet de recommandation de voyage en avion par chatbot. 
On se limitera uniquement à une aide informative et pas à une véritable réservation.
Il serait possible dans une moindre mesure de rediriger le client vers l'interface web de réservation.

* **Étudiants** : Maxime VINCENT, Abderrahim BENMELOUKA, Loris PONROY, Lina SAICHI, Myriem MOULOUEL, Syrine MARZOUGUI
* **Module** : Projet Apprentissage
* **Enseignant** : [Sylvain Conchon](https://www.lri.fr/~conchon/)
* **Formation** : M1 ISD Apprentissage
* **Établissemnt** : Université Paris-Saclay
* **Année** : 2021-2022

# Objectif du projet
De manière générale, le but du projet d’apprentissage est de développer un chatbot capable de communiquer avec un humain. 
Notre chatbot vise à répondre aux besoins des voyageurs pour la recherche d'informations et la réservation de billets d’avion.
Plus précisément, l’objectif est de communiquer aux clients d’une compagnie aérienne des informations en temps réel sur 
les vols et réservations. Il pourra remplacer les moteurs de recherche classiques, en proposant une interface moderne 
et facile d’utilisation via la célèbre application de messagerie instantanée Discord.

# Description fonctionnelle des besoins
Le client engage la conversation avec le chatbot dans le but d’obtenir des informations sur un vol. 
Le chatbot doit être capable de demander des informations au client pour formuler sa requête. 
Lorsque tous les paramètres nécessaires sont renseignés et vérifiés, le bot envoie la requête 
à une API de réservation et affiche à l’utilisateur tous les vols disponibles avec les tarifs associés.

# Paramètres de requête
Les paramètres principaux ci-dessous pourront être enrichis au besoin avec des paramètres optionnels supplémentaire.

* Date de départ
* Date de retour (optionnel)
* Dates flexibles (optionnel)
* Ville de départ
* Ville d’arrivée
* Nombre de passagers (1 par défaut)

# Exemple de discussion
**Client**: “I want to go to Berlin”
**Bot**: “Okay, from where ?”
**Client**: “From Paris”
**Bot**: “So you want to go from Paris to Berlin, at what date ?”
**Client**: “The first of May”
**Bot**: “Noted, do you want to look for a return trip as well ?”
**Client**: “Yes, for the 5th of May”
**Bot**: “Alright, I should look for tickets from Paris to Berlin on the 1st of May with a return date
on 5th of May, correct ?”
**Client**: “Yes”
**Bot**: “Searching now...”
**Bot**: “Here are your results”
<< Le bot affiche des résultats >>

# Environnement technique
Les données de vol sont issues de l'API Rest Airfrance KLM. Nous avons développé la librairie Python
[api_airfranceklm](https://github.com/orthose/api-airfranceklm-python-sdk) pour la requêter.
Nous utilisons la bibliothèque [NLTK](https://www.nltk.org/) en Python pour faire l’analyse de texte.
Le chatbot est hébergé sur un serveur [Discord](https://discord.com/) qui servira d'interface utilisateur.

# Utilisation
Je vous conseille d'utiliser un [environnement virtuel](https://docs.python.org/3/library/venv.html).
On commence par installer les dépendances du projet.
```commandline
pip install -r requirements.txt
```

Il faut ensuite configurer les variables d'environnement. Pour cela on crée un fichier `.env` dans le dossier `marley_chatbot`.
Voici un exemple ci-dessous. Il faut évidemment le remplir avec le token du bot discord et la clé d'API de Airfrance KLM.
Le nom de canal choisi sera celui sur lequel le bot attendra les messages des utilisateurs.
Le chemin vers l'archive de Stanford ne doit pas être modifié car il est inclus dans le dépôt.
Cependant, le chemin vers la virtual machine Java doit être adapté en fonction de votre système.
```
BOT_TOKEN=...
DISCORD_CHANEL=spam
JAVA_PATH=/opt/java/jdk-11.0.3/bin
STANFORD_PATH=stanford-ner-2020-11-17
API_KEY=...
```

Enfin on lance le programme principal qui est un serveur qui va intercepter les messages des utilisateurs Discord.
```commandline
cd marley_chatbot
python3 discord_server.py
```
