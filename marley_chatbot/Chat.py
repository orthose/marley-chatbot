import os
import re
from enum import Enum

import nltk
from dotenv import get_key

from api_airfranceklm.open_data import offers
import api_airfranceklm.utils as afkl
from typing import Optional
import pandas as pd
from datetime import datetime
from nltk.tag import StanfordNERTagger
import en_core_web_sm
import parsedatetime as pdt

java_path = get_key(".env", "JAVA_PATH")
os.environ['JAVAHOME'] = java_path
stanford = get_key(".env", "STANFORD_PATH")
stanfordClassifier = stanford + '/classifiers/english.muc.7class.distsim.crf.ser.gz'
stanfordNerPath = stanford + '/stanford-ner.jar'
st = StanfordNERTagger(stanfordClassifier, stanfordNerPath, encoding='utf8')

default_tagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
model = {'from': 'FROM', 'From': "FROM", "FROM": "FROM"}
tagger = nltk.tag.UnigramTagger(model=model, backoff=default_tagger)

grammar = r"""
  LOCATION: {<NNP>+}
  ARRIVAL: {<TO><LOCATION>}
  DEPARTURE: {<FROM><LOCATION>}
  """

dates_regex = [
    "\d{1,2}[/ -]\d{1,2}[/ -](\d{2}|\d{4})",
    "\d{1,2}[/ -]\w{3}[/ -](\d{2}|\d{4})",
    "(\d{2}|\d{4})[/ -]\d{1,2}[/ -]\d{1,2}",
    "(\d{2}|\d{4})[/ -]\w{3}[/ -]\d{1,2}",
]

nlp = en_core_web_sm.load()


class ResponseType(Enum):
    ERROR = -1
    IDLE = 0
    ASKING_ALL = 1
    ASKING_DEPARTURE_CITY = 2
    ASKING_ARRIVAL_CITY = 3
    ASKING_DEPARTURE_DATE = 4
    ASKING_HAS_RETURN = 5
    ASKING_RETURN_DATE = 6
    ASKING_CONFIRMATION = 7


class Chat:
    """
    Classe permettant de lancer une conversation entre Marley et un utilisateur humain.
    Elle permet de lier toutes les briques de base entre elles : du parsing jusqu'à la requête.
    """

    def __init__(self,
                 context_afkl: type(afkl.Context),
                 departure_city: Optional[str] = None,
                 arrival_city: Optional[str] = None,
                 departure_date: Optional[datetime.date] = None,
                 return_date: Optional[datetime.date] = None):
        """
        Les paramètres du vol sont optionnels car ils seront remplis
        au fur et à mesure de la discussion avec l'utilisateur
        :param context_afkl: Contexte de connexion à l'API Airfrance KLM
        :param departure_city: Ville de départ
        :param arrival_city: Ville d'arrivée
        :param departure_date: Date de départ
        :param return_date: Date de retour
        """
        self.started = False
        self.departure_city = departure_city
        self.arrival_city = arrival_city
        self.departure_date = departure_date
        self.return_date = return_date
        self.response_type = ResponseType.IDLE
        self.response, self.response_type = self._build_response()
        self.context_afkl = context_afkl

    def are_params_set(self) -> bool:
        """
        Les paramètres obligatoires ont-ils été entrés ?
        :return: True si tous les paramètres obligatoires ont été parsés False sinon
        """
        return (
                self.departure_city is not None
                and self.arrival_city is not None
                and self.departure_date is not None
                and self.return_date is not None
        )

    def get_offers(self) -> pd.DataFrame:
        """
        Requête à l'API Airfrance KLM en fonction des paramètres entrés
        :return: Dataframe Pandas des offres disponibles
        """
        ref = offers.reference_data(context=self.context_afkl)
        code_departure = ref[ref["location_name"] == self.departure_city]["location_code"]
        code_arrival = ref[ref["location_name"] == self.arrival_city]["location_code"]
        connections = [afkl.Connection(
            departure_date=self.departure_date,
            departure_location=afkl.Location(type=afkl.LocationType.CITY, code=code_departure),
            arrival_location=afkl.Location(type=afkl.LocationType.CITY, code=code_arrival))]

        # Paramètre optionnel de date retour
        if self.return_date is not None and self.return_date:
            connections.append(afkl.Connection(
                departure_date=self.return_date,
                departure_location=afkl.Location(type=afkl.LocationType.CITY, code=code_arrival),
                arrival_location=afkl.Location(type=afkl.LocationType.CITY, code=code_departure)))

        res = offers.all_available_offers(
            context=self.context_afkl,
            connections=connections,
            passengers=[afkl.Passenger(id=0, type=afkl.PassengerType.ADT)],
            output_format='dataframe',
            verbose=True)

        return res

    def _parse_date(self, sentence: str):
        cal = pdt.Calendar()
        doc = nlp(sentence)
        departure_count = 0
        return_count = 0
        for ent in doc.ents:
            if ent.label_ == "DATE" or re.match("|".join(dates_regex), ent.text):
                date = cal.parseDT(ent.text, datetime.now())[0].date()
                if self.response_type == ResponseType.ASKING_ALL:
                    if self.departure_date is None:
                        departure_count += 1
                        self.departure_date = date
                    elif self.return_date is None:
                        return_count += 1
                        self.return_date = date
                if self.response_type == ResponseType.ASKING_DEPARTURE_DATE:
                    departure_count += 1
                    self.departure_date = date
                if self.response_type == ResponseType.ASKING_RETURN_DATE:
                    return_count += 1
                    self.return_date = date
        if departure_count > 1:
            self.departure_date = None
        if self.response_type == ResponseType.ASKING_RETURN_DATE and self.return_date is None:
            self.return_date = False
        if return_count > 1:
            self.return_date = None

    def _parse_cities(self, sentence: str):
        s = tagger.tag(nltk.word_tokenize(sentence))
        tree = nltk.RegexpParser(grammar).parse(s)

        departure_count = 0
        arrival_count = 0
        for sub_tree in tree.subtrees():
            if self.response_type in [ResponseType.ASKING_DEPARTURE_CITY, ResponseType.ASKING_ALL] \
                    and sub_tree.label() == "DEPARTURE":
                departure_count += 1
                self.departure_city = " ".join(list(map(lambda x: x[0], sub_tree.leaves()[1:])))
            if self.response_type in [ResponseType.ASKING_ARRIVAL_CITY, ResponseType.ASKING_ALL] \
                    and sub_tree.label() == "ARRIVAL":
                arrival_count += 1
                self.arrival_city = " ".join(list(map(lambda x: x[0], sub_tree.leaves()[1:])))
            if self.response_type != ResponseType.ASKING_ALL and sub_tree.label() == "LOCATION":
                if self.response_type == ResponseType.ASKING_DEPARTURE_CITY:
                    departure_count += 1
                    self.departure_city = " ".join(list(map(lambda x: x[0], sub_tree.leaves())))
                else:
                    arrival_count += 1
                    self.arrival_city = " ".join(list(map(lambda x: x[0], sub_tree.leaves())))
        if departure_count > 1:
            self.departure_city = None
        if arrival_count > 1:
            self.arrival_city = None

    def parse(self, sentence: str):
        """
        Parse la phrase entrée par l'utilisateur en extrayant les paramètres
        Met à jour la réponse du chatbot en fonction des paramètres déjà entrés
        ou si une erreur de parsing est détectée
        :param sentence: Phrase entrée par l'utilisateur
        """
        self._parse_cities(sentence)
        self._parse_date(sentence)

    def _build_response(self) -> tuple[str, ResponseType]:
        missing_data = []
        if self.departure_city is None:
            missing_data.append("departure_city")
        if self.arrival_city is None:
            missing_data.append("arrival_city")
        if self.departure_date is None:
            missing_data.append("departure_date")
        if self.return_date is None:
            missing_data.append("return_date")
        if len(missing_data) == 4:
            if self.response_type == ResponseType.ASKING_ALL:
                response = "I'm sorry, I don't understand, where would you like to go ?"
                response_type = ResponseType.ASKING_ARRIVAL_CITY
            else:
                response = "Hello, I am Marley. How can I help you ?"
                response_type = ResponseType.ASKING_ALL
        elif "departure_city" in missing_data:
            if self.response_type == ResponseType.ASKING_DEPARTURE_CITY:
                response = "Sorry, I don't understand, what is the departure city ?"
            else:
                response = "Okay, what is the departure city ?"
            response_type = ResponseType.ASKING_DEPARTURE_CITY
        elif "arrival_city" in missing_data:
            if self.response_type == ResponseType.ASKING_ARRIVAL_CITY:
                response = "Sorry, I don't understand, what is the arrival city ?"
            else:
                response = "Okay, what is the arrival city ?"
            response_type = ResponseType.ASKING_ARRIVAL_CITY
        elif "departure_date" in missing_data:
            if self.response_type == ResponseType.ASKING_DEPARTURE_DATE:
                response = "Sorry, I don't understand, what is the departure date ?"
            else:
                response = "Okay, what is the departure date ?"
            response_type = ResponseType.ASKING_DEPARTURE_DATE
        elif "return_date" in missing_data:
            if self.response_type == ResponseType.ASKING_RETURN_DATE:
                response = "Sorry, I don't understand, do you want a return ticket as well ? " \
                           "If so, what is the return date ?"
            else:
                response = "Okay, do you want a return ticket as well ? If so, what is the return date ?"
            response_type = ResponseType.ASKING_RETURN_DATE
        else:
            response = []
            if self.response_type == ResponseType.ASKING_CONFIRMATION:
                response += ["Sorry, I don't understand, do you want to go"]
            else:
                response += ["Okay, so you want to go"]
            response += [
                "from", self.departure_city,
                "to", self.arrival_city,
                "on the", self.departure_date,
            ] + (['and return on the', self.return_date] if self.return_date else [])
            response = " ".join(response) + ", is that correct ?"
            response_type = ResponseType.ASKING_CONFIRMATION
        return response, response_type

    def respond(self) -> str:
        """
        Réponse du chatbot en fonction du dernier parsing
        :return: Réponse actuelle du chatbot
        """
        self.response, self.response_type = self._build_response()
        return self.response

    def converse(self):
        """
        Lancement de la conversation dans le terminal
        """
        print(self.respond())
        while not self.are_params_set():
            sentence = input('> ')
            self.parse(sentence)
            print(self.respond())
        print(self.get_offers().to_string())


if __name__ == '__main__':
    chat = Chat(afkl.Context(api_key_file='./api_key.txt', accept_language='us-US'))
    chat.converse()
