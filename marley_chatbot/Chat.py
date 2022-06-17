import os
import importlib
import re
from enum import Enum
import nltk
import spacy
from dotenv import get_key
from api_airfranceklm.open_data import offers
import api_airfranceklm.utils as afkl
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime
from nltk.tag import StanfordNERTagger
import parsedatetime as pdt
from dotenv import get_key
from text_to_num import text2num

if __name__ == '__main__':
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download('maxent_treebank_pos_tagger')

    if importlib.util.find_spec("en_core_web_sm") is None:
        spacy.cli.download("en_core_web_sm")

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

nlp = spacy.load("en_core_web_sm")


class ResponseType(Enum):
    ERROR = -1
    IDLE = 0
    ASKING_ALL = 1
    ASKING_DEPARTURE_CITY = 2
    ASKING_ARRIVAL_CITY = 3
    ASKING_DEPARTURE_DATE = 4
    ASKING_HAS_RETURN = 5
    ASKING_RETURN_DATE = 6
    ASKING_NUMBER_PASSENGERS = 7
    ASKING_ARE_DATES_FLEXIBLE = 8
    ASKING_CONFIRMATION = 9
    CONFIRMED = 10


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
                 return_date: Optional[datetime.date] = None,
                 flexible_dates: Optional[bool] = None,
                 number_passengers: Optional[int] = None,
                 response_type: Optional[ResponseType] = None):
        """
        Les paramètres du vol sont optionnels car ils seront remplis
        au fur et à mesure de la discussion avec l'utilisateur
        :param context_afkl: Contexte de connexion à l'API Airfrance KLM
        :param departure_city: Ville de départ
        :param arrival_city: Ville d'arrivée
        :param departure_date: Date de départ
        :param return_date: Date de retour
        """
        self.departure_city = departure_city
        self.arrival_city = arrival_city
        self.departure_date = departure_date
        self.return_date = return_date
        self.flexible_dates = flexible_dates
        self.number_passengers = number_passengers
        self.response = None
        self.response_type = response_type if response_type else ResponseType.IDLE
        self.context_afkl = context_afkl

    def reset(self):
        """
        Remise à zéro du chatbot pour préparer une nouvelle requête
        """
        self.__init__(self.context_afkl, response_type=ResponseType.ASKING_ALL)

    def are_params_set(self) -> bool:
        """
        Les paramètres obligatoires ont-ils été entrés ?
        :return: True si tous les paramètres obligatoires ont été parsés False sinon
        """
        return self.response_type == ResponseType.CONFIRMED

    # TODO: Fusionner __get_offers et get_offers
    def __get_offers(self) -> pd.DataFrame:
        """
        Requête à l'API Airfrance KLM en fonction des paramètres entrés
        :return: Dataframe Pandas des offres disponibles
        """
        # TODO: Si il y a une erreur car la location n'existe pas il faut changer l'état du chatbot
        ref = offers.reference_data(context=self.context_afkl)
        code_departure = ref[ref["location_name"] == self.departure_city].iloc[0]["location_code"]
        code_arrival = ref[ref["location_name"] == self.arrival_city].iloc[0]["location_code"]
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

        # TODO: Ratrapper les erreurs qui peuvent se produire ici en changeant l'état du chatbot
        res = offers.all_available_offers(
            context=self.context_afkl,
            connections=connections,
            passengers=[afkl.Passenger(id=0, type=afkl.PassengerType.ADT)],
            output_format='dataframe',
            verbose=True)

        return res

    def get_offers(self, top_offers=3, debug=False) -> str:
        """
        Construit la phrase de réponse du chatbot avec le top 3 des offres disponibles
        en fonction des paramètres entrés par l'utilisateur
        :param top_offers: Nombre d'offres à afficher
        :return: Réponse du chatbot sous forme d'une string
        """
        assert top_offers >= 1

        offers = self.__get_offers()
        if debug:
            print(offers.to_string())

        not_return_ticket = self.return_date is None or not self.return_date

        # On sélectionne uniquement quelques offres pour éviter de dépasser
        # la taille maximale de message dans Discord
        factor = 1 if not_return_ticket else 2
        top_offers = min(len(offers), top_offers * factor)
        offers = offers.iloc[0:top_offers]
        top_offers //= factor

        res = 'There is only one flight available ' if top_offers == 1 else f"These are the top {top_offers} cheapest flights "
        res += f"I can offer you from {self.departure_city} to {self.arrival_city}{'' if not_return_ticket else ' with a return ticket'}.\n\n"

        def response_offer(offer: pd.Series, return_flight=False) -> str:
            departure_datetime = offer['departure_datetime']
            arrival_datetime = offer['arrival_datetime']
            departure_airport = offer['departure_airport_name']
            arrival_airport = offer['arrival_airport_name']
            number_connections = offer['number_segments'] - 1
            price = offer['total_price']
            currency = offer['currency']

            return (
                        ("You" if not return_flight else "you") + f" can depart at {departure_datetime} from {departure_airport} and arrive at {arrival_datetime} at {arrival_airport}. "
                        + ("This route is without connection. " if number_connections == 0
                           else f"There {'is' if number_connections == 1 else 'are'} {number_connections} connection{'' if number_connections == 1 else 's'} for this route. ")
                        + f"Its price is {price} {'€' if currency == 'EUR' else currency}.")

        # Aller simple
        if not_return_ticket:
            for i in range(len(offers)):
                offer = offers.iloc[i]
                res += f"{i + 1}. " + response_offer(offer) + '\n\n'

        # Billet de retour demandé
        else:
            for i in range(0, len(offers), 2):
                offer1 = offers.iloc[i]
                offer2 = offers.iloc[i + 1]
                price1 = offer1['total_price']
                currency1 = offer1['currency']
                price2 = offer2['total_price']
                currency2 = offer2['currency']
                assert currency1 == currency2
                price = price1 + price2
                res += f"{i + 1}. " + response_offer(offer1)
                res += 'For the return journey, '
                res += response_offer(offer2)
                res += f"The total round trip is {price} {'€' if currency1 == 'EUR' else currency1}.\n\n"

        self.reset()
        return '```md\n' + res + '```'

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

    def _yes_no_score(self, sentence):
        tokens = tagger.tag(nltk.word_tokenize(sentence))
        affirmative_tokens = ["yes", "yeah", "sure", "okay"]
        affirmative_score = 0
        negative_tokens = ["no", "nope", "not", "never", "nevermind"]
        negative_score = 0
        for token, tag in tokens:
            if token.lower() in affirmative_tokens:
                affirmative_score += 1
            if token.lower() in negative_tokens:
                negative_score += 1
        score = affirmative_score / (affirmative_score + negative_score)
        return score

    def _parse_are_dates_flexible(self, sentence: str):
        score = self._yes_no_score(sentence)
        neutrality = 1 / 3
        if neutrality < score < 1 - neutrality:
            self.response_type = ResponseType.ASKING_ARE_DATES_FLEXIBLE
        else:
            self.flexible_dates = (score >= 1 - neutrality)

    def _parse_number_of_passengers(self, sentence: str):
        s = tagger.tag(nltk.word_tokenize(sentence))
        num_grammar = 'NumericalPhrase: {<NN|NNS>?<RB>?<JJR><IN><CD><NN|NNS>?}'
        parser = nltk.RegexpParser(num_grammar)
        res = parser.parse(s)
        numbers = []
        for x, y in res:
            z = None
            if y == "CD" and x.isnumeric():
                z = int(x) if int(x) == float(x) else float(x)
            elif y == "CD":
                try:
                    z = text2num(x.lower(), 'en')
                except ValueError:
                    pass
            if z is not None and z >= 1:
                numbers.append(z)
        self.number_passengers = numbers[0] if len(numbers) == 1 else None

    def _parse_confirmation(self, sentence: str):
        score = self._yes_no_score(sentence)
        neutrality = 1 / 3
        if neutrality < score < 1 - neutrality:
            self.response_type = ResponseType.ASKING_CONFIRMATION
        elif score >= 1 - neutrality:
            self.response_type = ResponseType.CONFIRMED
        else:
            self.reset()

    def parse(self, sentence: str):
        """
        Parse la phrase entrée par l'utilisateur en extrayant les paramètres
        Met à jour la réponse du chatbot en fonction des paramètres déjà entrés
        ou si une erreur de parsing est détectée
        :param sentence: Phrase entrée par l'utilisateur
        """
        if self.response_type in [
            ResponseType.ASKING_ALL,
            ResponseType.ASKING_DEPARTURE_CITY,
            ResponseType.ASKING_ARRIVAL_CITY,
        ]:
            self._parse_cities(sentence)

        if self.response_type in [
            ResponseType.ASKING_ALL,
            ResponseType.ASKING_DEPARTURE_DATE,
            ResponseType.ASKING_RETURN_DATE,
        ]:
            self._parse_date(sentence)

        if self.response_type == ResponseType.ASKING_ARE_DATES_FLEXIBLE:
            self._parse_are_dates_flexible(sentence)

        if self.response_type in [
            ResponseType.ASKING_ALL,
            ResponseType.ASKING_NUMBER_PASSENGERS
        ]:
            self._parse_number_of_passengers(sentence)

        if self.response_type == ResponseType.ASKING_CONFIRMATION:
            self._parse_confirmation(sentence)

    def _build_response(self) -> Tuple[str, ResponseType]:
        missing_data = []
        if self.departure_city is None:
            missing_data.append("departure_city")
        if self.arrival_city is None:
            missing_data.append("arrival_city")
        if self.departure_date is None:
            missing_data.append("departure_date")
        if self.return_date is None:
            missing_data.append("return_date")
        if self.flexible_dates is None:
            missing_data.append("flexible_dates")
        if self.number_passengers is None:
            missing_data.append("number_passengers")
        if len(missing_data) == 6:
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
        elif "flexible_dates" in missing_data:
            if self.response_type == ResponseType.ASKING_ARE_DATES_FLEXIBLE:
                response = "Sorry, I don't understand, are these dates flexible ? "
            else:
                response = "Okay, are these dates flexible ?"
            response_type = ResponseType.ASKING_ARE_DATES_FLEXIBLE
        elif "number_passengers" in missing_data:
            if self.response_type == ResponseType.ASKING_NUMBER_PASSENGERS:
                response = "Sorry, I don't understand, how many passengers for this trip ? "
            else:
                response = "Okay, how many passengers for this trip ?"
            response_type = ResponseType.ASKING_NUMBER_PASSENGERS
        else:
            response = []
            if self.response_type == ResponseType.ASKING_CONFIRMATION:
                response += ["Sorry, I don't understand, do you want to go"]
            else:
                response += ["Okay, so you want to go"]
            response += [
                            "from", self.departure_city,
                            "to", self.arrival_city,
                            "on the", str(self.departure_date),
                        ] + (['and return on the', str(self.return_date)] if self.return_date else [])
            response += ["with flexible dates"] if self.flexible_dates else []
            response += ["for " + str(self.number_passengers) + " " + ("people" if self.number_passengers > 1 else "person")]
            response = " ".join(response) + ", is that correct ?"
            response_type = ResponseType.ASKING_CONFIRMATION
            # TODO: Si CONFIRMED appeler la fonction get_offers dans ce cas là changer l'interface pour ne plus appeler get_offers depuis l'extérieur
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
        while not self.are_params_set():
            print(self.respond())
            sentence = input('>\t')
            self.parse(sentence)
        print(self.get_offers())


# Pour tester le chatbot dans le terminal
if __name__ == '__main__':
    chat = Chat(afkl.Context(api_key=get_key('.env', 'API_KEY'), accept_language='us-US'))
    chat.converse()
