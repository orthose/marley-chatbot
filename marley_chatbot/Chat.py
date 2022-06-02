from api_airfranceklm.open_data import offers
import api_airfranceklm.utils as afkl
from typing import Optional
import pandas as pd
import datetime

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
        self.departure_city = departure_city
        self.arrival_city = arrival_city
        self.departure_date = departure_date
        self.return_date = return_date
        self.response = "Hello, I am Marley. I am here to help you. Please tell me where and when you want to go."
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
        )

    def get_offers(self) -> pd.DataFrame:
        """
        Requête à l'API Airfrance KLM en fonction des paramètres entrés
        :return: Dataframe Pandas des offres disponibles
        """
        connections = [afkl.Connection(
            departure_date=self.departure_date,
            departure_location=afkl.Location(type=afkl.LocationType.CITY, code=self.departure_city),
            arrival_location=afkl.Location(type=afkl.LocationType.CITY, code=self.arrival_city))]

        # Paramètre optionnel de date retour
        if self.return_date is not None:
            connections.append(afkl.Connection(
                departure_date=self.return_date,
                departure_location=afkl.Location(type=afkl.LocationType.CITY, code=self.arrival_city),
                arrival_location=afkl.Location(type=afkl.LocationType.CITY, code=self.departure_city)))

        res = offers.all_available_offers(
            context=self.context_afkl,
            connections=connections,
            passengers=[afkl.Passenger(id=0, type=afkl.PassengerType.ADT)],
            output_format='dataframe',
            verbose=True)

        return res

    # TODO
    def _parse_date(self, sentence: str):
        self.departure_date = datetime.date.today()

    # TODO
    def _parse_cities(self, sentence: str):
        self.departure_city = 'PAR'
        self.arrival_city = 'LON'
        # raise ValueError()

    # TODO
    def parse(self, sentence: str):
        """
        Parse la phrase entrée par l'utilisateur en extrayant les paramètres
        Met à jour la réponse du chatbot en fonction des paramètres déjà entrés
        ou si une erreur de parsing est détectée
        :param sentence: Phrase entrée par l'utilisateur
        """
        try:
            self._parse_date(sentence)
            self._parse_cities(sentence)
            self.response = "Very good I have all I want !"
        except ValueError:
            self.response = "Something is wrong..."

    def respond(self) -> str:
        """
        Réponse du chatbot en fonction du dernier parsing
        :return: Réponse actuelle du chatbot
        """
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

