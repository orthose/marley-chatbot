import pandas as pd
from marley_chatbot import Chat
import api_airfranceklm.utils as afkl


class MultiChat:
    """
    Classe de discussion avec Marley multi-utilisateur et mono-thread.
    On utilise un dictionnaire pour stocker chaque discussion indépendemment.
    """
    def __init__(self, context_afkl: type(afkl.Context)):
        self.chats = dict()
        self.context_afkl = context_afkl

    def is_registered(self, user: str) -> bool:
        return self.chats.__contains__(user)

    def register(self, user: str):
        self.chats[user] = Chat(self.context_afkl)

    def unregister(self, user: str):
        self.chats.pop(user)

    def are_params_set(self, user: str) -> bool:
        return self.chats[user].are_params_set()

    def get_offers(self, user: str) -> pd.DataFrame:
        return self.chats[user].get_offers()

    def parse(self, user: str, sentence: str):
        self.chats[user].parse(sentence)

    def respond(self, user) -> str:
        return self.chats[user].respond()
