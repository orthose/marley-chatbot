import importlib
import traceback
import discord
import nltk
import spacy
from dotenv import get_key
import api_airfranceklm.utils as afkl
import marley_chatbot as marley

# Si les installations ne sont pas exécutées correctement
# Il faut les réaliser manuellement dans un interpréteur Python
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download('maxent_treebank_pos_tagger')

if importlib.util.find_spec("en_core_web_sm") is None:
    spacy.cli.download("en_core_web_sm")

client = discord.Client()
CHANNELS = ["spam"]
chats = marley.MultiChat(afkl.Context(api_key=get_key(".env", "API_KEY"), accept_language='us-US'))


@client.event
async def on_ready():
    print('Bot logged in as {0.user}'.format(client))


# noinspection PyBroadException
@client.event
async def on_message(message):
    try:
        if message.author == client.user:
            return
        if str(message.channel) not in CHANNELS:
            return

        content = message.content

        user = message.author.id
        # Enregistrement d'un nouvel utilisateur
        if not chats.is_registered(user):
            chats.register(user)

        # Analyse du message de l'utilisateur
        chats.parse(user, content)

        # Si tous les paramètres sont entrés
        # Affichage des offres disponibles
        if chats.are_params_set(user):
            offers = chats.get_offers(user)
            await message.reply(offers)
        else:
            await message.reply(chats.respond(user))

    except Exception:
        traceback.print_exc()
        await message.reply("An error occurred, please try again later")


client.run(get_key(".env", "BOT_TOKEN"))
