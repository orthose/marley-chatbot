import traceback
import discord
import nltk
from dotenv import get_key
import api_airfranceklm.utils as afkl
import marley_chatbot as marley

# Si les installations ne sont pas exécutées correctement
# Il faut les réaliser manuellement dans un interpréteur Python
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download('maxent_treebank_pos_tagger')

client = discord.Client()
CHANELS = ["spam"]
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
        if str(message.channel) not in CHANELS:
            return

        content = message.content

        if content == "!ping":
            await message.reply("pong")
            return

        user = message.author.id
        if not chats.is_registered(user):
            chats.register(user)

        chats.parse(user, content)
        if chats.are_params_set(user):
            offers = chats.get_offers(user)
            await message.reply(offers)
        else:
            await message.reply(chats.respond(user))
    except Exception:
        traceback.print_exc()
        await message.reply("An error occurred, please try again later")


client.run(get_key(".env", "BOT_TOKEN"))
