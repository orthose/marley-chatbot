import os
import traceback
import marley_chatbot as marley
import api_airfranceklm.utils as afkl
import discord
#from dotenv import load_dotenv
#from marley import parse


client = discord.Client()
CHANELS = []
chats = marley.MultiChat(afkl.Context(api_key_file='./api_key.txt', accept_language='us-US'))

@client.event
async def on_ready():
    print('Bot logged in as {0.user}'.format(client))


# noinspection PyBroadException
@client.event
async def on_message(message):
    try:
        if message.author == client.user:
            return
        if str(message.channel) != "spam":
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
            await message.reply(chats.get_offers(user).to_string())
        else:
            await message.reply(chats.respond(user))

        #data = parse(content)
        #await message.reply(str(data))


        pass
    except Exception:
        traceback.print_exc()
        await message.reply("500: Internal Error")


#load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
client.run(BOT_TOKEN)