from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types.chat import ChatActions
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import model
from config import TOKEN
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types.message import ContentType
import os

button_help = KeyboardButton('/help')
button_start = KeyboardButton('/start_transfer')
kb_help_and_start = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).add(button_help).add(button_start)
kb_help = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).add(button_help)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


class WaitForPic(StatesGroup):
    waiting_for_content = State()
    waiting_for_style = State()
    waiting_for_start = State()


@dp.message_handler(commands=['start'], state="*")
async def process_start_command(message: types.Message):
    await message.reply("Hi! To start transferring process send /start_transfer and then send two pictures. "
                        " If you need help at any stage - just send /help command.", reply_markup=kb_help_and_start)
    await WaitForPic.waiting_for_start.set()


@dp.message_handler(commands=['help'], state=WaitForPic.waiting_for_start)
async def process_help_command(message: types.Message):
    await message.reply("To start transferring process send /start_transfer and then send two pictures.",
                        reply_markup=kb_help_and_start)


@dp.message_handler(commands=['help'], state=WaitForPic.waiting_for_content)
async def process_help_command(message: types.Message):
    await message.reply("Now you need to send the photo style will be applied to.", reply_markup=kb_help)


@dp.message_handler(commands=['help'], state=WaitForPic.waiting_for_style)
async def process_help_command(message: types.Message):
    await message.reply("Now you need to send the photo which style will be applied to picture you send previously.")


@dp.message_handler(commands=['start_transfer'], state=WaitForPic.waiting_for_start)
async def start_transfer(message: types.Message):
    await message.reply("Now you need to send the photo style will be applied to.", reply_markup=kb_help)
    await WaitForPic.waiting_for_content.set()


@dp.message_handler(content_types=['photo'], state=WaitForPic.waiting_for_content)
async def get_content_photo(message):
    content_name = 'content_{}.jpg'.format(message.from_user.id)
    await message.photo[0].download(content_name)
    await message.reply("Now you need to send the photo which style will be applied to picture you send previously.",
                        reply_markup=kb_help)
    await WaitForPic.waiting_for_style.set()


@dp.message_handler(content_types=['photo'], state=WaitForPic.waiting_for_style)
async def get_content_photo(message):
    style_name = 'style_{}.jpg'.format(message.from_user.id)
    content_name = 'content_{}.jpg'.format(message.from_user.id)
    await message.photo[0].download(style_name)
    await bot.send_chat_action(message.from_user.id, ChatActions.UPLOAD_DOCUMENT)
    await message.reply("Got it! Please, wait few minutes for processing...")
    await model.run(content_name, style_name, message.from_user.id)

    await bot.send_photo(message.from_user.id, open('result_{}.jpg'.format(message.from_user.id), 'rb'),
                         reply_to_message_id=message.message_id, caption='Done! To repeat the process just send'
                                                                         '/start_transfer again :)',
                         reply_markup=kb_help_and_start)
    await WaitForPic.waiting_for_start.set()
    os.remove('result_{}.jpg'.format(message.from_user.id))
    os.remove('style_{}.jpg'.format(message.from_user.id))
    os.remove('content_{}.jpg'.format(message.from_user.id))


@dp.message_handler(content_types=ContentType.ANY, state="*")
async def unknown_message(msg: types.Message):
    message_text = 'I do not know what I am supposed to do with it! \nKind reminder: there is /help button'
    await msg.reply(message_text, reply_markup=kb_help)


if __name__ == '__main__':
    executor.start_polling(dp)
