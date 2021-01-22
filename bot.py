from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types.chat import ChatActions
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import torch
import nst_model
from config import TOKEN
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types.message import ContentType
import os
import io
import torchvision.transforms as transforms

button_help = KeyboardButton('/help')
button_start = KeyboardButton('/start_transfer')
button_monet = KeyboardButton('/Monet')
button_my_style = KeyboardButton('/my_style')

kb_help_and_start = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).add(button_help).add(button_start)
kb_help = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).add(button_help)
kb_choose_style = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).add(button_my_style).add(button_monet).add(button_help)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


class WaitForPic(StatesGroup):


    waiting_for_start = State()
    waiting_for_type_of_transfer = State()
    waiting_for_content_monet = State()
    waiting_for_content_custom = State()
    waiting_for_style = State()


@dp.message_handler(commands=['start', 'restart'], state="*")
async def process_start_command(message: types.Message):
    await message.reply("Hi! To start transferring process send /start_transfer and then send two pictures. "
                        " If you need help at any stage - just send /help command.", reply_markup=kb_help_and_start)
    await WaitForPic.waiting_for_start.set()


@dp.message_handler(commands=['help'], state=WaitForPic.waiting_for_start)
async def process_help_command(message: types.Message):
    await message.reply("To start transferring process send /start_transfer and then send two pictures.",
                        reply_markup=kb_help_and_start)

@dp.message_handler(commands=['help'], state=WaitForPic.waiting_for_type_of_transfer)
async def get_transfer_type(message: types.Message):
    await message.reply("Now you need to choose type of style - your one /my_style or from Monet pictures /Monet",
                        reply_markup=kb_choose_style)
    await WaitForPic.waiting_for_type_of_transfer.set()


@dp.message_handler(commands=['help'], state=[WaitForPic.waiting_for_content_monet, WaitForPic.waiting_for_content_custom])
async def process_help_command(message: types.Message):
    await message.reply("Now you need to send the photo style will be applied to.", reply_markup=kb_help)


@dp.message_handler(commands=['help'], state=WaitForPic.waiting_for_style)
async def process_help_command(message: types.Message):
    await message.reply("Now you need to send the photo which style will be applied to picture you send previously.")

@dp.message_handler(commands=['start_transfer'], state=WaitForPic.waiting_for_start)
async def get_transfer_type(message: types.Message):
    await message.reply("Now you need to choose type of style - your one /my_style or from Monet pictures /Monet",
                        reply_markup=kb_choose_style)
    await WaitForPic.waiting_for_type_of_transfer.set()


@dp.message_handler(commands=['Monet'], state=WaitForPic.waiting_for_type_of_transfer)
async def start_transfer_monet(message: types.Message):
    await message.reply("Now you need to send the photo style will be applied to.", reply_markup=kb_help)
    await WaitForPic.waiting_for_content_monet.set()

@dp.message_handler(commands=['my_style'], state=WaitForPic.waiting_for_type_of_transfer)
async def start_transfer_custom(message: types.Message):
    await message.reply("Now you need to send the photo style will be applied to.", reply_markup=kb_help)
    await WaitForPic.waiting_for_content_custom.set()

@dp.message_handler(content_types=['photo'], state=WaitForPic.waiting_for_content_monet)
async def monet_transfer(message):
    content_name = 'content_{}.jpg'.format(message.from_user.id)
    await message.photo[0].download(content_name)
    await message.reply("Got it! Please, wait few minutes for processing...")
    await bot.send_chat_action(message.from_user.id, ChatActions.UPLOAD_DOCUMENT)
    model = torch.load("monet.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = nst_model.image_loader(content_name, 512, device)
    img = model(img)

    unloader = transforms.ToPILImage()
    file = 'result_{}.jpg'.format(message.from_user.id)
    image = img.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(file)


    await bot.send_photo(message.from_user.id, open('result_{}.jpg'.format(message.from_user.id), 'rb'),
                         reply_to_message_id=message.message_id, caption='Done! To repeat the process just send'
                                                                         ' /start_transfer again :)',
                         reply_markup=kb_help_and_start)

    os.remove('content_{}.jpg'.format(message.from_user.id))
    await WaitForPic.waiting_for_start.set()


@dp.message_handler(content_types=['photo'], state=WaitForPic.waiting_for_content_custom)
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
    await message.reply("Got it! Please, wait few minutes for processing...")
    await bot.send_chat_action(message.from_user.id, ChatActions.UPLOAD_DOCUMENT)
    await nst_model.run(content_name, style_name, message.from_user.id)

    await bot.send_photo(message.from_user.id, open('result_{}.jpg'.format(message.from_user.id), 'rb'),
                         reply_to_message_id=message.message_id, caption='Done! To repeat the process just send'
                                                                         ' /start_transfer again :)',
                         reply_markup=kb_help_and_start)
    await WaitForPic.waiting_for_start.set()
    os.remove('result_{}.jpg'.format(message.from_user.id))
    os.remove('style_{}.jpg'.format(message.from_user.id))
    os.remove('content_{}.jpg'.format(message.from_user.id))


@dp.message_handler(content_types=ContentType.ANY, state="*")
async def unknown_message(msg: types.Message):
    message_text = 'I do not know what I am supposed to do with it! \nKind reminder: there is /help button ' \
                   'or you can restart bot /restart'
    await msg.reply(message_text, reply_markup=kb_help)


if __name__ == '__main__':
    executor.start_polling(dp)
