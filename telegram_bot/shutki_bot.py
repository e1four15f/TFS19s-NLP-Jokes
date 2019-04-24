import argparse
import logging

from models.ulmfit_model import ULMFiT_Model
from models.lstm_model import LSTM_Model
from models.textgenrnn_model import TextgenRNN_Model

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters


class ShutkiBot:

    def __init__(self, token, path='data'):
        print('Bot initialization...')
        self._init_bot(token)
        print('Handlers initialization...')
        self._init_handlers()
        print('Models initialization...')
        self._init_models(path)
        print(self._bot.get_me())

    def _init_bot(self, token):
        self._token = token
        self._bot = telegram.Bot(token=self._token)
        self._updater = Updater(token=self._token)
        self._dispatcher = self._updater.dispatcher

        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def _init_models(self, path):
        self._ulmfit_model = ULMFiT_Model(path)
        self._lstm_model = LSTM_Model(path)
        self._textgenrnn_model = TextgenRNN_Model(path)

    def _init_handlers(self):
        def start(bot, update):
            text = \
            """
Привет, я телеграмм бот, который генерирует шутки
/joke (/j) - генерация шуток (ULMFiT AWD-LSTM)
/joke_t (/jt) {temperature} - генерация шуток с температурой [0.1, 1.0]
/joke_w (/jw) {words} - генерация шутки по начальным словам
/joke_wt (/jwt) {words} {temperature} - генерация шутки по начальным словам с температурой [0.1, 1.0]
/joke_exp1 (/je1) - генерация шуток моделью из первого эксперимента (LSTM)
/joke_exp2 (/je2) - генерация шуток моделью из второго эксперимента (textgenrnn)
/joke_exp2_t (/je2t) {temperature} - генерация шуток с температурой [0.1, 1.0] (textgenrnn)
/help (/h) - помощь
/about (/a) - об авторах

Примеры:
/joke_t 0.8
/joke_w Решил я сделать бота шутника, а он
/jwt Новинка в магазине 0.4
            """
            self._bot.send_message(chat_id=update.message.chat_id, text=text)

        def echo(bot, update):
            self._bot.send_message(chat_id=update.message.chat_id, text='Ахахах) 🤡')

        def joke(bot, update):
            text = self._ulmfit_model.generate()
            self._bot.send_message(chat_id=update.message.chat_id, text=text)
        
        def joke_t(bot, update, args):
            try:
                temperature = float(args[0]) if args and 0.1 <= float(args[0]) <= 1.0 else 1.0
            except ValueError:
                temperature = 1.0
            text = f'Температура: {temperature}\n\n'
            text += self._ulmfit_model.generate(temperature=temperature)
            self._bot.send_message(chat_id=update.message.chat_id, text=text)
            
        def joke_w(bot, update, args):
            words = ' '.join(args)
            text = f'Начальные слова: {words}\n\n'
            text += self._ulmfit_model.generate(text=words)
            self._bot.send_message(chat_id=update.message.chat_id, text=text)

        def joke_wt(bot, update, args):
            try:
                temperature = float(args[-1]) if args and 0.1 <= float(args[-1]) <= 1.0 else 1.0
                words = ' '.join(args[:-1])
            except ValueError:
                temperature = 1.0
                words = ' '.join(args)
                
            text = f'Начальные слова: {words}\n'
            text += f'Температура: {temperature}\n\n'
            text += self._ulmfit_model.generate(text=words, temperature=temperature)
            self._bot.send_message(chat_id=update.message.chat_id, text=text)
            
        def joke_exp1(bot, update):
            text = self._lstm_model.generate()
            self._bot.send_message(chat_id=update.message.chat_id, text=text)

        def joke_exp2(bot, update):
            text = self._textgenrnn_model.generate()
            self._bot.send_message(chat_id=update.message.chat_id, text=text)

        def joke_exp2_t(bot, update, args):
            try:
                temperature = float(args[0]) if args and 0.1 <= float(args[0]) <= 1.0 else 1.0
            except ValueError:
                temperature = 1.0
            text = f'Температура: {temperature}\n\n'
            text += self._textgenrnn_model.generate(temperature=temperature)
            self._bot.send_message(chat_id=update.message.chat_id, text=text)
            
        def about(bot, update):
            text = 'Мы клоуняры 🤡\n\nhttps://github.com/e1four15f/TFS19s-NLP-Jokes'
            self._bot.send_message(chat_id=update.message.chat_id, text=text)

        def debug(bot, update, args):
            if args[0] == 'ON':
                logging.getLogger().setLevel(logging.DEBUG)
                self._bot.send_message(chat_id=update.message.chat_id, text='DEBUG MODE ON')
            else:
                logging.getLogger().setLevel(logging.INFO)
                self._bot.send_message(chat_id=update.message.chat_id, text='DEBUG MODE OFF')

        def kill(bot, update):
            self._bot.send_message(chat_id=update.message.chat_id, text='Выключаюсь! 😤')
            self.stop()

        self._handlers = [CommandHandler(['start', 'help', 'h'], start),
                          CommandHandler(['joke', 'j'], joke),
                          CommandHandler(['joke_t', 'jt'], joke_t, pass_args=True),
                          CommandHandler(['joke_w', 'jw'], joke_w, pass_args=True),
                          CommandHandler(['joke_wt', 'jwt'], joke_wt, pass_args=True),
                          CommandHandler(['joke_exp1', 'je1'], joke_exp1),
                          CommandHandler(['joke_exp2', 'je2'], joke_exp2),
                          CommandHandler(['joke_exp2_t', 'je2t'], joke_exp2_t, pass_args=True),
                          CommandHandler(['about', 'a'], about),
                          CommandHandler('DEBUG', debug, pass_args=True),
                          CommandHandler('KILL', kill),
                          MessageHandler(Filters.text, echo)]

        for handler in self._handlers:
            self._dispatcher.add_handler(handler)

    def start(self):
        self._updater.start_polling()

    def stop(self):
        self._updater.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--token', required=True,
                        help='Bot\'s token to access the HTTP API')
    
    parser.add_argument('-p', '--path', default='data',
                        help='Path to models')
                            
    args = parser.parse_args()
    
    bot = ShutkiBot(args.token, args.path)
    bot.start()