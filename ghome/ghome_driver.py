import pychromecast
from gtts_token import gtts_token
import urllib.parse

class GhomeDriver(object):
    def __init__(self,name):
        print("[INITIALIZING]\tsearching Google Home")
        self.googlehome_name = name
        chromecasts = pychromecast.get_chromecasts()
        self.cast = next(cc for cc in chromecasts if cc.device.friendly_name == self.googlehome_name)
        if self.cast ==[]:
            exit()

    # テキストを話させるメゾット
    def say(self,text,lang='ja'):
        token = gtts_token.Token()
        tk = token.calculate_token(text)

        payload = {
            'ie' : 'UTF-8',
            'q' : text,
            'tl' : lang,
            'total' : 1,
            'idx' : 0,
            'textlen' : len(text),
            'tk' : tk,
            'client' : 't',
            'ttsspeed' : 1.0
        }

        params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
        url = 'https://translate.google.com/translate_tts?{}'.format(params)
        self.cast.wait()
        mc = self.cast.media_controller
        mc.play_media(url, 'audio/mp3')

    # 挨拶
    def hello(self,text):
        message = ""
        if text == 'other':
            message = "初めまして，こんにちは"
        else:
            message = text + "，こんにちは，おげんきですか？"
        print("[TALKING]\t"+message)
        self.say(message)

'''
if __name__ == '__main__':
    ghome=GhomeDriver(name='オフィス')
    ghome.say("こんにちは")
'''
