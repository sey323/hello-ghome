import sys
sys.path.append('./ghome')
from ghome_driver import *

if __name__ == '__main__':
    if (len(sys.argv) != 2):   # 引数が足りない場合は、その旨を表示
        print('Usage: # python %s text' % sys.argv[0])
        quit()

    text = sys.argv[1]

    ghome=GhomeDriver(name='オフィス')
    ghome.say(text)
