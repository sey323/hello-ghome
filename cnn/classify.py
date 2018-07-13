# coding: UTF-8

import sys , os
import cv2

sys.path.append('./cnn/util')
import image_dataset
sys.path.append('./cnn/cnn')
from cnn import *


def main(file_name ):
    IMG_SIZE = 64

    '''
    モデル設定
    '''
    model = CNN(n_size=IMG_SIZE,
                n_in=[ IMG_SIZE ,IMG_SIZE ],
                n_channels =3,
                labels=3,
                nBatch = 1)

    model.restore('./models/face')

    # データのローディング
    src = cv2.imread(file_name)
    img = cv2.resize( src , (IMG_SIZE , IMG_SIZE))
    img = img.flatten().astype(np.float32)/255.0
    img = np.reshape( img , [1, IMG_SIZE , IMG_SIZE , 3 ])

    result = model.evaluate( img )
    print("結果は")
    print(result[0])


if __name__ == '__main__':
    if (len(sys.argv) != 2):   # 引数が足りない場合は、その旨を表示
        print('Usage: # python %s filename' % sys.argv[0])
        quit()

    file_name = sys.argv[1]

    main( file_name )
