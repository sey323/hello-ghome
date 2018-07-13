import os
import sys
import cv2
import json

sys.path.append('./cnn')
from cnn import *
sys.path.append('./face_camera')
import detector
sys.path.append('./ghome')
from ghome_driver import *

IMG_SIZE = 64
thereshold = 1


def camera_faceclassifer( model ,label , range = { "width" : 0 , "height" : 0 }):
    print("[INITIALIZING]\tcamera setting")
    # カメラの設定
    cap = cv2.VideoCapture(0)

    print("[INITIALIZING]\tghome setting")
    ghome=GhomeDriver(name='オフィス')

    print("[STARTING]\tCAMERA STARTING")
    while(True):
        ret, frame = cap.read()
        if cv2.waitKey(1) == 27:
            break

        # 顔の検出と保存
        image = frame
        face_list = detector.face_detect( image )

        # 顔が検出できた時
        if len(face_list) > 0:
            #顔だけ切り出して保存
            x = face_list[0][0] - range["width"]
            y = face_list[0][1] - range["height"]
            width = face_list[0][2]  - range["width"]
            height = face_list[0][3]  - range["height"]
            src = image[y:y+height, x:x+width]

            # tensorflow用に変換
            img = cv2.resize( src , (IMG_SIZE , IMG_SIZE))
            img = img.flatten().astype(np.float32)/255.0
            img = np.reshape( img , [1, IMG_SIZE , IMG_SIZE , 3 ])

            result = model.evaluate(img)
            max_result = max(result[0])
            max_index =np.argmax(result)
            print("[DETECT]\t" + label["member"][max_index]["name"] , max_result)
            # 閾値処理
            if max_result > thereshold:
                ghome.hello(label["member"][max_index]["name"] )

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

'''
Json Loader
'''
def load_json(filename="config.json"):
    f = open( filename )
    json_data = json.load( f )
    f.close()
    return json_data


def main():
    json = load_json('config.json')
    num = len(json["member"])
    print("[INITIALIZING]\tmodel setting")
    # モデル設定
    model = CNN(n_size=IMG_SIZE,
                n_in=[ IMG_SIZE ,IMG_SIZE ],
                n_channels =3,
                labels=num,
                nBatch = 1)

    model.restore('./models/face')
    camera_faceclassifer( model , json )

if __name__ == '__main__':
    main()
