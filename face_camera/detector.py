# -*- coding:utf-8 -*-
import sys
import os
import cv2

from datetime import datetime


#cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"

def face_detect( image ):
    #グレースケール変換
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔認識の実行
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

    return facerect

'''
切り抜きを行なったデータから顔領域のみを保存．
'''
def save_faceImage( image_path , facerect  , base = 256 , range = { "width" : 0 , "height" : 0 } , save_path = 'img' ):
    if type(image_path) is str:# 画像ファイルのパスで受け取った時
        image = cv2.imread(image_path)
        image_path = image_path.split("/")[-1].split(".")[0]
    else:# 画像を受け取った時．
        image = image_path
        image_path = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    #ディレクトリの作成
    if len(facerect) > 0:
        save_path = save_path
        if not os.path.exists( save_path ):
            os.mkdir( os.path.join( save_path ))

    for i , rect in enumerate(facerect):
        if rect[2] < base:
            continue
        #顔だけ切り出して保存
        x = rect[0] - range["width"]
        y = rect[1] - range["height"]
        width = rect[2]  - range["width"]
        height = rect[3]  - range["height"]
        dst = image[y:y+height, x:x+width]

        # 画像を保存
        new_image_path = save_path + '/' + image_path + "_" + str(i) + ".jpg";
        cv2.imwrite(new_image_path, dst)
        print(new_image_path + "is clip and saved!")
