import sys , os
import random

import numpy as np
import cv2

'''
データセットから画像とラベルをランダムに取得
'''
def random_sampling(*args,train_num,test_num = 0  ):
    zipped = list(zip(*args))
    #乱数を発生させ，リストを並び替える．
    np.random.shuffle(zipped)

    # バッチサイズ分画像を選択
    unzip = list(zip(*zipped))

    train_zipped = zipped[:train_num]
    train_zipped = list(zip(*train_zipped))
    # Numpy配列に再変換
    train_ary = []
    for ar in train_zipped:
        train_ary.append(np.asarray(ar))

    if test_num == 0: # 検証用データの指定がないとき
        return train_ary
    else:
        test_zipped = zipped[ train_num : train_num + test_num ]
        test_zipped = list(zip(*test_zipped))

        test_ary = []
        for ar in test_zipped:
            test_ary.append(np.asarray(ar))

        return train_ary,test_ary

'''
フォルダーの画像をランダム位置でクリップした後にリサイズして読み込む
type@ train_image : numpy.ndarray
type@ train_label : numpy.ndarray
'''
def make( folder_name , img_size = 0 , clip_num = 0 , clip_size = 0 ,train_num = 0 , test_num = 0 ):
    train_image = []
    test_image = []
    train_label = []
    test_label= []

    # フォルダ内のディレクトリの読み込み
    classes = os.listdir( folder_name )

    for i, d in enumerate(classes):
        files = os.listdir( folder_name + '/' + d  )

        tmp_image = []
        tmp_label = []
        for file in files:
            # 1枚の画像に対する処理
            if not 'png' in file and not 'jpg' in file:# jpg以外のファイルは無視
                continue

            # 画像読み込み
            img = cv2.imread( folder_name+ '/' + d + '/' + file )
            # one_hot_vectorを作りラベルとして追加
            label = np.zeros(len(classes))
            label[i] = 1

            # リサイズをする処理
            if img_size != 0:
                img = cv2.resize( img , (img_size , img_size ))
                img = img.flatten().astype(np.float32)/255.0
                tmp_image.append(img)
                tmp_label.append(label)
            elif clip_size != 0 and clip_num != 0:
                img = random_clip( img , clip_size , clip_num)
                tmp_image.extend( img )
                for j in range(clip_num):
                    tmp_label.append(label)
            else:
                img = img.flatten().astype(np.float32)/255.0
                tmp_image.append(img)
                tmp_label.append(label)
        # 枚数に指定がないときは全て取得．
        if train_num == 0 :
            train_image.extend( tmp_image )
            train_label.extend( tmp_label )
        # テスト画像の指定がないとき
        elif test_num == 0 :
            train = random_sampling( tmp_image , tmp_label , train_num=train_num )
            train_image.extend( train[0] )
            train_label.extend( train[1] )
        # train_numとtest_numでデータセットのサイズが指定されているときはその枚数分ランダムに抽出する
        else :
            train, test = random_sampling( tmp_image , tmp_label , train_num=train_num , test_num=test_num )
            train_image.extend( train[0] )
            train_label.extend( train[1] )
            test_image.extend( test[0] )
            test_label.extend( test[1] )

        print( "label:{0}\t  ,{1}".format(i , d))
        print(d + 'Reading complete!,' + str(len(files[0][0])) + ' Pictures exit. Unit On ' + str(train_num) )

    # numpy配列に変換
    train_image = np.asarray( train_image )
    train_label = np.asarray( train_label )

    if test_num != 0: #testデータセットがあるときは返り値が変わる
        test_image = np.asarray( test_image )
        test_label = np.asarray( test_label )
        return train_image , train_label , test_image , test_label

    return train_image , train_label
