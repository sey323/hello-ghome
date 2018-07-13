import sys , os
sys.path.append('./cnn/util')
import image_dataset
from BatchGenerator  import *
sys.path.append('./cnn/cnn')
from cnn import *

if __name__ == '__main__':
    IMG_SIZE = 64
    img_path = os.getenv("DATASET_FOLDER", "dataset")
    save_path = os.getenv("SAVE_FOLDER", "models")

    img_path = img_path
    config = tf.GPUOptions() #  specify GPU number)

    # データのローディング
    train_image ,train_label , test_image , test_label = image_dataset.make( img_path, train_num = 90 , test_num = 10 , img_size = IMG_SIZE )
    labelSize = len(train_label[0])
    train_batch = BatchGenerator( train_image , train_label , size = [ IMG_SIZE , IMG_SIZE  ])
    test_batch = BatchGenerator( test_image , test_label  , size = [ IMG_SIZE , IMG_SIZE  ])

    '''
    モデル設定
    '''
    model = CNN(n_size=IMG_SIZE,
                n_in=[IMG_SIZE ,IMG_SIZE ],
                n_channels =3,
                labels=len(train_label[0]),
                nBatch = 64,
                save_folder =  save_path)

    '''
    モデル学習
    '''
    model.fit(  train_batch.getBatch ,
                test_batch.getBatch ,
                nb_epoch = 1000)
