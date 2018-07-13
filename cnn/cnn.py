import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.append('./cnn/common')
import layer
import activation

'''
基本となるCNNのクラス
'''
class CNN( object ):
    def __init__(   self,
                    n_size,
                    n_in,
                    n_channels,
                    labels,
                    nBatch,
                    filter_size=[5,5],
                    layers=[64,128,256],
                    keep_prob = 0.5 ,
                    save_folder = "models"):
        self.n_size = n_size
        self.n_in = n_in
        self.n_channels = n_channels
        self.labels = labels
        self.conv = None
        self.nBatch = nBatch
        self.keep_prob = keep_prob
        self.filter_size = filter_size
        self.layers = layers

        self._x = None
        self._t = None
        self._keep_prob = None
        self._accuracy = None
        self.sess = None

        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.saveFolder = save_folder + "/" + now
        if self.saveFolder and not os.path.exists(os.path.join(self.saveFolder,"images")):
            os.makedirs(os.path.join(self.saveFolder,"images"))

    '''
    Cross Entropy loss
    '''
    def cross_entropy( self , x , labels, nBatch , name = '' ):
        with tf.variable_scope('cross_entropy_'+name):
            x  = tf.reshape(x , [nBatch, -1])
            labels  = tf.reshape(labels , [nBatch, -1])

            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=x, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy_mean')
        return cross_entropy_mean


    def build_model(self , images , keep_prob):
        # Convolution layer
        x_image = tf.reshape(images, [-1, self.n_in[0] , self.n_in[1] , 3])

        with tf.variable_scope("Discriminator") as scope:
            with tf.variable_scope("conv_layer1") as scope:
                output     = layer.conv2d( x = x_image , stride=2 , filter_size = [self.filter_size[0],self.filter_size[1] , 3 , self.layers[0]], i = 1 ,BatchNorm = True)
                output     = activation.leakyReLU( output )
                tf.summary.histogram("conv_layer1",output)

            with tf.variable_scope("conv_layer2") as scope:
                # ResidualBlock
                output     = layer.ResidualBlock( x = output , stride=1 , filter_size = [self.filter_size[0],self.filter_size[1] , self.layers[0] , self.layers[1]], i = str(2)+'_'+str(1) ,BatchNorm = True)
                output     = layer.ResidualBlock( x = output , stride=1 , filter_size = [self.filter_size[0],self.filter_size[1] , self.layers[0] , self.layers[1]], i = str(2)+'_'+str(2) ,BatchNorm = True)
                output     = layer.conv2d( x = output , stride=2 , filter_size = [self.filter_size[0],self.filter_size[1] , self.layers[0] , self.layers[1]], i = 2 ,BatchNorm = True)
                output     = activation.leakyReLU( output )
                output     = tf.nn.dropout(output, keep_prob)

            tf.summary.histogram("conv_layer2",output)

            with tf.variable_scope("conv_layer3") as scope:
                output     = layer.conv2d( x = output , stride=2 , filter_size = [self.filter_size[0],self.filter_size[1] , self.layers[1] , self.layers[2]], i = 3 ,BatchNorm = True)
                output     = activation.leakyReLU( output )
                tf.summary.histogram("conv_layer3",output)

            h_fc_1 = tf.nn.dropout(output, keep_prob)
            # Fc1
            output = layer.fc( h_fc_1 ,  self.labels , "",BatchNorm = False)

        return output


    def training(self ,  loss ):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        return train_step


    '''
    モデルの性能の評価
    '''
    def accuracy( self , y , t , name = 'accuracy'):
        with tf.variable_scope(name):
            correct_prediction = tf.equal( tf.argmax( y , 1 ) , tf.argmax(t ,1 ))
            accuracy = tf.reduce_mean( tf.cast(correct_prediction , tf.float32))
        return accuracy


    '''
    モデルの学習
    '''
    def fit(self, train_batch, test_batch , nb_epoch=100):
        # Model
        x = tf.placeholder(tf.float32, [ self.nBatch , self.n_in[0] , self.n_in[1] , 3] , name = "train_image")
        labels = tf.placeholder( tf.float32 , [ self.nBatch , self.labels ], name = "train_label")
        keep_prob = tf.placeholder(tf.float32)

        # モデルの構築
        conv = self.build_model(x ,keep_prob)

        # Loss function
        loss = self.cross_entropy(x=conv,labels=labels,nBatch=self.nBatch,name="cross_entropy")
        train_step = self.training(loss)

        # Accuracy
        accuracy = self.accuracy(conv, labels)

        # 初期化
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        self.sess = sess

        tf.summary.scalar( "loss" , loss )

        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)

        for step in range( nb_epoch + 1):
            train_image_batch , train_label_batch = train_batch( self.nBatch )
            test_image_batch , test_label_batch = test_batch( self.nBatch )

            _, summary = sess.run( [train_step,self.summary] , feed_dict={x: train_image_batch, labels: train_label_batch,keep_prob:self.keep_prob })

            if step>0 and step%10==0:
                self.writer.add_summary(summary , step )

            if step%100==0:
                # 毎ステップ、学習データに対する正答率と損失関数を記録
                loss_ = loss.eval(session=sess, feed_dict={ x: test_image_batch, labels: test_label_batch ,keep_prob:1.0 })
                accuracy_ = accuracy.eval(session=sess,feed_dict={ x:test_image_batch, labels: test_label_batch,keep_prob:1.0})
                print("step %d, training accuracy %g loss %g"%(step, accuracy_ , loss_ ))
                self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),step)



    '''
    モデルで評価
    '''
    def evaluate( self, test_image, test_label=None):
        if test_label is None:
            return self.conv.eval(session=self.sess, feed_dict={
                self._x: test_image,
                self._keep_prob:1.0
            })
        else:
            return self.conv.eval(session=self.sess, feed_dict={
                self._x: test_image,
                self._t: test_label,
                self._keep_prob: 1.0
            })


    def save( self , file_name='model.ckpt'):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess , self.saveFolder + '/' +file_name )

    '''
    モデルの再構築
    '''
    def restore( self , file_name ):
        # Modelの再構築
        x = tf.placeholder(tf.float32, [ self.nBatch , self.n_in[0] , self.n_in[1] , 3] , name = "train_image")
        labels = tf.placeholder( tf.float32 , [ self.nBatch , self.labels ], name = "train_label")
        keep_prob = tf.placeholder(tf.float32)

        self._x = x
        self._t = labels
        self._keep_prob = keep_prob
        # モデルの構築
        conv = self.build_model(self._x ,self.keep_prob)
        self.conv= conv

        self.saver = tf.train.Saver()
        # モデルファイルが存在するかチェック
        ckpt = tf.train.get_checkpoint_state( file_name )
        if ckpt:
            print( "[LOADING]\t" + file_name)
            self.sess = tf.Session()
            self.saver.restore(self.sess, file_name + "/"+ckpt.model_checkpoint_path.split("/")[-1])
        else:
            print( file_name + ' Not found')
            exit()
