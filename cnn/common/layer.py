import tensorflow as tf
import numpy as np
import activation


'''
畳み込み(Convolution)
@x          :input
@filter_size[0],[1] : Conv filter(width,height)
@filter_size[2]     : input_shape(直前のoutputの次元数と合わせる)
@filter_size[3]     : output_shape(出力する次元数)
'''
def conv2d( x , stride , filter_size , i ,padding = 'SAME',BatchNorm = False):
    conv_w,conv_b = _conv_variable([ filter_size[0], filter_size[1], filter_size[2] , filter_size[3] ],name="conv{0}".format(i))
    conv =  tf.nn.conv2d( x,                                # 入力
                        conv_w,                             # 畳み込みフィルタ
                        strides = [1, stride, stride, 1],   # ストライド
                        padding = padding) + conv_b
    tf.summary.histogram("conv{0}".format(i) ,conv)
    if BatchNorm:
        conv = batchNorm(conv)
    return conv

def _conv_variable( weight_shape , name="conv" ):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        # define variables
        weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
    return weight, bias


'''
逆畳み込み層(Fractionally-strided Convolution)
@x          :input
@filter_size[0],[1]  : Conv filter
@filter_size[2]      : output_shape
@filter_size[3]      : input_shape
@output_shape[0]     : Batch Num
@output_shape[1],[2] : output width , height
@output_shape[3]     : output_shape
'''
def deconv2d( x, stride , filter_size , output_shape ,  i ,BatchNorm=False):
    deconv_w,deconv_b = _deconv_variable([ filter_size[0], filter_size[1], filter_size[2] , filter_size[3] ],name="deconv{0}".format(i))
    deconv =  tf.nn.conv2d_transpose(x,
                                    deconv_w,
                                    output_shape=[output_shape[0],output_shape[1],output_shape[2],output_shape[3]],
                                    strides=[1,stride,stride,1],
                                    padding = "SAME",
                                    data_format="NHWC") + deconv_b

    tf.summary.histogram("deconv{0}".format(i) ,deconv)
    if BatchNorm:
        deconv = batchNorm(deconv)
    return deconv

# 逆畳み込み層の計算グラフの定義
def _deconv_variable( weight_shape  , name = "deconv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        output_channels = int(weight_shape[2])
        input_channels  = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        # define variables
        weight = tf.get_variable("w", weight_shape    , initializer = tf.contrib.layers.xavier_initializer_conv2d())
        bias   = tf.get_variable("b", [input_channels], initializer = tf.constant_initializer(0.0))
    return weight, bias

def calcImageSize( height , width ,stride , num ):
    import math
    for i in range(num):
        height = int(math.ceil(float(height)/float(stride)))
        width = int(math.ceil(float(width)/float(stride)))

    return height , width


'''
全結合層(Fully Connection)
@input      : input
@output     : classes num
@nBatch     : Batch size
'''
def fc( input , output , i = None,BatchNorm = False):
    n_b, n_h, n_w, n_f = [int(x) for x in input.get_shape()]
    h_fc = tf.reshape(input ,[n_b,n_h*n_w*n_f])
    d_fc_w, d_fc_b = _fc_variable([n_h*n_w*n_f,output],name="fc{0}".format(i))
    fc = tf.matmul( h_fc , d_fc_w) + d_fc_b
    tf.summary.histogram("fc{0}".format(i) ,fc)
    if BatchNorm:
        fc = batchNorm(fc)
    return fc

# Fully Connection層の計算グラフの定義
def _fc_variable( weight_shape , name="fc" ):
    with tf.variable_scope(name):
        # check weight_shape
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape    = ( input_channels, output_channels)
        # define variables
        weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
        bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
    return weight, bias

'''
逆全結合層(DeFully Connection)
@input         : input
@zdim          : Input Dimention
@output[0]     : Batch Num
@output[1],[2] : output width , height
@output[3]     : output_shape
'''
def defc( input , zdim , output , i = None , activation = activation.ReLU,BatchNorm = False):
    d_fc_w, d_fc_b = _fc_variable([zdim, output[1]*output[2]*output[3]],name="defc{0}".format(i))
    h_fc_r  = tf.matmul( input , d_fc_w) + d_fc_b
    h_fc_a  = activation( h_fc_r )
    defc    = tf.reshape(h_fc_a ,[output[0] , output[1] , output[2] , output[3]] )
    if BatchNorm:
        defc = batchNorm(defc)
    tf.summary.histogram("defc{0}".format(i) ,defc)
    return defc

'''
BatchNorm
x               : input
'''
def batchNorm( x , decay=0.9 , updates_collections=None, epsilon=1e-5, scale=True, is_training=True, scope=None):
    return tf.contrib.layers.batch_norm(x,
                                        decay=decay,
                                        updates_collections=updates_collections,
                                        epsilon=epsilon,
                                        scale=scale,
                                        is_training=is_training,
                                        scope = scope )

'''
Residual Block
x               : input
@filter_size[0],[1] : Conv filter(width,height)
@filter_size[2]     : input_shape
@filter_size[3]     : output_shape(Not Use)
'''
def ResidualBlock(x ,stride ,filter_size ,i,padding='VALID',activation = activation.ReLU,BatchNorm=False):
    block_name = "Residual_Block_" + str(i)
    with tf.variable_scope( block_name ):
        p = int((filter_size[0] - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = conv2d(y,stride,[filter_size[0],filter_size[1],filter_size[2],filter_size[2]], padding=padding, i="c1")
        y = tf.pad(activation(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = conv2d(y, stride,[filter_size[0],filter_size[1],filter_size[2],filter_size[2]], padding=padding, i="c2")
        output = y+x
    return output
