import numpy as np

'''
バッチを生成するクラス．
'''
class BatchGenerator(object):
    def __init__( self , image , label , size = [ 28 , 28 ] , channel = 3 , labeltype = 'onehot'):
        self.size = size
        self.channel = channel

        self.image = np.array(self._image2tensor(image))
        self.label = label

        self.idx = None

        # onehotじゃない時の処理
        if labeltype == 'onehot':
            self.label = np.array(self.label)
        else:
            self._onehot2array()

    '''
    画像をtensorflowで学習可能な形式に変換．
    '''
    def _image2tensor( self , img ):
        tensor = np.reshape( img , [len( img ), self.size[0] , self.size[1] , self.channel ])
        return tensor


    '''
    画像とラベルをバッチサイズ分取得する．
    type@ tensor : numpy.ndarray
    type@ label : numpy.ndarray
    '''
    def getBatch( self , nBatch , color = True , shuffle = True , idx = None):
        if not shuffle: # シャッフルしない
            self.idx = np.array([i for i in range(nBatch)])
        elif idx is None: # idxがない時
            self.idx = np.random.randint(0 , len( self.image ) - 1 , nBatch )
        else:
            self.idx = idx

        tensor,label = self.image[self.idx],self.label[self.idx]
        # normalized to -0.5 ~ +0.5
        tensor = (tensor-0.5)/1.0
        return tensor , label
