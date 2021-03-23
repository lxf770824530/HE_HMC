
from keras.utils.np_utils import *
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Multiply, multiply,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.engine.topology import Layer
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
from keras.callbacks import EarlyStopping
import math
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.01*3
	drop = 0.5
	epochs_drop = 3.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
from keras import backend as K
from keras import regularizers, initializers


class_num = 7

def createHierarchicalAttentionModel(maxSeq,
                                     embWeights=None, embeddingSize=None, vocabSize=None,  # embedding
                                     recursiveClass=GRU, wordRnnSize=100, sentenceRnnSize=100,  # rnn
                                     # wordDenseSize = 100, sentenceHiddenSize = 128, #dense
                                     dropWordEmb=0.2, dropWordRnnOut=0.2, dropSentenceRnnOut=0.5):
    wordsInputs = Input(shape=(maxSeq,), dtype='int32', name='words_input')
    if embWeights is None:
        # , mask_zero=True
        emb = Embedding(vocabSize, embeddingSize)(wordsInputs)
    else:
        emb = Embedding(embWeights.shape[0], embWeights.shape[1], weights=[embWeights], trainable=False,mask_zero=True)(wordsInputs)

    wordRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True,dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(emb)
    wordRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
                            merge_mode='concat')(wordRnn)

    features=wordRnn
    documentOut = Dense(256, activation="tanh", name="documentOut1")(features)
    documentOut=BatchNormalization()(documentOut)
    documentOut = Dense(class_num, activation="softmax", name="documentOut")(documentOut)           #类别数需要调整

    model = Model(inputs=[wordsInputs], outputs=[documentOut])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

import pickle as pl
import keras
import numpy as np
if __name__ == '__main__':
    filepath = "../DATA/patent11072/gru_layer1.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    lrate = LearningRateScheduler(step_decay)
    pretrained_w2v, _, _ = pl.load(open('../DATA/patent11072/emb_matrix_glove_300', 'rb'))
    print(np.shape(pretrained_w2v))
    train_txt,y1,y2,y3=pl.load(open('../DATA/patent11072/train_txt-len-y_300_pad0_glove','rb'))
    test_txt, ty1, ty2, ty3 = pl.load(open('../DATA/patent11072/test_txt-len-y_300_pad0_glove', 'rb'))
    y1=to_categorical(y1,class_num)
    ty1=to_categorical(ty1,class_num)                                                                           #类别数调整
    # y_train = keras.utils.to_categorical(y1, 9)
    # y_test = keras.utils.to_categorical(ty1, 9)

    # mode=createHierarchicalAttentionModel(300,embWeights=pretrained_w2v)

    # mode.fit([train_txt],[y1],batch_size=64,epochs=150,validation_data=([test_txt,],[ty1]),callbacks=[checkpoint])



    #获取预测标签
    mode=keras.models.load_model(filepath)    #加载模型
    print(len(test_txt))
    predict=mode.predict([test_txt])
    predict = np.argmax(predict, axis=1)
    print(predict)
    print(np.shape(predict))
    # pl.dump(predict,open('layer1_predict','wb'))

