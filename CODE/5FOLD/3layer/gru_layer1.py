
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
from sklearn.model_selection import StratifiedKFold

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


class_num = 6
layer = 1

#CALLBACK
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')




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
    # filepath = "G:\HTMC-2\model\HE_HMC\KAGGLE/gru_layer1.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    lrate = LearningRateScheduler(step_decay)
    pretrained_w2v, _, _ = pl.load(open(r'G:\HTMC-2\DATA\processed_data\kaggle\data29633_digital_label/emb_matrix_glove_300', 'rb'))
    print(np.shape(pretrained_w2v))
    train_txt,y1,y2,y3=pl.load(open(r'G:\HTMC-2\DATA\processed_data\kaggle\data29633_digital_label/train_txt-len-y_300_pad0_glove','rb'))
    test_txt, ty1, ty2, ty3 = pl.load(open(r'G:\HTMC-2\DATA\processed_data\kaggle\data29633_digital_label/test_txt-len-y_300_pad0_glove', 'rb'))
    data= np.array(train_txt+test_txt)
    label1 = np.array(y1+ty1)
    label2 = np.array(y2+ty2)
    label3 = np.array(y3+ty3)
    if layer == 1:
        label_layer=label1
    elif layer == 2:
        label_layer = label2
    elif layer == 3:
        label_layer = label3
    # label1=to_categorical(label1,class_num)
    # y1=to_categorical(y1,class_num)
    # ty1=to_categorical(ty1,class_num)                                                                           #类别数调整
    # y_train = keras.utils.to_categorical(y1, 9)
    # y_test = keras.utils.to_categorical(ty1, 9)
    seed = 7
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


    i=1
    scores=[]
    kfold_s = kfold.split(data, label_layer)
    data_i_list=[]
    for train_i, test_i in kfold_s:
        data_i_list.append([train_i,test_i])

    pl.dump(data_i_list, open(r'C:\Users\E1106-0\Desktop\qqqq\kfold_kaggle', 'wb'))               #save k-fold division result
    ks=pl.load(open(r'C:\Users\E1106-0\Desktop\qqqq\kfold_kaggle', 'rb'))
    for train, test in ks:


        y_train = to_categorical(label_layer[train].tolist(), class_num)                                 #层数不同 类别标签的层级也需要修改
        y_test = to_categorical(label_layer[test].tolist(),class_num)

        filepath = "G:\HTMC-2\model\HE_HMC\KAGGLE/gru_layer1_fold"+str(i)+".h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
        mode=createHierarchicalAttentionModel(300,embWeights=pretrained_w2v)
        mode.fit(data[train],[y_train],batch_size=256,epochs=100,validation_data=(data[test],[y_test]),callbacks=[checkpoint,early_stopping])



        #获取预测标签
        mode = keras.models.load_model(filepath)  # 加载模型
        score = mode.evaluate(data[test], [y_test], batch_size=256)
        scores.append(score[1])
        predict=mode.predict(data[test],batch_size=256)
        predict = np.argmax(predict, axis=1)
        label_save_name = r'G:\HTMC-2\DATA\processed_data\kaggle\data29633_digital_label\predict_label/layer1_fold'+str(i)+'_predict'
        pl.dump(predict,open(label_save_name,'wb'))
        i += 1

    #输出每个fold的分数

    sum=0

    for j in range(len(scores)):
        print('{0} fold score: {1}'.format(j+1,scores[j]))
        sum+=scores[j]
    mean_s = sum/5
    print('mean score:',mean_s)

    with open(r'../../../DATA/result/result.txt', 'w', encoding='utf-8') as f:
        for j in range(len(scores)):
            f.write(str(j+1)+'fold score:'+str(scores[j]))
        f.write('mean score:'+str(mean_s))
    f.close()

    #获取预测标签
    # mode = keras.models.load_model(filepath)  # 加载模型
    # print(len(test_txt))
    # predict=mode.predict([test_txt])
    # predict = np.argmax(predict, axis=1)
    # print(predict)
    # print(np.shape(predict))
    # pl.dump(predict,open('layer1_predict','wb'))

