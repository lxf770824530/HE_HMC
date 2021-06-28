## model creation on Keras


from keras.utils.np_utils import *
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Multiply, multiply,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
import math
def step_decay(epoch):
	initial_lrate = 0.01*3
	drop = 0.5
	epochs_drop = 3.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
from keras import backend as K
from keras import regularizers, initializers

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')


class_num=99

def createHierarchicalAttentionModel(maxSeq,
                                     embWeights=None, embeddingSize=None, vocabSize=None,
                                     recursiveClass=GRU, wordRnnSize=100):
    wordsInputs = Input(shape=(maxSeq,), dtype='int32', name='words_input')
    y1_ge_Inputs= Input(shape=(1,), dtype='int32', name='y1_input')
    labels_ge_vector = pl.load(open('G:\HTMC-2\DATA\processed_data\kaggle\data29633_digital_label/ge_label2_3', 'rb'))                                   #修改图嵌入路径------
    a = np.zeros((len(labels_ge_vector[1]), 300))
    for i in range(len(labels_ge_vector[1])):
        a[i] = labels_ge_vector[1][i]
    embedder_label = Embedding(class_num, 300, weights=[a], input_length=1, trainable=True)
    label_ge_1_2 = embedder_label(y1_ge_Inputs)
    label_ge_1_2=Lambda(lambda x:K.squeeze(x,1))(label_ge_1_2)

    if embWeights is None:
        emb = Embedding(vocabSize, embeddingSize)(wordsInputs)
    else:
        emb = Embedding(embWeights.shape[0], embWeights.shape[1], weights=[embWeights], trainable=False,mask_zero=True)(wordsInputs)

    wordRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True,dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(emb)
    wordRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
                            merge_mode='concat')(wordRnn)

    features=Concatenate(axis=-1)([label_ge_1_2,wordRnn])
    documentOut = Dense(500, activation="tanh", name="documentOut1")(features)
    documentOut=BatchNormalization()(documentOut)
    documentOut = Dense(class_num, activation="softmax", name="documentOut")(documentOut)
    model = Model(inputs=[wordsInputs,y1_ge_Inputs], outputs=[documentOut])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

import pickle as pl
import keras
if __name__ == '__main__':
    filepath = r"G:\HTMC-2\model\HE_HMC\ablation_study\graph_Embedding\kaggle/gru_layer3_ge.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    pretrained_w2v, _, _ = pl.load(open('G:\HTMC-2\DATA\processed_data\kaggle\data29633_digital_label/emb_matrix_glove_300', 'rb'))
    train_txt, y1, y2, y3 = pl.load(open('G:\HTMC-2\DATA\processed_data\kaggle\data29633_digital_label/train_txt-len-y_300_pad0_glove', 'rb'))  # 3层
    test_txt, ty1, ty2, ty3 = pl.load(open('G:\HTMC-2\DATA\processed_data\kaggle\data29633_digital_label/test_txt-len-y_300_pad0_glove', 'rb'))
    # train_txt, y1, y2= pl.load(open('../DATA/WOS5736/train_txt-len-y_300_pad0_glove', 'rb'))
    # test_txt, ty1, ty2 = pl.load(open('../DATA/WOS5736/test_txt-len-y_300_pad0_glove', 'rb'))
    # y1=to_categorical(y1,7)
    y3=to_categorical(y3,class_num)
    # ty1=to_categorical(ty1,7)
    ty3=to_categorical(ty3,class_num)
    mode=createHierarchicalAttentionModel(300,embWeights=pretrained_w2v)

    mode.fit([train_txt,y2],[y3],batch_size=64,epochs=150,validation_data=([test_txt,ty2],[ty3]),callbacks=[checkpoint,early_stopping])


    mode = keras.models.load_model(filepath)  # 加载模型
    score = mode.evaluate([test_txt,ty2],[ty3], batch_size=256)
    print(score)
    predict = mode.predict([test_txt,ty2], batch_size=256)
    predict = np.argmax(predict, axis=1)
    label_save_name = r'G:\HTMC-2\model\HE_HMC\ablation_study\graph_Embedding\kaggle\predict_label/layer3_predict'
    pl.dump(predict, open(label_save_name, 'wb'))



    #测试
    # mode=keras.models.load_model(filepath)
    # # ty1=pl.load(open('layer1_predict','rb'))
    # ty1=np.random.randint(0,9,size=[len(ty1)])
    # pl.dump(ty1,open("ty1_suiji",'wb'))
    # # print(mode.evaluate([test_txt,ty1],[ty2]))
    # predict=mode.predict([test_txt,ty1])
    # predict = np.argmax(predict, axis=1)
    # pl.dump(predict, open('layer2_predict_suiji', 'wb'))


