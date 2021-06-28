## model creation on Keras


from keras.utils.np_utils import *
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Multiply, multiply,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import math
def step_decay(epoch):
	initial_lrate = 0.01*3
	drop = 0.5
	epochs_drop = 3.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
from keras import backend as K
from keras import regularizers, initializers
class_f=14
class_num=76
layer = 2
def createHierarchicalAttentionModel(maxSeq,
                                     embWeights=None, embeddingSize=None, vocabSize=None,
                                     recursiveClass=GRU, wordRnnSize=100):
    wordsInputs = Input(shape=(maxSeq,), dtype='int32', name='words_input')


    #词嵌入
    y1Inputs = Input(shape=(1,), dtype='int32', name='y1_input')
    label_file=r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/label_1_vector'
    labels_vector = pl.load(open(label_file, 'rb'))                                                       #修改类别嵌入路径------
    embedder_label = Embedding(class_f, 300, weights=[labels_vector], input_length=1, trainable=True)
    label1 = embedder_label(y1Inputs)
    label1=Lambda(lambda x:K.squeeze(x,1))(label1)

    #图嵌入
    y1_ge_Inputs= Input(shape=(1,), dtype='int32', name='y1_ge_input')
    labels_ge_vector = pl.load(open(r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/ge_label1_2', 'rb'))                                                  #修改图嵌入路径---------
    a = np.zeros((len(labels_ge_vector[1]), 300))
    for i in range(len(labels_ge_vector[1])):
        a[i] = labels_ge_vector[1][i]
    embedder_label = Embedding(class_num, 300, weights=[a], input_length=1, trainable=True)                      #修改上一层类别数目-------------
    label_ge_1_2 = embedder_label(y1_ge_Inputs)
    label_ge_1_2=Lambda(lambda x:K.squeeze(x,1))(label_ge_1_2)




    if embWeights is None:
        emb = Embedding(vocabSize, embeddingSize)(wordsInputs)
    else:
        emb = Embedding(embWeights.shape[0], embWeights.shape[1], weights=[embWeights], trainable=False,mask_zero=True)(wordsInputs)

    wordRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True,dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(emb)
    wordRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
                            merge_mode='concat')(wordRnn)
    feature_wg = Concatenate(axis=-1)([label_ge_1_2,label1])
    features=Concatenate(axis=-1)([feature_wg,wordRnn])
    documentOut = Dense(800, activation="tanh", name="documentOut1")(features)
    documentOut=BatchNormalization()(documentOut)
    documentOut = Dense(class_num, activation="softmax", name="documentOut")(documentOut)
    model = Model(inputs=[wordsInputs,y1Inputs,y1_ge_Inputs], outputs=[documentOut])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

import pickle as pl
import keras
if __name__ == '__main__':
    # filepath = "../DATA/dbpedia/gru_layer2_we_ge.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    #callback
    lrate = LearningRateScheduler(step_decay)
    early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')

    pretrained_w2v, _, _ = pl.load(open(r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/emb_matrix_glove_300', 'rb'))
    print(np.shape(pretrained_w2v))
    train_txt,y1,y2=pl.load(open(r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/train_txt-len-y_300_pad0_glove','rb'))
    test_txt, ty1, ty2 = pl.load(open(r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/test_txt-len-y_300_pad0_glove', 'rb'))
    data = np.array(train_txt + test_txt)
    label1 = np.array(y1 + ty1)
    label2 = np.array(y2 + ty2)
    if layer == 1:
        label_layer = label1
    elif layer == 2:
        label_layer = label2

    ks = pl.load(open(r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label\kfold_bestbuy', 'rb'))          #加载分割好的5折数据集   不同数据集需要修改


    i=1
    scores=[]
    for train, test in ks:


        y_train = to_categorical(label_layer[train].tolist(), class_num)                                 #层数不同 类别标签的层级也需要修改
        y_test = to_categorical(label_layer[test].tolist(), class_num)

        filepath = r"G:\HTMC-2\model\HE_HMC\bestbuy/gru_layer2_fold"+str(i)+".h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
        mode=createHierarchicalAttentionModel(300,embWeights=pretrained_w2v)
        mode.fit([data[train],label1[train],label1[train]],[y_train],batch_size=64,epochs=100,validation_data=([data[test],label1[test],label1[test]],[y_test]),callbacks=[checkpoint,early_stopping])



        #获取预测标签
        mode = keras.models.load_model(filepath)  # 加载模型
        score = mode.evaluate([data[test],label1[test],label1[test]], [y_test], batch_size=256)
        scores.append(score[1])
        predict=mode.predict([data[test],label1[test],label1[test]],batch_size=256)
        predict = np.argmax(predict, axis=1)
        label_save_name = r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label\predict_label/layer2_fold'+str(i)+'_predict'
        pl.dump(predict,open(label_save_name,'wb'))
        i += 1

    #输出每个fold的分数
    j=0
    sum=0
    for j in range(len(scores)):
        print('{0} fold score: {1}'.format(j+1,scores[j]))
        sum+=scores[j]
    mean_s = sum/5
    print('mean score:',mean_s)

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
    # y1=to_categorical(y1,7)
    # y2=to_categorical(y2,class_num)
    # # ty1=to_categorical(ty1,7)
    # ty2=to_categorical(ty2,class_num)
    # mode=createHierarchicalAttentionModel(300,embWeights=pretrained_w2v)
    #
    #
    # mode.fit([train_txt,y1,y1],[y2],batch_size=64,epochs=150,validation_data=([test_txt,ty1,ty1],[ty2]),callbacks=[checkpoint])
    #真实标签+文本


    #预测+测试
    # mode=keras.models.load_model(filepath)
    # # ty1=pl.load(open('layer1_predict','rb'))
    # ty1=np.random.randint(0,9,size=[len(ty1)])
    # pl.dump(ty1,open("ty1_suiji",'wb'))
    # # print(mode.evaluate([test_txt,ty1],[ty2]))
    # predict=mode.predict([test_txt,ty1])
    # predict = np.argmax(predict, axis=1)
    # pl.dump(predict, open('layer2_predict_suiji', 'wb'))


