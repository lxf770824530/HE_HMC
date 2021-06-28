
from keras.utils.np_utils import *
import pickle as pl
import keras
import numpy as np
if __name__ == '__main__':



    class_num_3 = 219

    train_txt, y1, y2, y3 = pl.load(open('G:\HTMC-2\DATA\processed_data\dbpedia/train_txt-len-y_300_pad0_glove', 'rb'))
    test_txt, ty1, ty2, ty3 = pl.load(open('G:\HTMC-2\DATA\processed_data\dbpedia/test_txt-len-y_300_pad0_glove', 'rb'))
    data = np.array(train_txt + test_txt)
    label1 = np.array(y1 + ty1)
    label2 = np.array(y2 + ty2)
    label3 = np.array(y3 + ty3)

    i=1
    ks = pl.load(open(r'G:\HTMC-2\DATA\processed_data\dbpedia\predict_label\kfold_dbpedia', 'rb'))          #加载分割好的5折数据集   不同数据集需要修改
    scores=[]
    for train, test in ks:

        filepath = r"G:\HTMC-2\model\HE_HMC\DBpedia/gru_layer3_fold"+str(i)+".h5"


        ty3_c = to_categorical(label3[test].tolist(), class_num_3)


        ty2_p = pl.load(open('G:\HTMC-2\DATA\processed_data\dbpedia\predict_label/layer2_fold'+str(i)+'_predict', 'rb'))                    #修改文件名 layer2_predict_we



        mode=keras.models.load_model(filepath)    #加载模型


        loss,acc=mode.evaluate([data[test], ty2_p, ty2_p], [ty3_c], batch_size=256)                                                           #修改---------------------------
        scores.append(acc)
        i+=1
    #使用第二层预测的标签嵌入
    j = 0
    sum = 0
    for j in range(len(scores)):
        print('{0} fold loss: {1}'.format(j + 1, scores[j]))
        sum += scores[j]
    mean_s = sum / 5
    print('mean score:', mean_s)
