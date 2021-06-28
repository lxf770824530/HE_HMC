
from keras.utils.np_utils import *
import pickle as pl
import keras
import numpy as np
if __name__ == '__main__':


    class_num_1 = 7
    class_num_2 = 134


    train_txt, y1, y2 = pl.load(open(r'G:\HTMC-2\DATA\processed_data\WOS46985/train_txt-len-y_300_pad0_glove', 'rb'))
    test_txt, ty1, ty2 = pl.load(open(r'G:\HTMC-2\DATA\processed_data\WOS46985/test_txt-len-y_300_pad0_glove', 'rb'))
    data = np.array(train_txt + test_txt)
    label1 = np.array(y1 + ty1)
    label2 = np.array(y2 + ty2)

    i=1
    ks = pl.load(open(r'G:\HTMC-2\DATA\processed_data\WOS46985\predict_label\kfold_WOS', 'rb'))          #加载分割好的5折数据集   不同数据集需要修改
    scores=[]
    for train, test in ks:

        filepath = r"G:\HTMC-2\model\HE_HMC\WOS46985/gru_layer2_fold"+str(i)+".h5"

        ty2_c = to_categorical(label2[test].tolist(), class_num_2)


        ty1_p = pl.load(open(r'G:\HTMC-2\DATA\processed_data\WOS46985\predict_label/layer1_fold'+str(i)+'_predict','rb'))     #修改文件名 layer2_predict_we



        mode=keras.models.load_model(filepath)    #加载模型


        loss,acc=mode.evaluate([data[test], ty1_p, ty1_p], [ty2_c], batch_size=256)                                                           #修改---------------------------
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
