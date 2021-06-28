#对文本词表做词嵌入对应，即  文本词表——>300向量
import numpy as np
import re
import pickle as pl
import os
import random







def text_cleaner(text):
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()


def get_allwords():
    data_path = r"G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/X.txt"                                        #修改路径
    data_X=[]
    with open(data_path,encoding='utf-8') as f:
        f=f.readlines()
        f=[text_cleaner(x) for x in f]
    print('len f:',len(f))
    for str in f:
        sentence =re.split('[ |,|(|)|\-|_|\'|/|*|&|#|:|.|\[|\]|?|^|>|<|;|+|=]',str)
        s = [x.strip() for x in sentence if not x.strip() == '']
        data_X.extend(s)
    return set(data_X)


glove_path=r'G:\wordEmbedding\glove\glove.42B.300d.txt'
def build_emb_matrix_and_vocab(embedding_size=300):
    # 0 th element is the default vector for unknowns.
    with open(glove_path,encoding="utf8") as glove:
        glove=glove.readlines()
    embeddings={}
    for line in glove:
        line=line.split()
        word=line[0]
        vector=np.asarray(line[1:], dtype='float32')
        embeddings[word]=vector
    words=list(get_allwords())
    count=0
    for x in words:
        if x in embeddings.keys():
            count+=1
    emb_matrix = np.zeros((count+2, embedding_size),dtype=np.float32)
    word2index = {}
    index2word = {}
    count=0
    for x in words:
        if x in embeddings.keys():
            count+=1
            word2index[x]=count
            index2word[count]=x
            emb_matrix[count]=embeddings[x]
    word2index['UNK'] = 0
    index2word[0] = 'UNK'
    emb_matrix[0]=np.random.uniform(size=embedding_size)
    word2index['STOP'] = count+1
    index2word[count+1] = 'STOP'
    # emb_matrix[-1]=np.random.uniform(size=embedding_size)
    print(emb_matrix)
    return emb_matrix, word2index, index2word

def data_split_save_no_aug():
    max_len = 300
    # embedding_model = Word2Vec.load("../data/model")
    emb_matrix, word2index, index2word = build_emb_matrix_and_vocab()

    X = get_X_index()
    X_txt=[]

    for sentence in X:
        x,l=sent2index(sentence, word2index, max_len)
        X_txt.append(x)
    print('lenx:',len(X_txt))
    # Y1,Y2,Y3 = getlabels()                                                              #3层
    Y1, Y2 = getlabels()
    print('leny:',len(Y1))
    print("save word embedding matrix ...")
    emb_filename = os.path.join(r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label', "emb_matrix_glove_300")           #修改路径
    # emb_matrix.dump(emb_filename)
    pl.dump([emb_matrix, word2index, index2word], open(emb_filename, "wb"))

def get_X_index():
    data_path = r"G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/X.txt"                                            #修改路径--------------------------
    data_X=[]
    with open(data_path,encoding='utf-8') as f:
        f=f.readlines()
        f=[text_cleaner(x) for x in f]
    print('len f:',len(f))
    for str in f:
        sentence =re.split('[ |,|(|)|\-|_|\'|/|*|&|#|:|.|\[|\]|?|^|>|<|;|+|=]',str)
        s = [x.strip() for x in sentence if not x.strip() == '']
        data_X.append(s)
    return data_X



def sent2index(sent, word2index,max_len):
    words = sent
    sent_index = [word2index[word] if word in word2index else 0 for word in words]
    l = len(sent_index)
    if len(sent_index)<max_len:
        for i in range(max_len-l):
            sent_index.append(0)
    else:sent_index=sent_index[:max_len]
    return sent_index, min(l,max_len)


def getlabels():
    path=r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label'                                                     #修改路径-------------------
    y1_path='Y1.txt'
    y2_path='Y2.txt'
    # y3_path='Y3.txt'
    y1_path_=os.path.join(path,y1_path)
    y2_path_=os.path.join(path,y2_path)
    # y3_path_ = os.path.join(path, y3_path)                                      #3层
    with open(y1_path_) as f:
        y1=f.readlines()
    y1=np.array([(int)(x.strip()) for x in y1])
    with open(y2_path_) as f:
        y2=f.readlines()
    y2=np.array([(int)(x.strip()) for x in y2])

    # with open(y3_path_) as f:
    #     y3=f.readlines()
    # y3=np.array([(int)(x.strip()) for x in y3])                                 #3层
    # return y1,y2,y3
    return y1,y2

def data_split():
    max_len = 300
    # embedding_model = Word2Vec.load("../data/model")
    emb_matrix, word2index, index2word = pl.load(open(r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/emb_matrix_glove_300','rb'))      #修改路径-------------------

    X = get_X_index()
    X_txt=[]
    X_len=[]
    for sentence in X:
        x,l=sent2index(sentence, word2index, max_len)
        X_txt.append(x)
        X_len.append(l)
    # Y1,Y2,Y3 = getlabels()
    Y1,Y2 = getlabels()                                                                                  #3层需要修改



    # print("save word embedding matrix ...")
    # emb_filename = os.path.join('../data/', "emb_matrix_glove_300")
    # emb_matrix.dump(emb_filename)
    # pl.dump([emb_matrix, word2index, index2word], open(emb_filename, "wb"))


    random.seed(2021)
    index=[i for i in range(len(Y2))]
    random.shuffle(index)
    print(max(index),len(X_txt),len(X),len(Y2))
    X_txt=[X_txt[i] for i in index]
    X_len=[X_len[i] for i in index]
    # Y3 = [Y3[i] for i in index]                                                             #3层
    Y2=[Y2[i] for i in index]
    Y1=[Y1[i] for i in index]
    train=int(len(Y2)*0.8)
    train_txt=X_txt[:train]
    test_txt=X_txt[train:]
    train_len=X_len[:train]
    test_len=X_len[train:]
    # train_y3 = Y3[:train]                                                                   #3层
    # test_y3 = Y3[train:]
    train_y2=Y2[:train]
    test_y2=Y2[train:]
    train_y1=Y1[:train]
    test_y1=Y1[train:]

    # pl.dump([train_txt,train_y1,train_y2,train_y3],open('G:\HTMC-2\DATA\processed_data\webservice\data/train_txt-len-y_300_pad0_glove','wb'))        #3层  修改路径-------------
    # pl.dump([test_txt,test_y1,test_y2,test_y3],open('G:\HTMC-2\DATA\processed_data\webservice\data/test_txt-len-y_300_pad0_glove','wb'))

    pl.dump([train_txt,train_y1,train_y2],open(r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/train_txt-len-y_300_pad0_glove','wb'))        #2层   修改路径-------------
    pl.dump([test_txt,test_y1,test_y2],open(r'G:\HTMC-2\DATA\processed_data\bestbuy\bestbuy_digital_label/test_txt-len-y_300_pad0_glove','wb'))









if __name__=='__main__':
    data_split_save_no_aug()                  #获取词嵌入 .pk  lemb_matrix_glove_300
    data_split()                                #获取训练数据 .pkl