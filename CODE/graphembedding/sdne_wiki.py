
import numpy as np
import pickle as pl
from ge.classify import read_node_label, Classifier
from ge import SDNE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1],
                    label=c)  # c=node_colors)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    G = nx.read_edgelist('../data/dbpedia/ge_label2-3.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = SDNE(G, hidden_size=[256, 300],)
    model.train(batch_size=3000, epochs=200, verbose=2)
    embeddings = model.get_embeddings()
    # print(embeddings['0'])
    #
    #
    # print(len(embeddings))
    # print(type(embeddings))
    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)

    label = []
    emb = []
    # label, emb=pl.load(open('../data/WOS5736/wos5736_label1-2','rb'))
    for i in range(219):
        label.append(i)
        emb.append(embeddings[str(i)])
    # print(emb)
    print(label)
    print(len(emb))
    pl.dump([label,emb],open('../data/dbpedia/dbpedia_label2-3','wb'))    #label,emb 保存各个标签的图嵌入向量

