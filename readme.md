# Hybrid Embedding-Based Text Representation for Hierarchical Multi-Label Text Classification
　　A hybrid embedding-based text representation for HMTC with high accuracy. The hybrid embedding consists of both graph embedding of categories in the hierarchy and their word embedding of category labels. The Structural Deep Network Embedding based graph embedding model is used to simultaneously encode the global and local structural features of a given category in the whole hierarchy for making the category structurally discriminable. We further use the word embedding technique to encode the word semantics of each category label in the hierarchy for making different categories semantically discriminable. 
## Requirement
Python==3.6

Tensorflow-gpu==1.14

Keras==2.2.4

Pickle

ge

## DATA
### Dataset:
DBpedia (https://www.kaggle.com/danofer/dbpedia-classes)

WOS46985 (https://data.mendeley.com/datasets/9rw3vkcfy4/2)

Amazon product reviews (https://www.kaggle.com/kashnitsky/hierarchical-text-classification)

Bestbuy (https://github.com/BestBuyAPIs/open-data-set)

Webservice (https://www.programmableweb.com/category/all/apis)


### Format:
X.txt: Texts. Each line represents a text data

Y1.txt: The labels of first layer. Each line represents a (Digitizing) label.

Y2.txt: The labels of second layer.

Y3.txt: The labels of third layer.

### The path of sample data:

[DATA/sample/X.txt], 

[DATA/sample/Y1.txt], 

[DATA/sample/Y2.txt],
 
[DATA/sample/Y3.txt]

### Feature Extraction
**Word Embedding:** Global Vectors for Word Representation (GLOVE)

glove.42B.300d

**Graph Embedding:** The Structural Deep Network Embedding (SDNE)
## How to use
### 1. Graph embedding generation：

	Run: [CODE/graphembedding/sdne_wiki.py]

**Modify:** The graph structure path, Generate graph embedding matrix. 

**Graph structure file format:** CODE/graphembedding/data/flight/brazil-airports.edgelist

### 2. Category word embedding generation：

	Run: [CODE/product_label_emb.py]
**Modify:** The category word path, Generate category word embedding matrix.

**category word embedding file format:** DATA/sample/label_1.txt
### 3. Training data & testing data generation：

    Run: [CODE/product_text_emb.py]
**Modify:** Data path, include the path of X.txt, Y1.txt and, Y2.txt.

**Result:** Get file [emb_matrix_glove_300], [train_txt-len-y_300_pad0_glove], [test_txt-len-y_300_pad0_glove], 

**Note:** when evaluating the model by 5-fold cross validation, we first combine [train_txt-len-y_300_pad0_glove] and [test_txt-len-y_300_pad0_glove], and then Perform 5-fold cross validation.
### 4. Start training

Take our hybrid Embedding method as example.

**·Layer 1:**

    Run: [CODE/5FOLD/3layer/gru_layer1.py]
    
**Modify:** [class_num], [dataset path], [graph embedding path], [label word embedding path], [the path to store the model and predict results].

**·Layer 2:**

    Run: [CODE/5FOLD/3layer/gru_layer2_we_ge.py]
    
**Modify:** [class_f (the number of parent layer class)], [class_num], [dataset path], [graph embedding path], [label word embedding path], [the path to store the model and predict results].

**·Layer 3:**

Same operation as the second layer.
### 5. Save result

　　The training results are save in [DATA/result/result.txt], meanwhile, For the convenience of viewing the results, we also display the training results on the console.






