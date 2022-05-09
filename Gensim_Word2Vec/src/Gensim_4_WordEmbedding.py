#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 19/11/2020 14:45 
@Author: XinZhi Yao
Documentation: https://radimrehurek.com/gensim/models/word2vec.html
"""
import torch
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

#LineSentence是按行读取文件中的每一行

# 1. data path
corpus_file = '../data/result.txt'
model_save_file = '../data/Gensim.model.bin'
embedding_save_file = '../data/Gensim.embedding.txt'

# 2. Define hyper-parameters
# Dimensionality of the word vectors.
size = 100
# Maximum distance between the current and predicted word within a sentence.
window = 5
# Ignores all words with total frequency lower than this.
min_count = 5
# Training algorithm: 1 for skip-gram; otherwise CBOW.
sg = 1
# The initial learning rate.
alpha = 0.02
# Number of iterations (epochs) over the corpus.
epochs = 2
# Target size (in words) for batches of examples passed to worker threads.
batch_words = 10000

# 3. Training model
model = Word2Vec(LineSentence(corpus_file), vector_size=size, window=window,
                 min_count=min_count, sg=sg, alpha=alpha, epochs=epochs,
                 batch_words=batch_words)

# 4. Save Model and Embedding.
# full Word2Vec object state
model.save(model_save_file)
# just the KeyedVectors.
model.wv.save_word2vec_format(embedding_save_file, binary=False)




import matplotlib
# Don't show graph
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class config:
    def __init__(self):
        self.embedding_save_path = '../data/Gensim.embedding.txt'
        self.plot_only = 500
        self.tsne_save_file='../data/Gensim.png'
args = config()

def read_embedding(embedding_file: str, return_tensor: bool=True):
    token_list = []
    embedding_list = []
    line_num = 0 
    with open(embedding_file, encoding='utf-8') as f:
        for line in f:
            line_num += 1 
            if (line_num != 1): 
                word, embedding = line.strip().split(' ',1)
                token_list.append(word)
                embedding_list.append(list(map(float, embedding.split(' '))))

    if return_tensor:
        return token_list, torch.tensor(embedding_list)
    else:
        return token_list, embedding_list

def plot_with_labels(embeddings, nodes, filename='tsne.png'):
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(nodes):
    x, y = embeddings[ i, : ]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.savefig(filename)

word_list, final_embedding = read_embedding(args.embedding_save_path)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embeddings = tsne.fit_transform(final_embedding[ :args.plot_only, : ])
labels = word_list.copy()[:args.plot_only]
print('Visualizing.')
plot_with_labels(low_dim_embeddings, labels, filename=args.tsne_save_file)
print('TSNE visualization is completed, saved in {args.tsne_save_file}.')
