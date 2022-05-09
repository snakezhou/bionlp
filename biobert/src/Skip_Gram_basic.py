# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 15/05/2021 20:43
@Author: yao
"""

import logging

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as opt

import matplotlib
# Don't show graph.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Step 1: Define hyper-parameters
class config:
    def __init__(self):
        # data parameters
        self.data_path = '../data/litcovid.sentence.txt'

        self.vocab_size = 50000

        self.low_case = False
        self.unk_token = '[UNK]'
        self.linker = '&&&'

        self.token_fre_file = '../data/litcovid.TokenFrequency.txt'
        self.token_fre_low_file = '../data/litcovid.TokenFrequency.low.txt'

        # logging parameters
        self.save_log = True
        self.log_level = logging.INFO
        self.log_file = '../model/litcovid.log'

        # loader parameters
        self.window_size = 5 # 3-5
        self.train_size = 200000

        self.batch_size = 128
        self.shuffle = True

        # model parameters
        self.embedding_size = 100

        # training parameters
        self.use_cpu = True
        self.device = 'cuda' if torch.cuda.is_available() and (not self.use_cpu) else 'cpu'

        self.epoch = 20
        self.print_step = 50
        self.learning_rate = 0.03

        self.save_embedding = True
        self.embedding_save_path = '../model/litcovid.skip-gram.embedding.txt'

        # TSNE parameters
        self.plot_only = 500
        self.tsne_save_file = '../model/litcovid.tsne.png'

args = config()

logger = logging.getLogger(__name__)
if args.save_log:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=args.log_level,
                        filename=args.log_file,
                        filemode='w')
else:

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO, )

# Step 2: Load data
class SkipGram_Dataset(Dataset):
    def __init__(self, paras):
        self.data_path = paras.data_path

        self.vocab_size = paras.vocab_size

        self.unk_token = paras.unk_token
        self.low_case = paras.low_case
        self.linker = paras.linker

        self.train_size = paras.train_size

        if self.low_case:
            self.token_fre_file = paras.token_fre_file
        else:
            self.token_fre_file = paras.token_fre_low_file

        # 2 * window_size + 1
        self.window_size = paras.window_size

        self.word2idx = {self.unk_token: 0}
        self.idx2word = {0: self.unk_token}

        self.token_fre = {}
        self.vocab = set()
        
        self.data = []
        self.center = []
        self.target = []

        logging.info(f'Load token frequency from {self.token_fre_file}.')
        self.read_token_fre()
        logging.info(f'Load data from {self.data_path}.')
        self.read_data()

        logging.info(f'Data size: {len(self.data):,}.')

    def read_token_fre(self):

        with open(self.token_fre_file, encoding='utf-8') as f:
            for line in f:
                token, fre = line.strip().split('\t')
                self.token_fre[token] = int(fre)

            token_sort = sorted(self.token_fre, key=lambda x:self.token_fre[x],
                                reverse=True)
            # less unk_token
            self.vocab = token_sort[:self.vocab_size-1]

            for word in self.vocab:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word

    def read_data(self):
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                token_list = line.strip().split('\t')
                if self.low_case:
                    token_list = [token.lower() for token in token_list]

                for idx, token in enumerate(token_list):
                    left_start = idx - self.window_size if idx - self.window_size > 0 else 0
                    right_end = idx + self.window_size if idx+self.window_size < len(token_list) else len(token_list)
                    for left_token in token_list[left_start: idx]:
                        self.data.append(f'{token_list[ idx ]}{self.linker}{left_token}')
                    if idx == len(token_list)-1:
                        continue
                    for right_token in token_list[idx: right_end]:
                        self.data.append(f'{token_list[ idx ]}{self.linker}{right_token}')

        if self.train_size and len(self.data) > self.train_size:
            self.data = random.sample(self.data, self.train_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

litcovid_dataset = SkipGram_Dataset(args)
vocab = litcovid_dataset.vocab
word_to_idx = litcovid_dataset.word2idx
idx_to_word = litcovid_dataset.idx2word
litcovid_dataloader = DataLoader(litcovid_dataset, batch_size=args.batch_size,
                                 shuffle=args.shuffle)

# input('Please press enter to continue.')
"""
Sentence:
    Pneumonia of unknown aetiology in Wuhan China potential for
    international spread via commercial air travel.
Central word：
    Wuhan
Window Size: 
    5
Left Context:
    Pneumonia, of, unknown, aetiology, in
Right Context:
    China, potential, for, international, spread
Return data:
    Wuhan, Pneumonia
    Wuhan, of
    Wuhan, unknown
    Wuhan, aetiology
    Wuhan, in
    Wuhan, China
    Wuhan, potential
        ......
"""

# Step 3: Build Skit-Gram model.
class SkipGram(nn.Module):
    def __init__(self, paras):
        super().__init__()

        self.vocab_size = paras.vocab_size
        self.embedding_size = paras.embedding_size

        # W_{vd}
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # output layer W'_{dv}
        self.out_layer = nn.Linear(self.embedding_size, self.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_idx):
        input_embedding = self.embedding(input_idx)
        predict = self.out_layer(input_embedding)
        log_probability = self.log_softmax(predict)
        return log_probability

model = SkipGram(args).to(args.device)
logger.info(model)
# input('Please press enter to continue.')

# Step 4: Training.
def batch_process(word2idx: dict, _batch_data, linker: str,
                  unk_token: str='[UNK]', return_tensor:bool=True):

    center_list = []
    target_list = []
    for _batch in _batch_data:
        center, target = _batch.split(linker)
        center_idx = word2idx[center] if word2idx.get(center) else word2idx[unk_token]
        target_idx = word2idx[target] if word2idx.get(target) else word2idx[unk_token]
        center_list.append(center_idx)
        target_list.append(target_idx)

    if return_tensor:
        return torch.tensor(center_list), torch.LongTensor(target_list)
    else:
        return center_list, target_list

def tensor_to_list(tensor, str_instance: bool=False):
    if str_instance:
        return list(map(str, tensor.numpy().tolist()))
    else:
        return tensor.numpy().tolist()

def save_embedding(best_model: torch.nn.Module, idx2word: dict,
                   save_file: str):
    best_model.to('cpu')
    embedding_matrix = best_model.embedding.weight.data
    with open(save_file, 'w', encoding='utf-8') as wf:
        for idx in range(embedding_matrix.shape[0]):
            word = idx2word[idx]
            embedding_wf = ' '.join(tensor_to_list(embedding_matrix[idx], True))
            wf.write(f'{word}\t{embedding_wf}\n')


if args.device == 'cuda':
    logger.info('Training with GPU.')
else:
    logger.info('Training with CPU.')

nll_loss = nn.NLLLoss()
adam_optimizer = opt.Adam(model.parameters(), lr=args.learning_rate)

best_loss = 1e5
logger.info('Start training.')
for epoch in range(args.epoch):
    epoch_loss = 0
    for step, batch in enumerate(litcovid_dataloader):
        adam_optimizer.zero_grad()
        batch_data, batch_label = batch_process(word_to_idx, batch,
                                                args.linker, args.unk_token,
                                                True)

        batch_data.to(args.device)
        batch_label.to(args.device)

        log_pro = model(batch_data)
        loss = nll_loss(log_pro, batch_label)

        loss.backward()
        adam_optimizer.step()

        epoch_loss += loss.item()
        if step % args.print_step == 0:
            logger.info(f'epoch: {epoch}, step: {step}, loss: {loss.item():.4f} ')

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        logger.info(f'Update best epoch loss: {epoch_loss:.4f}.')
        logger.info(f'Update Embedding file: {args.embedding_save_path}.')
        save_embedding(model, idx_to_word, args.embedding_save_path)

logger.info('Training Done.')

#input('Please press enter to continue.')

# Step 5: Visualize the embeddings by TSNE.
logger.info(f'Loading Embedding from {args.embedding_save_path}.')
def read_embedding(embedding_file: str, return_tensor: bool=True):
    token_list = []
    embedding_list = []
    with open(embedding_file, encoding='utf-8') as f:
        for line in f:

            word, embedding = line.strip().split('\t')
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
logger.info(f'Visualizing.')
plot_with_labels(low_dim_embeddings, labels, filename=args.tsne_save_file)
logger.info(f'TSNE visualization is completed, saved in {args.tsne_save_file}.')
