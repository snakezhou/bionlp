# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 16/05/2021 12:54
@Author: yao
"""

import logging

import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib
# Don't show graph
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from transformers import BertTokenizer, BertModel

# Step 1: Define hyper-parameters
class config:
    def __init__(self):
        # data parameters
        self.vocab_size = 50000

        self.low_case = False

        self.token_fre_file = '../data/result4.txt'
        self.token_fre_low_file = '../data/result5.low.txt'#统一大小写后的是.low

        # Model parameters
        #self.model_name = 'dmis-lab/biobert-base-cased-v1.1'
        self.model_name = 'mervenoyan/PubMedBERT-QNLI'


        # logging parameters
        self.save_log = False
        self.log_level = logging.INFO
        self.log_file = '../model/litcovid.biobert.log'

        # loader parameters
        self.batch_size = 128

        # training parameters
        self.use_cpu = True
        self.device = 'cuda' if torch.cuda.is_available() and (not self.use_cpu) else 'cpu'

        self.print_step = 20
        self.save_embedding = True
        self.embedding_save_path = '../model/litcovid.bio-bert.embedding.txt'

        # TSNE parameters
        self.plot_only = 500
        self.tsne_save_file = '../model/litcovid.tsne.bio-bert.png'

args = config()

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
class Bert_Dataset(Dataset):
    def __init__(self, paras):
        self.vocab_size = paras.vocab_size

        self.low_case = paras.low_case

        if self.low_case:
            self.token_fre_file = paras.token_fre_file
        else:
            self.token_fre_file = paras.token_fre_low_file

        self.vocab = set()
        self.token_fre = {}

        self.data = []
        logging.info(f'Loading data from {self.token_fre_file}')
        self.read_token_fre()
        logging.info(f'Data size: {len(self.data):,}.')

    def read_token_fre(self):
        with open(self.token_fre_file, encoding='utf-8') as f:
            for line in f:
                word, fre = line.strip().split('\t')
                self.token_fre[word] = int(fre)

        token_sort = sorted(self.token_fre, key=lambda x:self.token_fre[x],
                                reverse=True)

        self.vocab = token_sort[:self.vocab_size]

        self.data = list(self.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

litcovid_dataset = Bert_Dataset(args)
vocab = litcovid_dataset.vocab
litcovid_dataloader = DataLoader(litcovid_dataset, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)
"""
Batch Data:
'the', 'of', 'and', 'in', 'to', 'a', 'with', 'for',
'covid-19', 'patients', 'were', 'is', 'on', 'was', 
'that', 'as', 'pandemic', 'by', 'from', 'are', 'be',
"""
# input('Please press enter to continue.')

# Step 3: Model initialization
model = BertModel.from_pretrained(args.model_name)
tokenizer = BertTokenizer.from_pretrained(args.model_name)
model.to(args.device)
logging.info(model)
"""
Token:
    disease
token_input:
    token_input = tokenizer(token, return_tensors='pt')
    {'input_ids': [101, 4302, 102],
    'token_type_ids': [0, 0, 0],
    'attention_mask': [1, 1, 1]}
token_decode:
    token_decode = tokenizer.decode(token_input[input_ids'])
    '[CLS] disease [SEP]'
"""
# input('Please press enter to continue.')

# Step 4: Embedding Generation
def tensor_to_list(tensor, str_instance: bool=False):
    if str_instance:
        return list(map(str, tensor.numpy().tolist()))
    else:
        return tensor.numpy().tolist()

logging.info('Embedding Generating.')
save_count = 0
wf = open(args.embedding_save_path, 'w', encoding='utf-8')
with torch.no_grad():
    for step, batch_token in enumerate(litcovid_dataloader):

        if step % args.print_step == 0:
            logging.info(f'step: {step}, {save_count} embedding saved.')
        save_count += len(batch_token)
        encoded_input = tokenizer(batch_token, return_tensors='pt',
                                  padding=True)
        encoded_input = encoded_input.to(args.device)

        output = model(**encoded_input, output_hidden_states=True,
                       output_attentions=True)
        # batch_size, seq_length, embedding_size
        last_hidden_state = output['last_hidden_state']
        for idx, token in enumerate(batch_token):
            token_cls_embedding = last_hidden_state[idx][0]

            embedding_wf = ' '.join(tensor_to_list(token_cls_embedding, True))
            wf.write(f'{token}\t{embedding_wf}\n')

logging.info(f'Embedding save done, {save_count} embeddings saved.')
wf.close()

# Step 5: Visualize the embeddings by TSNE.
logging.info(f'Loading Embedding from {args.embedding_save_path}.')
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
logging.info(f'Visualizing.')
plot_with_labels(low_dim_embeddings, labels, filename=args.tsne_save_file)
logging.info(f'TSNE visualization is completed, saved in {args.tsne_save_file}.')
