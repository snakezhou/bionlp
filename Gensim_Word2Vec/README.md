# Embedding-experiment-in-BioNLP-course


## 1. Data preparation. 

```
unzip data/litcovid-trainingdata.zip
```

## 2. Environmental preparation. 
If pytorch installation fails, please use the official website command line to install.
```
pip3 install -r requirements.txt
```

## 3. Quick start. 
## 3.1 Word2Vec-base. 

You can pursue a better embedding effect by modifying the parameter part of the code, such as train_size, embedding_size, batch_size, learning_rage, etc.
```
python3 Skip_Gram_basic.py
```

## 3.2 Bert  
```
python3 Bert_4_Litcovid_WordEmbedding.py
```
