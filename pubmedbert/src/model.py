from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer

#model_name = 'hfl/chinese-roberta-wwm-ext'
model_name = '/store/sjzhou/anaconda3/envs/tensorflow/lib/python3.6/site-packages/transformers/bert−base−uncased'

config = BertConfig.from_pretrained(model_name)	# 这个方法会自动从官方的s3数据库下载模型配置、参数等信息（代码中已配置好位置）
tokenizer = BertTokenizer.from_pretrained(model_name)		 # 这个方法会自动从官方的s3数据库读取文件下的vocab.txt文件
model = BertModel.from_pretrained(model_name)		# 这个方法会自动从官方的s3数据库下载模型信息
