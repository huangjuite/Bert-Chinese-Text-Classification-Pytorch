# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
# from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
import os

class Config(object):

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                
        self.dev_path = dataset + '/data/dev.txt'                                    
        self.test_path = dataset + '/data/test.txt'                                  
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                
        try:
            os.mkdir(dataset + '/saved_dict/')
        except:
            pass
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        
        self.result_path = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        self.require_improvement = 1000                                 
        self.num_classes = len(self.class_list)                         
        self.num_epochs = 3                                             
        self.batch_size = 128                                          
        self.pad_size = 64                                              
        self.learning_rate = 5e-5                                  
        self.bert_path = './chinese_wwm_ext_pytorch'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
