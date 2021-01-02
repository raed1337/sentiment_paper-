# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:27:14 2020

@author: moham
"""
import numpy as np
import pandas as pd
import glob
import random
from tqdm import tqdm, trange
import time

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from Utilities import  *

from transformers import BertTokenizer
from transformers import XLNetTokenizer
from transformers import RobertaTokenizer
from transformers import AlbertTokenizer
from transformers import AdamW

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

torch.cuda.empty_cache()

# Hyperparameters
MAX_LEN=256
BATCH_SIZE=16
EPOCHS=4
LEARNING_RATE=2e-5


#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertForSequenceClassification,BertTokenizer,'bert-base-cased','bert'),
          (XLNetForSequenceClassification, XLNetTokenizer,'xlnet-base-cased','xlnet'),
          (RobertaForSequenceClassification, RobertaTokenizer,'roberta-base','Roberta'), 
          (AlbertForSequenceClassification, AlbertTokenizer,'albert-base-v1','albert')
          ]

MODEL_NAMES = ['bert', 'xlnet', 'Roberta', 'albert']
def read_data(filenamePattern):
    all_files = glob.glob("{}.csv".format(filenamePattern))

    li = []    
    for filename in all_files:
    
        df = pd.read_csv(filename, index_col=0 , header=0)
        li.append(df)
       
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.set_index('File')

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

seed_torch(42)

#cross validation 
kf = 10







# use your path


df  = read_data("DatasetSenti4SDSO_test_*")
df['ManualLabel']=df['ManualLabel'].replace({'o': 0, 'p': 1,'n':2})
final_pred = pd.Series()
dfs = []




for j in range(4):
    
    m_num=j
    cur_model=MODELS[m_num]
    m_name=MODEL_NAMES[m_num]
    df_model = pd.DataFrame()
        
    for i in range(kf):
        test_df  = df[df.index == "DatasetSenti4SDSO_test_{}".format(i)]
        train_df = df[df.index != "DatasetSenti4SDSO_test_{}".format(i)]
        
        tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)
    
        sentences=train_df.Sentence.values
        numericuls = train_df[['Senti4SD','SentiCR','SentistrengthSE','POME','DsoLabelFullText']].to_numpy()
        labels=train_df.ManualLabel.values
        input_ids = []
        attention_masks = []
        input_numerciculs=[]
        
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(numericuls)
        
        for sent,num in zip(sentences,numericuls):
            encoded_dict = tokenizer.encode_plus(
                                str(sent), 
                                add_special_tokens = True, 
                                max_length = MAX_LEN,
                                pad_to_max_length = True,
                                return_attention_mask = True, 
                                return_tensors = 'pt',
                                truncation=True
                            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

            num = np.array(num).reshape(1,5)
            input_numerciculs.append(torch.from_numpy(enc.transform(num).toarray()))
            
        train_inputs = torch.cat(input_ids, dim=0)
        train_numericul = torch.cat(input_numerciculs, dim=0)
        train_masks = torch.cat(attention_masks, dim=0)
        train_labels = torch.tensor(labels)
        
        print('Training data {} {} {} {}'.format(train_inputs.shape, train_masks.shape, train_labels.shape,train_numericul.shape))
        
        
        train_data = TensorDataset(train_inputs, train_masks, train_labels,train_numericul)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
        
        # Train Model
        cur_model[0].num_size = train_numericul.shape[1]
        model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
        model.cuda()
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
              'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
              'weight_decay_rate': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
        
        begin=time.time()
        train_loss_set = []
        
        for _ in trange(EPOCHS, desc="Epoch"): 
            model.train()
        
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(train_dataloader):
            
                batch = tuple(t.to(device) for t in batch)
              
                b_input_ids, b_input_mask, b_labels,b_input_numericul = batch
                optimizer.zero_grad()
                # Forward pass
                outputs = model(b_input_ids, token_type_ids=None, \
                                attention_mask=b_input_mask, labels=b_labels,input_numericul=b_input_numericul)
                loss = outputs[0]
                logits = outputs[1]
                train_loss_set.append(loss.item())    
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
        
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
        
        end=time.time()
        print('Training used {} second'.format(end-begin))
        
        # Train Model
        begin=time.time()    
        sentences=test_df.Sentence.values
        labels = test_df.ManualLabel.values
        numericuls = test_df[['Senti4SD','SentiCR','SentistrengthSE','POME','DsoLabelFullText']].to_numpy()
        
        input_ids = []
        attention_masks = []
        input_numerciculs=[]
        
        for sent ,num in zip(sentences,numericuls):
            encoded_dict = tokenizer.encode_plus(
                            str(sent), 
                            add_special_tokens = True, 
                            max_length = MAX_LEN,
                            pad_to_max_length = True,
                            return_attention_mask = True, 
                            return_tensors = 'pt'
                            )
             
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            num = np.array(num).reshape(1,5)
            input_numerciculs.append(torch.from_numpy(enc.transform(num).toarray()))
        
        prediction_inputs = torch.cat(input_ids,dim=0)
        prediction_masks = torch.cat(attention_masks,dim=0)
        prediction_labels = torch.tensor(labels)
        prediction_numericul = torch.cat(input_numerciculs, dim=0)
        
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels,prediction_numericul)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)
        
        model.eval()
        predictions,true_labels=[],[]
        
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels,b_input_numericul = batch
        
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,input_numericul=b_input_numericul)
                logits = outputs[0]
        
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            predictions.append(logits)
            true_labels.append(label_ids)
            
        end=time.time()
        print('Prediction used {:.2f} seconds'.format(end-begin))
        
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        
        print("Accuracy of {} on Senti4SD is: {}".format(m_name, accuracy_score(flat_true_labels,flat_predictions)))
        print(classification_report(flat_true_labels, flat_predictions))
        test_df[cur_model[3]] = flat_predictions
        df_model = df_model.append(test_df)
        torch.cuda.empty_cache()
    df_model.reset_index(inplace=True)   
    dfs.append(df_model)
df_final =pd.concat(dfs, axis=1, join='inner')

df_final = df_final.loc[:,~df_final.columns.duplicated()]
df_final.to_excel('DatasetSenti4SDSO.xlsx',index=False)
    
