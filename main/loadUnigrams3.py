import time
import torch
from google_ngram_downloader import readline_google_store
from string import punctuation
import re
import csv
from SpellCorrect import spellCorrect
import sys
sys.path[0:0] = ['../model']
from Utils import read_corpus, CharDataset
from Model import CharLM
from Vocab import Vocabulary

import pandas as pd
import os
import sys
# punct = punctuation+'«»—…“”*–'
# if len(record.ngram.strip(punct)) > 2
'''
A method to load Google ngrams in csv files tables.
Table structure depends on the year boundary
Params: nram length, language, index (the first letter)
'''

mapping = {'а':'a',
       'я':'a',
       'ь':'a',
       'б':'b',
       'ц':'c',
       'ч':'c',
       'д':'d',
       'е':'e',
       'э':'e',
       'ф':'f',
       'г':'g',
       'х':'h',
       'и':'i',
       'й':'j',
       'к':'k',
       'л':'l',  
       'м':'m',
       'н':'n',
       'о':'o',
       'п':'p',
       '':'q',
       'р':'r',
       'с':'s',
       'ш':'s',
       'щ':'s',
       'т':'t',
       'у':'u',
       'в':'v',
       '':'w',
       '':'x',
       '':'y',
       'з':'z',
       'ж':'z'}



data = read_corpus('../data/vocabulary.txt')
V = Vocabulary(data)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = CharLM(V.vocab_size, word_len=V.pad_len, emb_dim=128, hidden_size=128)
#     model = CharLM(V)
model.to(device)
model_filename = "old_rus_lm.pth"
print('Loading a model')
model.load_state_dict(torch.load('../data/' + model_filename, map_location={'cuda:0': 'cpu'}))
print('Done')

correct = spellCorrect(V=V, model=model)

def normalize(ngram):
    '''Rid of tags'''
    if '_' in ngram:
        index = ngram.find('_')
        ngram = ngram[: index]
        
    return(ngram)

indices = ''

for file in os.listdir('../'):
    if 'google' in file:
        dfall = pd.read_csv('../'+file, sep='\t', header=None)
        dfall.columns = ['ngram', 'year', 'match_count', 'volume_count']
        my_indices = file[-1]
        e = 0
        count = 0
        ngram = ''
        df = dfall[dfall.year<1918]
        print('length of', my_indices, len(df))
        for i in df.index:
    #         if e>-1:
            new_idx = my_indices
            if df.ngram[i] == ngram:
                new_idx = indices
                df.at[i, 'my_indices']= my_indices
                df.at[i, 'normalized']= normalized
                df.at[i, 'new_idx']= new_idx
                df.at[i, 'is_bastard']= is_bastard
                df.at[i, 'new_ngram']=  new_ngram
    #             writer.writerow([my_indices,
    #                          ngram,
    #                          normalized,
    #                          record.year,
    #                          record.match_count,
    #                          record.volume_count,
    #                          new_idx, #new_idx
    #                         is_bastard, #is_bastard
    #                         new_ngram])
                e += 1
                if e%10000==0:
                    print('loaded: ' + str(e)) # отладка
            else:
                ngram = str(df['ngram'][i])
                normalized = normalize(ngram)
                new_ngram = correct.to_check(normalized, 
                                     seqprob=False, 
                                     upper_boundary=0.0008771931170485914, 
                                     lower_boundary=0.0003082964103668928)
                if new_ngram!=normalized:
                    is_bastard = True
                else:
                    is_bastard = False
                    if new_ngram[0]!=normalized[0]:
                        if new_ngram[0].lower() in mapping.keys():
                            new_idx = mapping[new_idx]
                            indices = new_idx
                        else:
                            indices = my_indices
                df.at[i, 'my_indices']= my_indices
                df.at[i, 'normalized']= normalized
                df.at[i, 'new_idx']=  new_idx
                df.at[i, 'is_bastard']= is_bastard
                df.at[i, 'new_ngram']= new_ngram
    #             writer.writerow([my_indices,
    #                              ngram,
    #                              normalized,
    #                              record.year,
    #                              record.match_count,
    #                              record.volume_count,
    #                              new_idx, #new_idx
    #                             is_bastard, #is_bastard
    #                             new_ngram]) #new_ngram]) 
#                         else:
                e += 1
                if e%10000==0:
                    df.to_csv('./unigrams_'+my_indices+'.tsv', sep='\t', index=False, header=None)
                    print(df.head())
                    print(sys.exc_info())
                    print('loaded: ' + str(e))
                
        df.to_csv('unigrams_'+my_indices+'.tsv', sep='\t', index=False, header=None)
        print(df.tail())

