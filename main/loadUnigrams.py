### TODO: define boundaries, model, etc
### condition if word == the word before

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
# punct = punctuation+'«»—…“”*–'
# if len(record.ngram.strip(punct)) > 2
'''
A method to load Google ngrams in csv files tables.
Table structure depends on the year boundary
Params: nram length, language, index (the first letter)
'''

def normalize(ngram):
    '''Rid of tags'''
    if '_' in ngram:
        index = ngram.find('_')
        ngram = ngram[: index]
        
    return(ngram)

def load_ngrams(my_len, my_lang, my_indices, before_1918=False):
    fname, url, records = next(readline_google_store(ngram_len=my_len, lang=my_lang, indices=my_indices))
    record = next(records)
    e = 0
    count = 0
    with open('unigrams_' + my_index + '.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        while True:
            try:
                if before_1918:
                    if record.year < 1918:
                        record = next(records)
                    elif record.year >= 1918:
                        ngram = record.ngram
                        normalized = normalize(ngram)
                        writer.writerow([my_indices,
                                         ngram,
                                         normalized,
                                         record.year,
                                         record.match_count,
                                         record.volume_count]) #(idx, raw_n_gram, n_gram, year, match_count, volume_count)"
                        
                else:
                    if record.year < 1918:
                        record = next(records)
                    elif record.year >= 1918:
                        ngram = record.ngram
                        normalized = normalize(ngram)
                        new_ngram = to_check(normalized, 
                                             seqprob=False, 
                                             upper_boundary=0.0008771931170485914, 
                                             lower_boundary=0.0003082964103668928)
                        writer.writerow([my_indices,
                                         ngram,
                                         normalized,
                                         record.year,
                                         record.match_count,
                                         record.volume_count,
                                         '', #new_idx
                                        False, #is_bastard
                                        new_ngram]) #new_ngram]) 
                        #`idx`, `raw_n_gram`, `n_gram`, `year`, `match_count`, `volume_count`, `new_idx`, `is_bastard`, `new_ngram`
                if e%1000==0:
                    print('loaded: ' + str(e)) # отладка
                e += 1
                record = next(records)
            except StopIteration:
                break
                print("StopIteration")
                                         
    print(str(count) + " " + my_indices + " ngrams saved")
    return 0


def main():
    
    data = read_corpus('../data/vocabulary.txt')
    V = Vocabulary(data)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
#     model = CharLM(V.vocab_size, word_len=V.pad_len, emb_dim=128, hidden_size=128)
    model = CharLM(V)
    model.to(device)
    model_filename = "old_rus_lm.pth"
    print('Loading a model')
    model.load_state_dict(torch.load('../data/' + model_filename, map_location={'cuda:0': 'cpu'}))
    print('Done')
    
    correct = spellCorrect()

    unigram_indices = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                       'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                       'u', 'v', 'w', 'x', 'y', 'z', 'other']
                                         
                                         
    for idx in unigram_indices:
        load_unigrams(idx)
        
    return 0

if __name__ == '__main__':
    main()

