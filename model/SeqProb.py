import sys
from functools import reduce

def seq_prob(model, dataset, vocab):
    for i in range(10):
        x, y, word_len = dataset[i]
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)[0]
        probs = []
        output = vocab.transform_vec(y[:word_len])
        for char_idx in range(1,word_len[0]):
            char = output[char_idx]
            if char=='<':
                break
            idx = vocab.char2idx[char]
            probs.append(y_pred[char_idx][idx])
            multiplication = reduce(lambda x, y: x*y, probs)
        return multiplication/(word_len-1)
