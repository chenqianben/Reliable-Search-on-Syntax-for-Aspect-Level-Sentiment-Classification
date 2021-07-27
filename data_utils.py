import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
from pytorch_pretrained import BertTokenizer

class Vocab(object):
    ''' vocabulary for tokens '''
    def __init__(self, vocab_list, add_pad=True, add_unk=True):
        self._vocab_dict = dict()         #key是词，value是位置id
        self._reverse_vocab_dict = dict()   #相反
        self._length = 0
        if add_pad: # pad_id should be zero (for masks)
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w
    
    def word_to_id(self, word): # transform tokens to indices
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, idx): # transform indices to tokens
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(idx, self.unk_word)
        return self._reverse_vocab_dict[idx]
    
    def has_word(self, word):
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length
    

class Tokenizer(object):
    ''' transform tokens to indices and pad sequences '''
    def __init__(self, word_vocab, pos_vocab, deprel_vocab, max_slen, max_plen, lower):
        self.vocab = {
            'word': word_vocab, # vocabulary for words
            'pos': pos_vocab, # vocabulary for part-of-speech tags
            'deprel': deprel_vocab # vocabulary for dependency relations
        }
        self.max_slen = max_slen # maximum length of sentences
        self.max_plen = max_plen # maximum length of dependency paths
        self.lower = lower # whether lowering the words in sentences
    
    @classmethod
    def from_files(cls, fnames, lower=True):
        berttokenizer = BertTokenizer.from_pretrained('pretrained_bert')
        all_tokens, all_pos, all_deprel = set(), set(), set() # parpare for vocabularies
        max_slen, max_plen = 0, 0
        for fname in fnames:
            fdata = json.load(open(fname, 'r', encoding='utf-8'))
            for data in fdata:
                data_str = ' '.join(data['token'])
                text = berttokenizer.tokenize(data_str)
                
                all_tokens.update(map(lambda x: x.lower() if lower else x, data['token']))
                all_pos.update(data['pos'])
                all_deprel.update(data['deprel'])
                max_slen = max(max_slen, len(text))
                max_plen = max(max_plen, max([len(p)+1 for aspect in data['aspects'] for p in aspect['path']])) # path length should add 1 for extra relation
        all_deprel.update([f"rev#{rel}" for rel in all_deprel])
        word_vocab, pos_vocab, deprel_vocab = Vocab(all_tokens), Vocab(all_pos), Vocab(all_deprel)
        return cls(word_vocab, pos_vocab, deprel_vocab, max_slen, max_plen, lower)
    
    @staticmethod
    def _pad_sequence(sequence, pad_id, maxlen, dtype, padding, truncating):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x
    
    def to_sequence(self, tokens, vocab_name, maxlen=None, reverse=False, dtype='int64', padding='post', truncating='post'):
        if vocab_name == 'word' and self.lower:
            tokens = map(lambda x: x.lower(), tokens)
        sequence = list(map(self.vocab[vocab_name].word_to_id, tokens))
        pad_id = self.vocab[vocab_name].pad_id
        if maxlen is None:
            maxlen = self.max_slen
        if reverse:
            sequence.reverse()
        return Tokenizer._pad_sequence(sequence, pad_id, maxlen, dtype, padding, truncating)
    
    def pad_sequence(self, sequence, pad_id, maxlen=None, reverse=False, dtype='int64', padding='post', truncating='post'):
        if dtype == 'int64':
            sequence = list(map(int, sequence))
        elif dtype == 'float32':
            sequence = list(map(float, sequence))
        if maxlen is None:
            maxlen = self.max_slen
        if reverse:
            sequence.reverse()
        return Tokenizer._pad_sequence(sequence, pad_id, maxlen, dtype, padding, truncating)

    
class MyDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, berttokenizer, mytokenizer, mode):
        data_file = os.path.join('dats', os.path.join(mode, os.path.split(fname)[-1].replace('.json', '.dat'))) # dataset cache
        vocab = self.load_vocab("pretrained_bert/vocab.txt")
        if os.path.exists(data_file):
            print(f"loading dataset: {data_file}")
            dataset = pickle.load(open(data_file, 'rb'))
        else:
            print('building dataset...')
            dataset = self._read_data(fname, berttokenizer, mytokenizer)
            pickle.dump(dataset, open(data_file, 'wb'))
        self._dataset = dataset

    def load_vocab(self, vocab_file):
        vocab = []
        with open(vocab_file, 'r', encoding='utf-8') as of:
            for line in of.readlines():
                vocab.append(line.rstrip('\n'))
        vocab = {v:i for i,v in enumerate(vocab)}
        return vocab
    
    @staticmethod
    def _read_data(fname, berttokenizer, mytokenizer):
        dataset = list()
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        fdata = json.load(open(fname, 'r', encoding='utf-8'))
        for data in fdata:
            text_mask = len(data['token'])*[1] + (mytokenizer.max_slen - len(data['token']))*[0]
            text_mask = np.asarray(text_mask, dtype='int64')

            data_str = ' '.join(data['token'])
            textp = berttokenizer.tokenize(data_str)
            textp = ['[CLS]'] + textp  # 放在最后一步加了CLS，在使用的时候从位置一开始读取就可
            textp_id = berttokenizer.convert_tokens_to_ids(textp)
            textp_id = textp_id + (mytokenizer.max_slen + 1 - len(textp_id))*[0] # +1是考虑了cls最后在model处理的时候要去掉
            textp_id = np.asarray(textp_id, dtype='int64')

            wordpiece_mask = np.sum(textp_id != 0)*[1] + (mytokenizer.max_slen + 1 - np.sum(textp_id != 0))*[0]
            wordpiece_mask = np.asarray(wordpiece_mask, dtype='int64')
            wordpiece_gather_idx = get_wordpiece_gather_idx(textp, data['token'], mytokenizer.max_slen)

            pos = mytokenizer.to_sequence(data['pos'], 'pos')   
            
            for aspect in data['aspects']:
                deprel = mytokenizer.to_sequence(aspect['deprel'], 'deprel')
                start, end = int(aspect['from']), int(aspect['to'])
                aspect_mask = [1 if start <= i < end else 0 for i in range(len(data['token']))]
                aspect_head = mytokenizer.pad_sequence(aspect['head'], 0)
                aspect_mask = mytokenizer.pad_sequence(aspect_mask, 0)
                path = list()
                for i in range(len(data['token'])):
                    aspect['path'][i].append(i+1+mytokenizer.max_slen)  # add a stop node with extra relation
                    path.append(mytokenizer.pad_sequence(aspect['path'][i], 0, maxlen=mytokenizer.max_plen))
                for i in range(mytokenizer.max_slen - len(data['token'])): # pad the paths
                    path.append(mytokenizer.pad_sequence([], 0, maxlen=mytokenizer.max_plen))
                path = np.asarray(path)
                polarity = polarity_dict[aspect['polarity']]
                
                '''compute position'''
                if aspect_mask.sum() == 0:
                    break
                dataset.append(((text_mask, wordpiece_mask, wordpiece_gather_idx, textp_id, pos, deprel, aspect_head, aspect_mask, path), polarity)) # samples in (x, y) tuples
        return dataset
    
    def __getitem__(self, index):
        return self._dataset[index]
    
    def __len__(self):
        return len(self._dataset)


def get_wordpiece_gather_idx(textp, text, max_slen):
    max_num = 100
    wordpiece_idx = np.zeros((max_slen + 1, max_num), dtype=np.int64)
    i = 1    # row index for wordpiece idx
    j = 0    # column index for wordpiece idx
    ind = 1  # index for textp
    for complete_word in text:
        complete_word = complete_word.lower()
        word = ''
        while ind < len(textp):
            piece = textp[ind].replace('##', '')
            word += piece
            if word == complete_word:
                wordpiece_idx[i][j] = ind
                j = 0
                i += 1
                ind += 1
                break
            else:
                wordpiece_idx[i][j] = ind
                j += 1
                ind += 1
        if ind > len(textp):
            break
    return wordpiece_idx


def build_tokenizer(fnames, dataset, mode):
    if not os.path.exists(os.path.join('dats', mode)):
        os.mkdir(os.path.join('dats', mode))
    data_file = os.path.join('dats', os.path.join(mode, f"{dataset}_tokenizer.dat"))  # tokenizer cache
    if os.path.exists(data_file):
        print(f"loading tokenizer: {data_file}")
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        print('building tokenizer...')
        tokenizer = Tokenizer.from_files(fnames)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer

def _load_wordvec(data_path, word_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        word_vec['<pad>'] = np.zeros(word_dim).astype('float32') # embedding vector for <pad> is always zero
        for line in f:
            tokens = line.rstrip().split()
            if (len(tokens) - 1) != word_dim:
                continue
            if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
                continue
            if vocab is None or vocab.has_word(tokens[0]):
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        return word_vec

def build_embedding_matrix(vocab, dataset, mode, word_dim=300):
    data_file = os.path.join('dats', os.path.join(mode, f"{dataset}_embedding_matrix.dat")) # embedding matrix cache
    glove_file = os.path.join('glove', 'glove.840B.300d.txt') # pre-trained glove embedding file
    if os.path.exists(data_file):
        print(f"loading embedding matrix: {data_file}")
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), word_dim)).astype('float32') # sample from U(-0.25,0.25)
        word_vec = _load_wordvec(glove_file, word_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix

def map_to_ids(tokens, vocab, lower=True):
    if lower:
        ids = [vocab[t.lower()] if t.lower() in vocab else vocab['[UNK]'] for t in tokens]
    else:
        ids = [vocab[t] if t in vocab else vocab['[UNK]'] for t in tokens]
    return ids
