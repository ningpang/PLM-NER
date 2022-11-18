import os
import re
import codecs
import torch
from collections import Counter
from arguments import get_args_parser
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

class TextDataset(object):
    def __init__(self, args):
        self.args = args
        self.base_dir = os.path.join('./dataset', args.dataset)
        self.words, self.word_to_id, self.tag_to_id = self.read_vocab(self.args)
        self.train_data, self.val_data, self.test_data = self.get_data()

    def BMES2BIO(self,tag):
        tag=tag.split('_')
        if tag[0]=='O':
            return 'O'
        elif tag[0]=='B' or tag[0]=='S':
            return 'B_'+tag[1]
        elif tag[0]=='M' or tag[0]=='E':
            return 'I_' + tag[1]
        else:
            return tag[0]

    def read_vocab(self,args):
        if args.model=='Bert_BiLSTM' or args.model=='Bert':
            word_to_id = load_vocab(os.path.join(args.bert_path,'vocab.txt'))
            tag_to_id = load_vocab(os.path.join(self.base_dir,'tag.txt'))
            words=[]
        else:
            data_dir = os.path.join(self.base_dir, 'data.txt')
            raw_data = codecs.open(data_dir,encoding='utf-8')
            tags = set()
            tags.add('')
            tag_to_id = {}
            datas = []
            for line in raw_data.readlines():
                sentence = re.split('[，。！？、‘’“”:]/[O]', line)
                sentence = ' '.join(sentence)
                sentence = sentence.strip().split()
                if sentence == []:
                    continue
                for word in sentence:
                    word = word.split('/')
                    datas.append(word[0])
                    tags.add(self.BMES2BIO(word[1]))
                # datas.append(sentence[0])
                # tags.add(sentence[1])
            counter = Counter(datas)
            count_pairs = counter.most_common(self.args.vocab_size-1)
            words, _ = list(zip(*count_pairs))
            words = ['[PAD]'] + list(words)+['[UNK]']
            word_to_id = dict(zip(words, range(len(words))))
            for tag in tags:
                tag_to_id[tag] = len(tag_to_id)
        return words, word_to_id, tag_to_id

    def get_data(self):
        data_dir = os.path.join(self.base_dir, 'data.txt')
        all_data, all_labels = self.process_file(data_dir)
        x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2,
                                                            random_state=2021)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                            random_state=2021)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def process_file(self, filename):
        raw_data = codecs.open(filename,encoding='utf-8')
        contents = []
        labels = []
        if args.dataset == 'renmin':
            for line in raw_data.readlines():
                token_id = []
                tag_id = []
                mask = []
                sentence = re.split('[，。！？、‘’“”:]/[O]', line)
                sentence = ' '.join(sentence)
                sentence = sentence.strip().split()
                if sentence == []:
                    continue
                sentence = ['[CLS]/<start>'] + sentence + ['[SEP]/<eos>']
                for word in sentence:
                    word = word.split('/')
                    token_id.append(self.word_to_id[word[0]] if word[0] in self.word_to_id else self.word_to_id['[UNK]'])
                    tag_id.append(self.tag_to_id[self.BMES2BIO(word[1])])
                if self.args.max_length:
                    if len(token_id)< self.args.max_length:
                        mask = [1] * len(token_id) + [0] * (self.args.max_length - len(token_id))
                        token_id += [0] * (self.args.max_length - len(token_id))
                        tag_id += [0] * (self.args.max_length - len(tag_id))
                    else:
                        mask = [1] * self.args.max_length
                        token_id = token_id[:self.args.max_length]
                        tag_id = tag_id[:self.args.max_length]
                assert len(mask) == len(token_id) == len(tag_id)
                contents.append((token_id, mask))
                labels.append(tag_id)

        elif args.dataset == 'conll2003':
            words = []
            tags = []
            token_id = []
            tag_id = []
            for line in raw_data.readlines():
                word = line.strip().split(' ')[0]
                tag = line.strip().split(' ')[-1]
                if line!='\n':
                    words.append(word)
                    tags.append(tag)
                else:
                    words = ['[CLS]'] + words + ['[SEP]']
                    tags = ['<start>'] + tags + ['<eos>']
                    for token in range(len(words)):
                        token_id.append(self.word_to_id[words[token]] if words[token] in self.word_to_id else self.word_to_id['[UNK]'])
                        tag_id.append(self.tag_to_id[tags[token]])
                    if self.args.max_length:
                        if len(token_id) < self.args.max_length:
                            mask = [1] * len(token_id) + [0] * (self.args.max_length - len(token_id))
                            token_id += [0] * (self.args.max_length - len(token_id))
                            tag_id += [0] * (self.args.max_length - len(tag_id))
                        else:
                            mask = [1] * self.args.max_length
                            token_id = token_id[:self.args.max_length]
                            tag_id = tag_id[:self.args.max_length]
                    assert len(mask) == len(token_id) == len(tag_id)
                    contents.append((token_id, mask))
                    labels.append(tag_id)

                    words = []
                    tags = []
                    token_id = []
                    tag_id = []

        return contents, labels

class data_set(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        words = torch.tensor([item[0][0] for item in data])
        mask = torch.tensor([item[0][1] for item in data], dtype=torch.uint8)
        label = torch.tensor([item[1] for item in data])
        return words, mask, label

def data_loader(args, data, shuffle=True, drop_last=False, batch_size=None):
    dataset = data_set(data)
    if batch_size == None:
        batch_size = min(args.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last
    )
    return data_loader

args = get_args_parser()
dataset = TextDataset(args)

def get_train_loader():
    train_words = dataset.train_data[0]
    train_tags = dataset.train_data[1]
    train_data = list(zip(train_words, train_tags))
    return data_loader(args, train_data)


def get_val_loader():
    val_words = dataset.val_data[0]
    val_tags = dataset.val_data[1]
    val_data = list(zip(val_words, val_tags))
    return data_loader(args, val_data)

def get_test_loader():
    test_words = dataset.test_data[0]
    test_tags = dataset.test_data[1]
    test_data = list(zip(test_words, test_tags))
    return data_loader(args, test_data)

def get_word_tag():
    word2id = dataset.word_to_id
    tag2id = dataset.tag_to_id
    return word2id, tag2id
