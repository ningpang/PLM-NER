from torchcrf import CRF
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
import torch

class LSTM(nn.Module):
    def __init__(self, args, word2id, tag2id):
        super(LSTM, self).__init__()

        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.vocab_size = len(word2id)+1
        self.tag2id = tag2id
        self.tag_size = len(tag2id)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout()

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, num_layers=1, bidirectional=True, batch_first=False)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size, batch_first=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask):
        embedding = self.embedding_layer(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        if self.decoder=='CRF':
            outputs = self.crf.decode(outputs, mask=mask)
        elif self.decoder == 'tokenclassification':
            outputs=nn.Softmax(dim=1)(outputs)
            outputs=torch.max(outputs, 1)[1].cpu().numpy()
        else:
            raise NotImplementedError
        return outputs

    def log_likehood(self, x, tags, mask):
        embedding = self.embedding_layer(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        if self.decoder=='CRF':
            return -self.crf(outputs, tags, mask=mask)
        elif self.decoder=='tokenclassification':
            return self.criterion(outputs.permute(0,2 ,1), tags)
        else:
            raise NotImplementedError


class Bert_BiLSTM(nn.Module):
    def __init__(self, args, word2id, tag2id):
        super(Bert_BiLSTM, self).__init__()

        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.vocab_size = len(word2id)+1
        self.tag2id = tag2id
        self.tag_size = len(tag2id)

        self.embedding_layer = BertModel.from_pretrained('bert-base-uncased' if args.bert_path=='./dataset/bert_english' else 'bert-base-chinese')
        self.dropout = nn.Dropout()

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, num_layers=1, bidirectional=True, batch_first=False)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.decoder=args.decoder
        self.crf = CRF(self.tag_size, batch_first=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask):
        embedding,_ = self.embedding_layer(x,attention_mask=mask, output_all_encoded_layers=False)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        if self.decoder=='CRF':
            outputs = self.crf.decode(outputs, mask=mask)
        elif self.decoder == 'tokenclassification':
            outputs=nn.Softmax(dim=2)(outputs)
            outputs=torch.max(outputs.data, 2)[1].cpu().numpy()
        else:
            raise NotImplementedError
        return outputs

    def log_likehood(self, x, tags, mask):
        embedding,_ = self.embedding_layer(x,attention_mask=mask, output_all_encoded_layers=False)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        if self.decoder=='CRF':
            return -self.crf(outputs, tags, mask=mask)
        elif self.decoder=='tokenclassification':
            return self.criterion(outputs.permute(0,2 ,1), tags)
        else:
            raise NotImplementedError


class Bert(nn.Module):
    def __init__(self, args, word2id, tag2id):
        super(Bert, self).__init__()

        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.vocab_size = len(word2id)+1
        self.tag2id = tag2id
        self.tag_size = len(tag2id)

        self.embedding_layer = BertModel.from_pretrained('bert-base-uncased' if args.bert_path=='./dataset/bert_english' else 'bert-base-chinese')
        self.dropout = nn.Dropout()


        self.hidden2tag = nn.Linear(self.embedding_dim, self.tag_size)
        self.decoder=args.decoder
        self.crf = CRF(self.tag_size, batch_first=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask):
        outputs,_ = self.embedding_layer(x,attention_mask=mask, output_all_encoded_layers=False)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        if self.decoder=='CRF':
            outputs = self.crf.decode(outputs, mask=mask)
        elif self.decoder == 'tokenclassification':
            outputs=nn.Softmax(dim=2)(outputs)
            outputs=torch.max(outputs.data, 2)[1].cpu().numpy()
        else:
            raise NotImplementedError
        return outputs

    def log_likehood(self, x, tags, mask):
        outputs,_ = self.embedding_layer(x,attention_mask=mask, output_all_encoded_layers=False)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        if self.decoder=='CRF':
            return -self.crf(outputs, tags, mask=mask)
        elif self.decoder=='tokenclassification':
            return self.criterion(outputs.permute(0,2 ,1), tags)
        else:
            raise NotImplementedError