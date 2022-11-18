import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', default='conll2003', type=str,help="('renmin','conll2003')")
    parser.add_argument('--vocab_size', default=4000, type=int)
    parser.add_argument('--max_length', default=100, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    # model
    parser.add_argument('--model', default='Bert', type=str,help="('LSTM','Bert_BiLSTM','Bert)")
    parser.add_argument('--embedding_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--decoder', default='CRF', type=str,help="('tokenclassification','CRF')")
    # training
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--save_dict', default='', type=str)
    parser.add_argument('--bad_count', default=5, type=int)
    #bert
    parser.add_argument('--bert_path' , default= './dataset/bert_english', type=str,help="填这个路径没有模型会自动下载('./dataset/bert_chinese', './dataset/bert_english')")

    args = parser.parse_args()
    args.embedding_dim=768 if args.model=='Bert_BiLSTM' or args.model=='Bert' else args.embedding_dim
    args.hidden_dim = 1000 if args.model == 'Bert_BiLSTM' or args.model=='Bert' else args.hidden_dim
    args.save_dict = './save_dict/' + args.model+'.ckpt'

    return args
