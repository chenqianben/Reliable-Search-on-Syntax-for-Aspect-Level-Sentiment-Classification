import os
import math
import torch
import random
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
from model import RepWalk
from loss_func import CrossEntropy
from data_utils import MyDataset, build_tokenizer
from pytorch_pretrained import BertTokenizer


class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt): # prepare for training the model
        opt.mytokenizer = build_tokenizer(fnames=opt.dataset_file.values(), dataset=opt.dataset, mode=opt.mode,)  # transfrom tokens to indices
        opt.berttokenizer = BertTokenizer.from_pretrained('pretrained_bert')  # 注意，用bert自带的tokenizer分词器，可以做到subword分词！！
        self.opt = opt  # hyperparameters and options
        self.trainset = MyDataset(opt.dataset_file['train'], opt.berttokenizer, opt.mytokenizer, opt.mode) # training set
        self.testset = MyDataset(opt.dataset_file['test'], opt.berttokenizer, opt.mytokenizer, opt.mode) # testing set
        self.model = RepWalk(opt).to(opt.device) # neural network model
        self._print_args() # print arguments
    
    def _print_args(self): # pring arguments
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        if self.opt.device.type == 'cuda':
            print(f"cuda memory allocated: {torch.cuda.memory_allocated(self.opt.device.index)}")
        print(f"n_trainable_params: {int(n_trainable_params)}, n_nontrainable_params: {int(n_nontrainable_params)}")
        print('training arguments:')
        for arg in vars(self.opt):
            print(f">>> {arg}: {getattr(self.opt, arg)}")
    
    def _reset_params(self):  # reset model parameters
        for name, param in self.model.named_parameters():
            # if param.requires_grad and not name.startswith('bert'):
            if param.requires_grad and 'bert' not in name:
                if 'embedding' in name:  # treat embedding matrices as special cases
                        weight = torch.nn.init.xavier_uniform_(torch.zeros_like(param))  # use xavier_uniform to initialize embedding matrices
                        weight[0] = torch.tensor(0, dtype=param.dtype, device=param.device)  # the vector corresponding to padding index should be zero
                        setattr(param, 'data', weight)  # update embedding matrix
                else:
                    if len(param.shape) > 1:
                        torch.nn.init.xavier_uniform_(param)  # use xavier_uniform to initialize weight matrices
                    else:
                        stdv = 1. / math.sqrt(param.size(0))
                        torch.nn.init.uniform_(param, a=-stdv, b=stdv)  # use uniform to initialize bias vectors

    def _train(self, dataloader, criterion, optimizer): # train the model
        train_loss, n_correct, n_train = 0, 0, 0 # reset counters
        self.model.train() # switch model to training mode
        for sample_batched in dataloader: # mini-batch optimization
            inputs = list(map(lambda x: x.to(self.opt.device), sample_batched[0])) # move tensors to target device (e.g. cuda)
            labels = sample_batched[1].to(self.opt.device) # move labels to target device
            if not self.opt.test:
                outputs = self.model(inputs)     # compute outputs
            else:
                outputs, _ = self.model(inputs)  # compute outputs and node weights
            
            optimizer.zero_grad() # clear gradient accumulators
            loss = criterion(outputs, labels) # compute batch loss
            loss.backward() # compute gradients through back-propagation
            optimizer.step() # update model parameters
            
            train_loss += loss.item() * len(labels) # update train loss
            n_correct += (torch.argmax(outputs[0], -1) == labels).sum().item() # update correct sample number
            n_train += len(labels) # update train sample number
        return train_loss / n_train, n_correct / n_train
    
    def _evaluate(self, dataloader, criterion): # evaluate the model
        test_loss, n_correct, n_test = 0, 0, 0 # reset counters
        labels_all, predicts_all = None, None # initialize variables
        self.model.eval() # switch model to evaluation mode
        with torch.no_grad(): # turn off gradients
            for sample_batched in dataloader:
                inputs = list(map(lambda x: x.to(self.opt.device), sample_batched[0]))
                labels = sample_batched[1].to(self.opt.device)
                if not self.opt.test:
                    outputs = self.model(inputs)  # compute outputs
                else:
                    outputs, _ = self.model(inputs)  # compute outputs and node weights
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * len(labels)
                n_correct += (torch.argmax(outputs[0], -1) == labels).sum().item()
                n_test += len(labels)
                labels_all = torch.cat((labels_all, labels), dim=0) if labels_all is not None else labels
                predicts_all = torch.cat((predicts_all, outputs[0]), dim=0) if predicts_all is not None else outputs[0]
        f1 = metrics.f1_score(labels_all.cpu(), torch.argmax(predicts_all, -1).cpu(), labels=[0, 1, 2], average='macro') # compute f1 score
        return test_loss / n_test, n_correct / n_test, f1

    def _keywords(self, dataloader, tokenizer, filename):
        import json
        gates_all = []
        self.model.eval()  # switch model to evaluation mode
        with torch.no_grad():  # turn off gradients
            for sample_batched in dataloader:
                inputs = list(map(lambda x: x.to(self.opt.device), sample_batched[0]))
                labels = sample_batched[1]
                outputs, gates = self.model(inputs)  # compute outputs and node weights, (bs, sl)
                text_len = torch.sum(inputs[3] != 0, dim=-1)  # length of sentences, shape (batch_size)
                # gates = torch.gather(gates, 1, inputs[5])   # final argument: for gather_idx

                gates = gates.cpu().numpy() if torch.cuda.is_available() else gates.numpy()
                inputs[3] = inputs[3].cpu().numpy() if torch.cuda.is_available() else inputs[3].numpy()    # for text
                outputs[0] = outputs[0].cpu().numpy() if torch.cuda.is_available() else outputs[0].numpy() # for predictions
                labels = labels.cpu().numpy() if torch.cuda.is_available() else labels.numpy()
                for text, pred, gat, lab, tl in zip(inputs[3], outputs[0], gates, labels, text_len):
                    # 为了解决text考虑p，而gate没有考虑p的问题
                    sent = tokenizer.convert_ids_to_tokens(text[1:tl])
                    gat = gat[:tl-1].tolist()  # 不能只用list，否则元素的数据类型没有从numpy float变成python float

                    indices = np.argsort(-np.array(gat))
                    pred = np.argmax(pred)

                    keywords = sent[indices[0]]
                    gates_all.append({'sent': ' '.join(sent), 'gate': gat, 'label': int(lab), 'pred': int(pred), 'keywords': keywords})
        if not os.path.exists(f'gates/{self.opt.mode}'):
            os.mkdir(f'gates/{self.opt.mode}')
        if not os.path.exists(f'gates/{self.opt.mode}/{self.opt.dataset}'):
            os.mkdir(f'gates/{self.opt.mode}/{self.opt.dataset}')
        with open(f'gates/{self.opt.mode}/{self.opt.dataset}/{filename}.json', 'w') as fw:
            json.dump(gates_all, fw, indent=4)
        return

    def run(self):
        print(f'current mode: {self.opt.mode}')
        print(f'current file name: {self.opt.filename}')
        # _params = filter(lambda p: p.requires_grad, self.model.parameters())  # trainable parameters
        # optimizer = torch.optim.Adam(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)  # use the Adam optimizer
        criterion = CrossEntropy(beta=self.opt.beta, eps=self.opt.eps, gamma=self.opt.gamma)  # loss function implemented as described in paper
        #
        # train_dataloader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)  # training dataloader
        test_dataloader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)  # testing dataloader
        #
        # self._reset_params()  # reset model parameters
        # best_test_acc, best_test_f1 = 0, 0  # record the best acc and f1 score on testing set
        # for epoch in range(self.opt.num_epoch):
        #     train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)  # train the model
        #     test_loss, test_acc, test_f1 = self._evaluate(test_dataloader, criterion)  # evaluate the model
        #     if test_acc > best_test_acc:
        #         if self.opt.test:
        #             if not os.path.exists(f'../model_weights/{self.opt.mode}'): os.mkdir(f'../model_weights/{self.opt.mode}')
        #             if not os.path.exists(f'../model_weights/{self.opt.mode}/{self.opt.dataset}'): os.mkdir(f'../model_weights/{self.opt.mode}/{self.opt.dataset}')
        #             torch.save(self.model.state_dict(), f'../model_weights/{self.opt.mode}/{self.opt.dataset}/{test_acc:.4f}_{cur_seed}.pth')
        #             if os.path.exists(f'..model_weights/{self.opt.mode}/{self.opt.dataset}/{best_test_acc:.4f}_{cur_seed}.pth'):
        #                 os.remove(f'../model_weights/{self.opt.mode}/{self.opt.dataset}/{best_test_acc:.4f}_{cur_seed}.pth')
        #         best_test_acc = test_acc
        #         best_test_f1 = test_f1
        #     print(f"{100 * (epoch + 1) / self.opt.num_epoch:6.2f}% > loss: {train_loss:.4f}, acc: {train_acc:.4f}, test acc: {test_acc:.4f}, test f1: {test_f1:.4f}")
        # print('#' * 50)
        # print(f"best test acc: {best_test_acc:.4f}, best test f1: {best_test_f1:.4f}")
        if self.opt.test:
            self.model.load_state_dict(torch.load(f'model_weights/{self.opt.mode}/{self.opt.dataset}/{self.opt.filename}.pth', map_location=lambda storage, loc: storage))
            # self._keywords(test_dataloader, self.opt.berttokenizer, self.opt.filename)
        test_loss, test_acc, test_f1 = self._evaluate(test_dataloader, criterion)
        print(test_loss, test_acc, test_f1)


def main(opt):
    ''' run the model '''
    for _ in range(5):
        ins = Instructor(opt)
        ins.run()


if __name__ == '__main__':
    dataset_files = {
        'restaurant': {
            'train': os.path.join('datasets', 'Restaurants_Train.json'),
            'test': os.path.join('datasets', 'Restaurants_Test.json')
        },
        'laptop': {
            'train': os.path.join('datasets', 'Laptops_Train.json'),
            'test': os.path.join('datasets', 'Laptops_Test.json')
        },
        'twitter': {
            'train': os.path.join('datasets', 'Tweets_Train.json'),
            'test': os.path.join('datasets', 'Tweets_Test.json')
        },
        'restaurant16': {
            'train': os.path.join('datasets', 'Restaurants16_Train.json'),
            'test': os.path.join('datasets', 'Restaurants16_Test.json')
        },
        'restaurant16a': {
            'train': os.path.join('datasets', 'Restaurants16a_Train.json'),
            'test': os.path.join('datasets', 'Restaurants16a_Test.json')
        },
        'mams': {
            'train': os.path.join('datasets', 'MAMS-ATSA_Train.json'),
            'test': os.path.join('datasets', 'MAMS-ATSA_Test.json')
        }
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    ''' hyperparameters '''
    parser.add_argument('--mode', default='bert_shell', type=str)
    parser.add_argument('--filename', default='0.7413_1022644396', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--test', default=True, type=bool)

    parser.add_argument('--pos_dim', default=30, type=int)
    parser.add_argument('--dep_dim', default=50, type=int)
    parser.add_argument('--hidden_dim', default=500, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epoch', default=75, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--embed_dropout', default=0.5, type=float)
    parser.add_argument('--bilinear_dropout', default=0.5, type=float)
    parser.add_argument('--fc_dropout', default=0.5, type=float)
    parser.add_argument('--beta', default=0.04, type=float)
    parser.add_argument('--eps', default=0.05, type=float)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='cpu, cuda')
    parser.add_argument('--num_filters', default=128, type=int)
    parser.add_argument('--window_sizes', default=[1,3,5], nargs='+', type=int)

    ''' parse arguments '''
    opt = parser.parse_args()
    opt.dataset_file = dataset_files[opt.dataset]
    # opt.seed = opt.seed if opt.seed else random.randint(0, 1e4)
    opt.device = torch.device(opt.device) if opt.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ''' if you are using cudnn '''
    torch.backends.cudnn.deterministic = True  # Deterministic mode can have a performance impact，，避免计算中的随机性
    torch.backends.cudnn.benchmark = False  # 若设置成True，将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。但是RNN是动态的，seq_len会变，因此要设置成False

    main(opt)