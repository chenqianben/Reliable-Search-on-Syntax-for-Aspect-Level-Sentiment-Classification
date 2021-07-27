import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import DynamicLSTM
from pytorch_pretrained import BertModel


class RepWalk(nn.Module):
    ''' Neural Network Structure '''

    def __init__(self, opt):
        super(RepWalk, self).__init__()  # initialize the super class
        self.opt = opt
        ''' common variables '''
        # WD = opt.word_dim  # dimension of word embeddings 300
        WD = 768
        PN = len(opt.mytokenizer.vocab['pos'])  # number of pos tags in vocabulary 31
        PD = opt.pos_dim  # dimension of pos tag embeddings 30
        P_PAD = opt.mytokenizer.vocab['pos'].pad_id  # padding index of pog tags 0
        RN = len(opt.mytokenizer.vocab['deprel'])  # number of dependency relation in vocabulary 58
        RD = opt.dep_dim  # dimension of dependency relation embeddings 50
        R_PAD = opt.mytokenizer.vocab['deprel'].pad_id  # padding index of dependency relation 0
        HD = opt.hidden_dim  # dimension of bi-gru hidden state 300
        PL = opt.mytokenizer.max_plen
        self.num_filters = opt.num_filters         # num of filters for 2d conv
        self.window_sizes = opt.window_sizes       # sizes of windows for 2d conv
        self.opt = opt
        ''' embedding layer '''
        # self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))  # pre-trained embedding layer
        self.bert = BertModel.from_pretrained('pretrained_bert')
        for param in self.bert.parameters():
            param.requires_grad = True
        self.pos_embedding = nn.Embedding(PN, PD, padding_idx=P_PAD)  # pos tag embedding layer
        self.deprel_embedding = nn.Embedding(RN, RD, padding_idx=R_PAD)  # dependency relation embedding layer
        self.pathdepth_embedding = nn.Embedding(PL + 2, PD, padding_idx=0)
        ''' other parameter '''
        self.pad_word = nn.Parameter(torch.zeros(HD * 2),requires_grad=False)  # the padding word for training   0向量（1，HD*2)
        self.pad_edge = nn.Parameter(torch.ones(1), requires_grad=False)  # the padding edge for training       1向量（1）
        self.pad_zeroedge = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.ext_rel = nn.Parameter(torch.Tensor(RD),requires_grad=True)  # embedding for the edge with 'ext' relation   随机向量
        ''' main layer '''
        self.fc1 = nn.Linear(WD + PD + PD, HD * 2) # 注意！！！ 用线性层代替了rnn
        self.bilinear = nn.Bilinear(HD * 4, RD, 1)  # bilinear layer for score function
        self.fc_out = nn.Linear(self.num_filters * len(self.window_sizes), 3)  # fully-connected output layer
        '''cnn layer'''
        # self.convs = nn.ModuleList([
        #     nn.Conv1d(in_channels=2*HD, out_channels=self.num_filters, kernel_size=window_size, padding=window_size//2) for window_size in self.window_sizes
        # ])
        self.kernel_weights = nn.ParameterList(
            [nn.Parameter(torch.empty((ks, 2 * HD, self.num_filters)), requires_grad=True) for ks in self.window_sizes])
        self.kernel_biases = nn.ParameterList(
            [nn.Parameter(torch.empty((self.num_filters)), requires_grad=True) for ks in self.window_sizes])
        ''' dropout layer '''
        self.embed_dropout = nn.Dropout(opt.embed_dropout)  # dropout for embeddings
        self.bilinear_dropout = nn.Dropout(opt.bilinear_dropout)  # dropout for bilinear layer
        self.fc_dropout = nn.Dropout(opt.fc_dropout)  # dropout for fully-connected layer

    def forward(self, inputs):
        text_mask, wordpiece_mask, wordpiece_gather_idx, textp, pos, deprel, aspect_head, aspect_mask, path = inputs  # input features, shape (batch_size, wordpiece_seq_len)(注意，对于这些tensors，只是padding到了wordpiece_seq_len的长度，但是还是按照一整个一整个单词来做的feature) except for the path whose shape is (batch_size, wordpiece_seq_len, path_len) and the wordpiece_mask/wordpiece_existing_mask/text whose shape are (batch_size, wordpiece_seq_len+1)
        # original_text = self.opt.berttokenizer.convert_ids_to_tokens(text[0][:pieceword_len[0]].cpu().numpy())
        ''' common variables '''
        pieceword_len = torch.sum(textp != 0, dim=-1)   # length of pieceword sentences, shape (batch_size), -1表示去掉长度cls
        text_mask = text_mask.unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1)         # mask of aspects (aspects: 1, others: 0), shape (batch_size, wordpiece_seq_len)
        path_depth = torch.sum(path != 0, dim=-1)       # (bs, sl)
        ''' bert encoder '''
        word_feature, text_cls = self.bert(textp, attention_mask=wordpiece_mask.squeeze(), output_all_encoded_layers=False) # 注意！！bert模型
        # text(bs, wordpiece_seq_len+1) encoder_out(bs,wordpiece_seq_len+1, 768) text_cls(bs, 768)
        '''把有cls打头的都去掉'''
        wordpiece_mask = wordpiece_mask[:,1:]       # (bs, wordpiece_seq_len)
        word_feature = word_feature[:,1:,:]   # (bs, wordpiece_seq_len, 768) # 注意，这里去掉了数据开头的cls
        wordpiece_gather_idx = wordpiece_gather_idx[:, 1:, :]  # (bs, wordpiece_seq_len, max_piece_len)
        pieceword_len -= 1
        ''' embedding '''
        BS, SL, WPL = wordpiece_gather_idx.shape
        word_feature = word_feature.permute(0, 2, 1).unsqueeze(1).expand(-1, SL, -1, -1)      # (bs, sl, 768, sl)，注意最后一个sl才是真正的sl信息，前面的sl是复制的维度
        pad_zeroedge_feature = self.pad_zeroedge.reshape(1, 1, 1, 1).expand(BS, SL, 768, -1)  # (bs, sl, 768, 1) 全0
        word_feature = torch.cat((pad_zeroedge_feature, word_feature), dim=-1)                # (bs, sl, 768, sl+1)
        word_feature = torch.gather(word_feature, 3, wordpiece_gather_idx.reshape(BS, SL, 1, WPL).expand(-1, -1, 768, -1))  # (bs, wordpiece_seq_len, 768, max_piece_len)
        num_pieces = (torch.sum(wordpiece_gather_idx != 0, dim=-1)).unsqueeze(-1)
        word_feature = torch.div(torch.sum(word_feature, dim=-1), torch.where(num_pieces != 0, num_pieces, torch.ones_like(num_pieces)))
        word_feature = self.embed_dropout(word_feature)  # (bs, sl, 768)

        pos_feature = self.embed_dropout(self.pos_embedding(pos))     # embed pos tags to vectors, shape (batch_size, wordpiece_seq_len, pos_emb_dim)
        deprel_feature = self.embed_dropout(self.deprel_embedding(deprel))  # emb dependency relations to vectors, shape (batch_size, wordpiece_seq_len, dep_emb_dim)
        extrel_feature = self.embed_dropout(self.ext_rel)  # embedding vector of extra relation, shape (dep_emb_dim)
        pathdepth_feature = self.embed_dropout(self.pathdepth_embedding(path_depth))
        ''' basic feature '''
        node_feature = F.relu(self.fc1(torch.cat((word_feature, pos_feature, pathdepth_feature), dim=-1))) #这些feature都是(bs, wordpiece_seq_len, embedding(三个embedding的维度大小不一样)),并用线性层代替了rnn node_feature (bs,sl,2*hidden_dim)
        BS, SL, FD = node_feature.shape  # shape of node representations
        extrel_feature = extrel_feature.reshape(1, 1, -1).expand(BS, SL,-1)  # expand extra relation embedding, shape (batch_size, wordpiece_seq_len, dep_emb_dim)
        padword_feature = self.pad_word.reshape(1, 1, -1).expand(BS, -1, -1)  # expand padding word embedding, shape (batch_size, 1, hidden_dim*2)全0
        exttext_feature = self.pad_word.reshape(1, 1, -1).expand(BS, SL, -1)  # expand extra text feature, shape (batch_size, wordpiece_seq_len, hidden_dim*2)全0
        padedge_feature = self.pad_edge.reshape(1, 1, -1).expand(BS, -1, -1)  # expand padding edge embedding, shape (batch_size, 1, 1)全1
        ''' edge feature '''
        deptext_feature = torch.cat((padword_feature, node_feature), dim=1).permute(2, 0, 1)  # dependents features, shape (hidden_dim*2, batch_size, wordpiece_seq_len+1)所有节点文本特征
        aspect_head = aspect_head.unsqueeze(0).expand(FD, -1, -1)  # head indices of current aspect, shape (hidden_dim*2, batch_size, wordpiece_seq_len)父节点索引
        deptext_feature = torch.gather(deptext_feature, 2, aspect_head).permute(1, 2, 0)  # permuted dependents features, shape (batch_size, wordpiece_seq_len, hidden_dim*2)父节点特征
        head_text_feature = torch.cat((deptext_feature, node_feature),dim=1)  # the features of start node at each edge, shape (batch_size, wordpiece_seq_len*2, hidden_dim*2)父节点特征+节点特征
        tail_text_feature = torch.cat((node_feature, exttext_feature), dim=1)  # the features of end node at edge edge, shape (batch_size, wordpiece_seq_len*2, hidden_dim*2)节点特征+0
        edge_feature = torch.cat((head_text_feature, tail_text_feature), dim=-1)  # the features of edges, shape (batch_size, wordpiece_seq_len*2, hidden_dim*4)起始节点和终止节点特征
        ''' score function '''
        label_feature = torch.cat((deprel_feature, extrel_feature),dim=1)  # compose label features, shape (batch_size, wordpiece_seq_len*2, dep_emb_dim)关系标签+随机学习向量
        edge_score = torch.sigmoid(self.bilinear(self.bilinear_dropout(edge_feature),label_feature))  # compute score for each edge, shape (batch_size, wordpiece_seq_len*2, 1)要改一下边的权值
        edge_score = torch.cat((padedge_feature, edge_score.transpose(1, 2)), dim=-1).expand(-1, SL, -1)  # expand edge scores, shape (batch_size, wordpiece_seq_len, wordpiece_seq_len*2+1)
        ''' node weight '''
        node_weight = torch.prod(torch.gather(edge_score, 2, path), dim=-1,keepdim=True)  # compute node weights, shape (batch_size, wordpiece_seq_len, 1)按路径将边的得分相乘
        node_weight = torch.where(text_mask != 0, node_weight, torch.zeros_like(node_weight))  # eliminate values out of texts, shape (batch_size, wordpiece_seq_len, 1)不是文本权重就为0
        node_weight = torch.where(aspect_mask == 0, node_weight, torch.zeros_like(node_weight))  # compute final node weights, shape (batch_size, wordpiece_seq_len, 1)是aspect权重就为0
        weight_norm = torch.sum(node_weight.squeeze(-1), dim=-1)  # compute L1 norm of weights, shape (batch_size)
        # ''' sentence representation '''
        # sentence_feature = torch.sum(node_weight * node_feature,dim=1)  # compute sentence features, shape (batch_size, hidden_dim*2)
        # predicts = self.fc_out(self.fc_dropout(sentence_feature))  # use fully-connected network to generate predicts, shape (batch_size, label_dim)
        '''cnn'''
        # for j, (conv, window_size) in enumerate(zip(self.convs, self.window_sizes)):
        #     node_weight_padded = F.pad(node_weight, (0, 0, window_size//2, window_size//2, 0, 0))  # (bs, sl + 2*(window_size//2), 1)
        #     paired_node_weight = torch.cat([torch.mean(node_weight_padded[:, i:i + window_size, :], dim=1) for i in range(node_feature.size(1))], dim=1)  # (bs, sl)
        #     weighted_node_feature = paired_node_weight.unsqueeze(-1) * node_feature     # (batch_size, seq_len, hidden_dim*2), use text mask to mask padding
        #     x = F.relu(conv(weighted_node_feature.permute(0, 2, 1)))      # (bs, nf, sl)
        #     paired_sentence_feature = F.max_pool1d(x * text_withoutp_mask.squeeze().unsqueeze(1), x.size(2))  # (bs, nf, 1)
        #     total_paired_sentence_feature = torch.cat((total_paired_sentence_feature, paired_sentence_feature), dim=1) if j != 0 else paired_sentence_feature
        # predicts = self.fc_out(self.fc_dropout(total_paired_sentence_feature.squeeze()))  # use fully-connected network to generate predicts, shape (batch_size, label_dim)
        node_feature = node_feature.permute(0, 2, 1)  # (bs, hs, sl)
        node_weight = node_weight.permute(0, 2, 1)  # (bs, 1, sl)
        cnn_outputs = list()
        for ki, ks in enumerate(self.window_sizes):
            raw_indexes = torch.arange(0, SL - ks + 1, 1, dtype=torch.int64, device=node_feature.device)  # (conv_num)
            cur_node_weight, cur_node_feature = list(), list()
            for k in range(ks):
                weight_indexes = (raw_indexes + k).expand(BS, 1, -1)  # (batch_size, 1, conv_num)
                feature_indexes = (raw_indexes + k).expand(BS, FD, -1)  # (batch_size, hidden_dim, conv_num)
                temp_node_weight = torch.gather(node_weight, dim=-1, index=weight_indexes)  # (batch_size, 1, conv_num)
                temp_node_feature = torch.gather(node_feature, dim=-1,
                                                 index=feature_indexes)  # (batch_size, hidden_dim, conv_num)
                cur_node_weight.append(temp_node_weight)
                cur_node_feature.append(temp_node_feature)
            cur_node_weight = torch.stack(cur_node_weight, dim=-1)  # (batch_size, 1, conv_num, filter_size)
            cur_node_feature = torch.stack(cur_node_feature, dim=-1)  # (batch_size, hidden_dim, conv_num, filter_size)
            cur_node_weight = torch.mean(cur_node_weight, dim=-1, keepdim=True)  # (batch_size, 1, conv_num, 1)
            cur_node_feature = cur_node_weight * cur_node_feature  # (batch_size, hidden_dim, conv_num, filter_size)
            CN = cur_node_feature.size(2)  # conv_num
            cur_node_feature = cur_node_feature.permute(0, 2, 3, 1).unsqueeze(-1).expand(BS, CN, ks, FD,
                                                                                         self.num_filters)  # (batch_size, conv_num, filter_size, hidden_dim, filter_num)
            cur_weight = self.kernel_weights[ki].expand(BS, CN, ks, FD,
                                                        self.num_filters)  # (batch_size, conv_num, filter_size, hidden_dim, filter_num)
            cur_bias = self.kernel_biases[ki].expand(BS, CN, self.num_filters)  # (batch_size, conv_num, filter_num)
            cur_node_feature = torch.sum(cur_node_feature * cur_weight,
                                         dim=(2, 3))  # (batch_size, conv_num, filter_num)
            cur_node_feature = cur_node_feature + cur_bias  # (batch_size, conv_num, filter_num)
            cur_node_feature = cur_node_feature.permute(0, 2, 1)  # (batch_size, filter_num, conv_num)
            cur_node_feature = F.max_pool1d(cur_node_feature, CN).squeeze()  # (batch_size, filter_num)
            cnn_outputs.append(cur_node_feature)
        ''' sentence representation '''
        sentence_feature = torch.cat(cnn_outputs, dim=-1)  # (batch_size, filter_num * num_of_filter_sizes)
        predicts = self.fc_out(self.fc_dropout(sentence_feature))
        return [predicts, weight_norm] if not self.opt.test else ([predicts, weight_norm], node_weight.squeeze())
