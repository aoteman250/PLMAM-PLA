import torch
import numpy as np
from tqdm import tqdm
import metrics
import torch.nn as nn
import esm
from transformers import AutoModel
from functools import reduce
from cross_attention import EncoderLayer
import torch.nn.functional as F
class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=256,  name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
            nn.ReLU(),
            nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
        )
        self.name = name

    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return x + self.layers(x)


class ResDilaCNNBlocks(nn.Module):

    def __init__(self, feaSize, filterSize, blockNum=5, dilaSizeList=[1, 2, 4, 8, 16], dropout=0.2,
                 ):
        super(ResDilaCNNBlocks, self).__init__()  #
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize, filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}",
                                        ResDilaCNNBlock(dilaSizeList[i % len(dilaSizeList)], filterSize,
                                                        ))
        self.act = nn.ReLU()



    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x)  # => batchSize × seqLen × filterSize
        x = self.blockLayers(x.transpose(1, 2))  # => batchSize × seqLen × filterSize


        x = self.act(x.transpose(1, 2))  # => batchSize × seqLen × filterSize


        return x



class SKConv(nn.Module):
    def __init__(self, in_channels=320, out_channels=256, stride=1, M=2, r=8, L=32):
        '''
        :param in_channels: 输入通道维度
        :param out_channels: 输出通道维度，输入输出通道维度相同
        :param M: 分支数
        :param r: 特征Z的长度，计算其维度d时所需的比率
        :param L: 默认为32
        采用分组卷积： groups = 32，所以输入channel的数值必须是group的整数倍
        '''
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)  # 计算从向量C降维到向量Z的长度d
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()  # 根据分支数量添加不同核的卷积操作
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)  # 自适应池化到指定维度，这里指定为1，实现GAP
        self.fc1 = nn.Sequential(
            nn.Conv1d(out_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
        )  # 降维
        self.fc2 = nn.Conv1d(d, out_channels * M, kernel_size=1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1使得多个全连接层对应位置进行softmax,保证对应位置a+b+..=1

    def forward(self, input):
        input = input.transpose(1, 2)
        batch_size = input.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            output.append(conv(input))  # [batch_size, out_channels, d]

        # the part of fusion
        U = reduce(lambda x, y: x + y, output)  # 逐元素相加生成混合特征U [batch_size, channel, d]
        s = self.global_pool(U)  # [batch_size, channel, 1]
        z = self.fc1(s)  # S->Z降维 [batch_size, d, 1]
        a_b = self.fc2(z)  # Z->a，b升维 [batch_size, out_channels * M, 1]
        a_b = a_b.view(batch_size, self.M, self.out_channels, -1)  # 调整形状 [batch_size, M, out_channels, 1]
        a_b = self.softmax(a_b)  # 使得多个全连接层对应位置进行softmax [batch_size, M, out_channels, 1]

        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))  # 分割成多个 [batch_size, 1, out_channels, 1], [batch_size, 1, out_channels, 1]
        a_b = list(map(lambda x: x.view(batch_size, self.out_channels, 1), a_b))  # 调整形状 [batch_size, out_channels, 1]
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应不同卷积核输出的U逐元素相乘
        V = reduce(lambda x, y: x + y, V)  # 多个加权后的特征逐元素相加 [batch_size, out_channels, d]
        # V = self.max_pool(V).squeeze(2)
        # V = Reduce('b c t -> b c', 'max')(V)
        V = V.transpose(1, 2)
        return V  # [batch_size, out_channels, d]



class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """

    def __init__(self, embed_size, head_num, dropout, residual=True):
        """
        """
        super(MultiHeadAttentionInteract, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num

        self.W_Q = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))
        self.W_K = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))
        self.W_V = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))
        self.act = nn.ReLU()

        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """

        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))

        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))

        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5

        attn_w = F.softmax(inner, dim=-1)

        attn_w = F.dropout(attn_w, p=self.dropout)

        results = torch.matmul(attn_w, Value)

        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)

        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))

        results = self.act(results)

        return results





class Selfattention(nn.Module):

    def __init__(self, field_dim, embed_size, head_num, dropout=0.1):
        super(Selfattention, self).__init__()


        hidden_dim = 1024

        self.vec_wise_net = MultiHeadAttentionInteract(embed_size=embed_size,
                                                       head_num=head_num,
                                                       dropout=dropout)

        self.trans_vec_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
        )

    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """

        b, f, e = x.shape
        vec_wise_x = self.vec_wise_net(x).reshape(b, f * e)
        m_vec = self.trans_vec_nn(vec_wise_x)

        m_x = m_vec
        return m_x





class MultiViewNet(nn.Module):

    def __init__(self,embed_dim=256):
        super(MultiViewNet, self).__init__()
        self.model = AutoModel.from_pretrained("/data/stu1/saj_pycharm_project/PL2/MoLFormer-XL-both-10pct",
                                               trust_remote_code=True
                                               )

        self.esmmodel,_ = esm.pretrained.esm2_t6_8M_UR50D()
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.esmmodel.parameters():
            param.requires_grad = False

        self.SKConv = SKConv(in_channels=320)
        self.SKConv1 = SKConv(in_channels=768)
        self.embed_smile = nn.Embedding(65, embed_dim)
        self.embed_prot = nn.Embedding(26, embed_dim)
        self.onehot_smi_net = ResDilaCNNBlocks(embed_dim, embed_dim)
        self.onehot_prot_net = ResDilaCNNBlocks(embed_dim, embed_dim)

        self.smi_attention_poc = EncoderLayer(256, 256, 0.1, 0.1, 4)  #交叉注意力机制
        self.seq_attention_tdlig = EncoderLayer(256, 256, 0.1, 0.1, 4)

        proj_dim = 256
        field_dim = 4
        self.feature_interact = Selfattention(field_dim=field_dim, embed_size=proj_dim, head_num=8)


        self.transform = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU())

        self.norm = nn.LayerNorm(proj_dim)

    def forward(self,smile,sequence,smi,seq):

        smile_vectors_onehot = self.embed_smile(smile)
        proteinFeature_onehot = self.embed_prot(sequence)
        compoundFeature_onehot1 = self.onehot_smi_net(smile_vectors_onehot)
        proteinFeature_onehot1 = self.onehot_prot_net(proteinFeature_onehot)
        #
        smiles_embedding = self.model(smi)[0]
        logits = self.SKConv1(smiles_embedding)

        seq_embedding = self.esmmodel(seq, repr_layers=[6], return_contacts=True)
        token_representations = seq_embedding["representations"][6]
        logits1 = self.SKConv(token_representations)

        drug = self.smi_attention_poc(compoundFeature_onehot1,proteinFeature_onehot1,proteinFeature_onehot1)
        pro = self.seq_attention_tdlig(proteinFeature_onehot1,compoundFeature_onehot1,compoundFeature_onehot1)
        drug1 = self.smi_attention_poc(logits,logits1,logits1)
        pro1 = self.seq_attention_tdlig(logits1,logits,logits)


        all_features = torch.stack([drug,pro,drug1,pro1], dim=2)
        all_features = self.norm(all_features.permute(0, 2, 1))
        all_features = self.feature_interact(all_features)

        out = self.transform(all_features)
        return out





def test(model: nn.Module, test_loader, loss_function, device, show, _p):
    path = 'result-free/'
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            smile = x[0].to(device)
            sequence = x[1].to(device)
            smi = x[2].to(device)
            seq = x[3].to(device)


            y = y.to(device)

            y_hat = model(smile,sequence,smi,seq  )

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    np.savetxt(path + _p  + 'targets.csv', targets)  # , fmt ='%d'
    np.savetxt(path + _p  + 'outputs.csv', outputs)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
        'pearson': metrics.get_pearson(targets, outputs),
    }

    return evaluation

#     return evaluation