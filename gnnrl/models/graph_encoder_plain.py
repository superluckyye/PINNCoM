import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class graph_encoder_pyg(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature):
        super(graph_encoder_pyg, self).__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature

        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.linear1 = nn.Linear(hidden_feature, out_feature)
        self.tanh = nn.Tanh()
        self.relu = torch.relu

    def forward(self, Graph):

        x, edge_index, batch = Graph.x, Graph.edge_index, Graph.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        embedding = global_mean_pool(x, batch)
        embedding = self.linear1(embedding)
        embedding = self.tanh(embedding)

        return embedding

if __name__ == '__main__':
    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1],
                               [2, 3],
                               [2, 4],
                               [2, 5]], dtype=torch.long)
    # print(edge_index.shape)
    x = torch.randn([6, 50])
    batch = torch.zeros([6, 1]).long()
    #x.squeeze()
    # contiguous() 可以理解为深拷贝，强制拷贝一份tensor与之前的没有联系
    G = Data(x=x, edge_index=edge_index.t().contiguous())
    mode = graph_encoder_pyg(50, 20, 15)
    a = mode(G)
    # a的输出维度与输入的out_feature相关
    # print(a.shape)
    # unsqueeze(0) 扩展维度
    # print(a.unsqueeze(0).shape)
    # print(a)