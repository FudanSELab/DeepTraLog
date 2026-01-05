import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv
from torch.nn import Embedding
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import reset


class GGNN(nn.Module):
    def __init__(self, num_layers=5, hidden_dim=150):
        super(GGNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        # self.readout = GlobalAttention(gate_nn=nn.Sequential(
        #     nn.Linear(hidden_dim, 1),
        #     nn.Dropout(0.2), nn.ReLU(), nn.Sigmoid()))
        self.readout = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, 200), nn.Tanh(),
            nn.Linear(200, 1)))

    def forward(self, data):
        x = self.ggnn(data.x, data.edge_index)
        x, attention_scores = self.readout(x, data.batch)
        return x, attention_scores

class GGNN_NODE(nn.Module):
    def __init__(self, num_layers=5, hidden_dim=150):
        super(GGNN_NODE, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        # self.readout = GlobalAttention(gate_nn=nn.Sequential(
        #     nn.Linear(hidden_dim, 1),
        #     nn.Dropout(0.2), nn.ReLU(), nn.Sigmoid()))

    def forward(self, data):
        x = self.ggnn(data.x, data.edge_index)
        # x = self.readout(x, data.batch)
        return x


class GlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)

