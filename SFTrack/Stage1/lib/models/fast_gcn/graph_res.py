import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.nn.conv import SplineConv,GCNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from .max_pool import MaxPooling
from .max_pool_x import MaxPoolingX

import torch.nn as nn
from torch_sparse import SparseTensor


class GraphRes(torch.nn.Module):
    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(GraphRes, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        if dataset == "ncars" or dataset == "syn":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1" or dataset == "event_data":
            kernel_size = 8
            n = [1, 16, 64]
            pooling_outputs = 64
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        self.conv1 = GCNConv(n[0], n[1], normalize=False)
        self.norm1 = BatchNorm(in_channels=n[1])

        self.conv2 = GCNConv(n[1], n[2], normalize=False)
        self.norm2 = BatchNorm(in_channels=n[2])

        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.pool7 = MaxPoolingX(input_shape[:2] // 3, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)
        
        self.layer_norm1 = nn.LayerNorm(n[1])
        self.layer_norm2 = nn.LayerNorm(n[2])
        
        self.fc1 = Linear(n[1], out_features=num_outputs, bias=bias)
        self.fc2 = Linear(n[2], out_features=num_outputs, bias=bias)

    def conv_norm(self, data, conv_layer, norm_layer):
        # Apply convolution and normalization
        data.x = elu(conv_layer(data.x, data.edge_index))

        data.x = norm_layer(data.x)
        return data

    def forward(self, data: torch_geometric.data.Batch,  search=False) -> torch.Tensor:
        # Forward pass through the layers
        data = self.conv_norm(data, self.conv1, self.norm1)

        output1 = data.x
        output1 = torch.max(output1, dim=0, keepdim=True)[0]
        output1 = self.layer_norm1(output1)
        output1 = self.fc1(output1)
        
        data = self.conv_norm(data, self.conv2, self.norm2)
        
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        output2 = data.x
        output2 = torch.max(output2, dim=0, keepdim=True)[0]
        output2 = self.layer_norm2(output2)
        output2 = self.fc2(output2)

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)

        return [output1, output2, x]