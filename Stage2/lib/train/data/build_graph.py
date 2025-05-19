import torch
import torch.nn as nn
import numpy as np
import math

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph, knn_graph
from torch_geometric.transforms import FixedPoints

class Build_Graph(nn.Module):
    def __init__(self):
        super(Build_Graph, self).__init__()
    
    def forward(self, data, gt, H, W, ratio=2, return_multi_results=False):

        data = self.for_in_bbox(data, gt, H, W, ratio, return_multi_results)    
        
        return data
    
    def for_in_bbox(self, bin_event, gt_bbox, W, H, ratio, return_multi_results):
        t_sub = bin_event.pos[:,2]
        x_sub = bin_event.pos[:,0]
        y_sub = bin_event.pos[:,1]
        p_sub = bin_event.x.squeeze(1)
        event_sub = torch.stack((x_sub, y_sub, t_sub, p_sub), dim=1)

        cur_bbox = gt_bbox[0] # x,y,w,h
        x, y, w, h = cur_bbox

        x, y, w, h = int(x), int(y), int(w), int(h)
        crop_sz = math.ceil(math.sqrt(w * h) * ratio)
        
        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz

        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - W + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - H + 1, 0)

        # Crop target
        crop_coor = [x1 + x1_pad, x2 - x2_pad, y1 + y1_pad, y2 - y2_pad]
        cur_bbox = (crop_coor[0], crop_coor[2], crop_coor[1] - crop_coor[0], crop_coor[3] - crop_coor[2])
        cur_bbox =  np.array(cur_bbox, dtype=np.int_)
        
        cur_bbox[0] = np.clip(cur_bbox[0], 0 ,W)
        cur_bbox[1] = np.clip(cur_bbox[1], 0 ,H)
        cur_bbox[2] = np.clip(cur_bbox[2], 0 ,W-cur_bbox[0])
        cur_bbox[3] = np.clip(cur_bbox[3], 0 ,H-cur_bbox[1])

        index = (x_sub >= cur_bbox[0]) & (x_sub <= cur_bbox[0] + cur_bbox[2]) & \
                (y_sub >= cur_bbox[1]) & (y_sub <= cur_bbox[1] + cur_bbox[3])
       
        event_sub = event_sub[index]    
        if event_sub.numel() < 9:
           return None

        if return_multi_results:
            data_list = self.generate_graphs_from_event(event_sub)
            return data_list
        else:
            if len(event_sub) > 300:
                indices = torch.randperm(len(event_sub))[:300]
                event_sub = event_sub[indices]

            event_sub[:, 2] = self.normalize_time(event_sub[:, 2])
            
            pos = event_sub[:, :3]
            edge_index = knn_graph(pos, k=8)
            
            return Data(x=event_sub[:, 3:], pos=pos, edge_index=edge_index)
    
    def generate_graphs_from_event(self, event_sub):
        num_total = min(event_sub.shape[0], 300)
        indices = torch.randperm(len(event_sub))[:num_total]
        sampled = event_sub[indices]

        sorted_event_sub = sampled[sampled[:, 2].argsort()]

        chunk_size = num_total // 3
        graphs = []

        for i in range(3):
            end = (i + 1) * chunk_size if i < 2 else num_total
            sub = sorted_event_sub[:end].clone()
            
            sub[:, 2] = self.normalize_time(sub[:, 2])

            pos = sub[:, :3]
            edge_index = knn_graph(pos, k=8)

            data = Data(x=sub[:, 3:], pos=pos, edge_index=edge_index)
            graphs.append(data)

        return graphs


    def load(self, events):
        x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
        return Data(x=x, pos=pos)
    
    def normalize_time(self, ts: torch.Tensor, beta: float = 0.5e-5) -> torch.Tensor:
        return (ts - torch.min(ts)) * beta
    
    def sub_sampling(self, data: Data, n_samples: int, sub_sample: bool) -> Data:
        if sub_sample:
            sampler = FixedPoints(num=n_samples, allow_duplicates=False, replace=False)
            return sampler(data)
        else:
            sample_idx = np.arange(n_samples)
            for key, item in data:
                if torch.is_tensor(item) and item.size(0) != 1:
                    data[key] = item[sample_idx]
            return data