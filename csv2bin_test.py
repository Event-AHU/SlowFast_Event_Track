import torch
import os
import numpy as  np
from tqdm import tqdm
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import FixedPoints
import math

def normalize_time(ts: torch.Tensor, beta: float = 0.5e-5) -> torch.Tensor:
    return (ts - torch.min(ts)) * beta

def load(events):
    x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
    return Data(x=x, pos=pos)


def pre_transform(data: Data) -> Data:
    data = sub_sampling(data, n_samples=25000, sub_sample=True)
    data.pos[:, 2] = normalize_time(data.pos[:, 2])

    return data

def sub_sampling(data: Data, n_samples: int, sub_sample: bool) -> Data:
    if sub_sample:
        sampler = FixedPoints(num=n_samples, allow_duplicates=False, replace=False)
        return sampler(data)
    else:
        sample_idx = np.arange(n_samples)
        for key, item in data:
            if torch.is_tensor(item) and item.size(0) != 1:
                data[key] = item[sample_idx]
        return data

if __name__ == '__main__':
    device = torch.device("cpu")
    csv_path = ''
    save_path =  ''
    csv_list = sorted(os.listdir(csv_path))

    for i in tqdm(range(len(csv_list))):
        video_file_path = os.path.join(csv_path, csv_list[i])
        if not csv_list[i].endswith("csv"):
            continue         
        foldName = os.path.splitext(csv_list[i])[0]
        print("============>> foldName: ", foldName)
        bin_save = os.path.join(save_path, foldName, foldName+'_bin')
        if not os.path.exists(bin_save):
            os.makedirs(bin_save)
        start_bin = os.path.join(bin_save,"frame0000.bin")
        if os.path.exists(start_bin):
            print("Skip this video : ", foldName)
            continue
        
        read_path = video_file_path    
        gt_path =  os.path.join(save_path, foldName, "groundtruth.txt")
        gt_bbox = np.loadtxt(gt_path,delimiter=',')    
  
        ## read csv;
        dt = pd.read_csv(read_path, dtype=np.int32, delimiter=",", usecols=(0, 1, 2, 3)).values
        dt = torch.tensor(dt, dtype=torch.int)

        x_all, y_all, p_all, t_all = torch.chunk(dt, 4, dim=1)
        all_events = torch.cat(( x_all, y_all, t_all, p_all), dim=1)

        ## 
        sorted_indices = torch.argsort(all_events[:,2])
        all_events = all_events[sorted_indices].contiguous()
        t_all = all_events[:,2].contiguous()
        
        time_length = all_events[-1,2] - all_events[0,2]
        
        deltaT = time_length / 499
        target_times = (t_all[0] + torch.arange(500) * deltaT).contiguous()  # 500
        start_idx = torch.searchsorted(t_all, target_times)
        start_idx = torch.clamp(start_idx, 0, len(all_events) - 1)
        start_idx = start_idx.tolist()

        count_IMG = 0
        for imgID in tqdm(range(len(start_idx)-1)):
            start_time_stamp = start_idx[imgID]
            end_time_stamp = start_idx[imgID+1]
            
            sub_event = all_events[start_time_stamp:end_time_stamp].cuda()
            t = t_all[start_time_stamp:end_time_stamp]
            
            time_length = t[-1] - t[0]
            ## rescale the timestampes to start from 0 up to 1000
            t = (((t-t[0]).float() / time_length) * 1000).cuda()
            all_idx = torch.where(sub_event[:,3] != -1)     # all event
            neg_idx = torch.where(sub_event[:,3] == 0)       # neg event

            t = t[all_idx].unsqueeze(-1)
            x = sub_event[:,0][all_idx].unsqueeze(-1)
            y = sub_event[:,1][all_idx].unsqueeze(-1)
            p = sub_event[:,3][all_idx].unsqueeze(-1)
            p[neg_idx] = -1     # negtive voxel change from 0 to -1. because after append 0 operation.

            current_events = torch.cat((x, y, t, p), dim=1)
            
            current_data = load(current_events)
            
            process_data = pre_transform(current_data)

            bin_file = os.path.join(bin_save, 'frame{:0>4d}.bin'.format(count_IMG))

            torch.save(process_data.to("cpu"), bin_file)
            count_IMG += 1