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
    data = sub_sampling(data, n_samples=500, sub_sample=True)
    data.pos[:, 2] = normalize_time(data.pos[:, 2])
    # Radius graph generation.
    data.edge_index = radius_graph(data.pos, r=3.0, max_num_neighbors=32)
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

def for_in_bbox(sub_event, ratio=2):
    t_sub = sub_event[:,2].unsqueeze(1).to(device)
    x_sub = sub_event[:,0].unsqueeze(1).to(device)
    y_sub = sub_event[:,1].unsqueeze(1).to(device)
    p_sub = sub_event[:,3].unsqueeze(1).to(device)
    event_sub = torch.cat((x_sub, y_sub, t_sub, p_sub), dim=1)

    cur_bbox = gt_bbox[count_IMG] # x,y,w,h
    x, y, w, h = cur_bbox
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
    
    index_x1 = x_sub>=cur_bbox[0]
    index_x2 = x_sub<=cur_bbox[0]+cur_bbox[2]
    index_y1 = y_sub>=cur_bbox[1]
    index_y2 = y_sub<=cur_bbox[1]+cur_bbox[3]
    index = index_x1 *index_x2 *index_y1 *index_y2
    
    return index, event_sub

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

        # is_sorted = all(t_all[i]<=t_all[i+1] for i in range(len(t_all)-1))
        x_all, y_all, p_all, t_all = torch.chunk(dt, 4, dim=1)
        all_events = torch.cat(( x_all, y_all, t_all, p_all), dim=1)

        ## 
        sorted_indices = torch.argsort(all_events[:,2])
        all_events = all_events[sorted_indices].contiguous()
        t_all = all_events[:,2].contiguous()
        
        # all_events = all_events.numpy()

        time_length = all_events[-1,2] - all_events[0,2]
        
        deltaT = time_length / 499
        target_times = (t_all[0] + torch.arange(500) * deltaT).contiguous()  # 500
        start_idx = torch.searchsorted(t_all, target_times)
        start_idx = torch.clamp(start_idx, 0, len(all_events) - 1)
        start_idx = start_idx.tolist()

        W, H = (1280, 720) # eventvot
        count_IMG = 0
        for imgID in tqdm(range(len(start_idx)-1)):
            start_time_stamp = start_idx[imgID]
            end_time_stamp = start_idx[imgID+1]
            
            sub_event = all_events[start_time_stamp:end_time_stamp]
            t = t_all[start_time_stamp:end_time_stamp]
            if count_IMG == 0:
                ratio = 2
            else:
                ratio = 4
            index, event_sub = for_in_bbox(sub_event, ratio)
            event_sub = event_sub[index.squeeze(1),:]

            if count_IMG > 0:
                if event_sub.numel() == 0 or event_sub.shape[0] < 105:
                    index, event_sub = for_in_bbox(sub_event, ratio=8)
                    event_sub = event_sub[index.squeeze(1),:]
            
            if event_sub.numel() == 0:
                t_sub = sub_event[:,2].unsqueeze(1).to(device)
                x_sub = sub_event[:,0].unsqueeze(1).to(device)
                y_sub = sub_event[:,1].unsqueeze(1).to(device)
                p_sub = sub_event[:,3].unsqueeze(1).to(device)
                event_sub = torch.cat((x_sub, y_sub, t_sub, p_sub), dim=1)
            
            time_length = t[-1] - t[0]
            ## rescale the timestampes to start from 0 up to 1000
            t = (((t-t[0]).float() / time_length) * 1000).cuda()
            all_idx = torch.where(event_sub[:,3] != -1)     # all event
            neg_idx = torch.where(event_sub[:,3] == 0)       # neg event

            t = t[all_idx].unsqueeze(-1)
            x = event_sub[:,0][all_idx].unsqueeze(-1)
            y = event_sub[:,1][all_idx].unsqueeze(-1)
            p = event_sub[:,3][all_idx].unsqueeze(-1)
            p[neg_idx] = -1     # negtive voxel change from 0 to -1. because after append 0 operation.
            
            current_events = torch.cat((x, y, t, p), dim=1)
            
            current_data = load(current_events)
            
            process_data = pre_transform(current_data)

            bin_file = os.path.join(bin_save, 'frame{:0>4d}.bin'.format(count_IMG))

            torch.save(process_data.to("cpu"), bin_file)
            count_IMG += 1