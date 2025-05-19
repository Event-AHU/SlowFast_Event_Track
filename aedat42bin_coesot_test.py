import torch
import os
import numpy as  np
from tqdm import tqdm
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import FixedPoints

from dv import AedatFile
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
    use_mode = 'frame_exposure_time'
    device = torch.device("cpu")
    video_path = ""
    save_path = ""
    videos_list = sorted(os.listdir(video_path))

    for i in tqdm(range(len(videos_list))):
        video_file_path = os.path.join(video_path, videos_list[i])
        if videos_list[i].endswith("txt"):
            continue         
        foldName = os.path.splitext(videos_list[i])[0]
        print("============>> foldName: ", foldName)
        bin_save = os.path.join(save_path, foldName, foldName+'_25wbin')
        if not os.path.exists(bin_save):
            os.makedirs(bin_save)
        end_bin = os.path.join(bin_save,"frame0000.bin")
        if os.path.exists(end_bin):
            print("Skip this video : ", foldName)
            continue

        aedat4_file = foldName + '.aedat4'
        read_path = os.path.join(video_file_path, aedat4_file) 
        gt_path =  os.path.join(video_file_path, "groundtruth.txt")
        gt_bbox = np.loadtxt(gt_path,delimiter=',')
        
        # read aeda4
        frame_all = []
        frame_exposure_time = []
        frame_interval_time = []
        with AedatFile(read_path) as f:
            for frame in f['frames']:
                frame_all.append(frame.image)
                frame_exposure_time.append([frame.timestamp_start_of_exposure,
                                            frame.timestamp_end_of_exposure])  ## [1607928583397102, 1607928583401102]
                frame_interval_time.append([frame.timestamp_start_of_frame,
                                            frame.timestamp_end_of_frame])    ## [1607928583387944, 1607928583410285]
        if use_mode == 'frame_exposure_time':
            frame_timestamp = frame_exposure_time
        elif use_mode == 'frame_interval_time':
            frame_timestamp = frame_interval_time
        frame_num = len(frame_timestamp)

        events = np.hstack([packet for packet in f['events'].numpy()])

        t_all = torch.tensor(events['timestamp']).unsqueeze(1).to(device)
        x_all = torch.tensor(events['x']).unsqueeze(1).to(device)
        y_all = torch.tensor(events['y']).unsqueeze(1).to(device)
        p_all = torch.tensor(events['polarity']).unsqueeze(1).to(device)
        # all_events = torch.cat((x_all, y_all, t_all, p_all), dim=1)
        
        count_IMG = 0
        for frame_no in range(0, int(frame_num) - 1):
            start_idx = np.searchsorted(events['timestamp'], frame_timestamp[frame_no][0], side='left')
            end_idx = np.searchsorted(events['timestamp'], frame_timestamp[frame_no][1], side='left')
            sub_event = events[start_idx:end_idx]
            t = t_all[start_idx: end_idx]
            if len(t) == 0:
                empty_pos = torch.empty((0, 3), dtype=torch.float)      
                empty_x = torch.empty((0, 1), dtype=torch.float)        

                process_data = Data(pos=empty_pos, x=empty_x)
                
                bin_file = os.path.join(bin_save, 'frame{:0>4d}.bin'.format(count_IMG))
            
                torch.save(process_data.to("cpu"), bin_file)
                count_IMG += 1
                
                continue 
                
            time_length = t[-1] - t[0]
            # rescale the timestampes to start from 0 up to 1000
            t = ((t-t[0]).float() / time_length) * 1000
            all_idx = np.where(sub_event['polarity'] != -1)      # all event
            neg_idx = np.where(sub_event['polarity'] == 0)       # neg event
            t = t[all_idx]
            x = x_all[start_idx:end_idx][all_idx]
            y = y_all[start_idx:end_idx][all_idx]
            p = p_all[start_idx:end_idx][all_idx]
            p[neg_idx] = -1     # negtive voxel change from 0 to -1. because after append 0 operation.
            current_events = torch.cat((x, y, t, p), dim=1)
            
            current_data = load(current_events)
            process_data = pre_transform(current_data)
            bin_file = os.path.join(bin_save, 'frame{:0>4d}.bin'.format(count_IMG))
            torch.save(process_data.to("cpu"), bin_file)
            count_IMG += 1