import torch
import os
import numpy as  np
from tqdm import tqdm
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph, knn_graph
from torch_geometric.transforms import FixedPoints

from dv import AedatFile
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
    data.edge_index = knn_graph(data.pos, k=8)
    
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
    t_sub = torch.tensor(sub_event['timestamp']).unsqueeze(1).to(device)
    x_sub = torch.tensor(sub_event['x']).unsqueeze(1).to(device)
    y_sub = torch.tensor(sub_event['y']).unsqueeze(1).to(device)
    p_sub = torch.tensor(sub_event['polarity']).unsqueeze(1).to(device)
    event_sub = torch.cat((x_sub, y_sub, t_sub, p_sub), dim=1)

    cur_bbox = gt_bbox[count_IMG] # x,y,w,h
    x, y, w, h = cur_bbox
    crop_sz = math.ceil(math.sqrt(max(0, w * h)) * ratio)
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
    use_mode = 'frame_exposure_time'
    video_path = ""
    save_path = ""
    videos_list = sorted(os.listdir(video_path))

    for i in tqdm(range(len(videos_list))):
        video_file_path = os.path.join(video_path, videos_list[i])
        if videos_list[i].endswith("txt"):
            continue         
        foldName = os.path.splitext(videos_list[i])[0]
        print("============>> foldName: ", foldName)
        bin_save = os.path.join(save_path, foldName, foldName+'_bin')
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

        # read aeda4;
        frame_all = []
        frame_exposure_time = []
        frame_interval_time = []
        with AedatFile(read_path) as f:
            # print(f.names)
            for frame in f['frames']:
                frame_all.append(frame.image)
                frame_exposure_time.append([frame.timestamp_start_of_exposure,
                                            frame.timestamp_end_of_exposure])  ## [1607928583397102, 1607928583401102]
                frame_interval_time.append([frame.timestamp_start_of_frame,
                                            frame.timestamp_end_of_frame])  ## [1607928583387944, 1607928583410285]
        if use_mode == 'frame_exposure_time':
            frame_timestamp = frame_exposure_time
        elif use_mode == 'frame_interval_time':
            frame_timestamp = frame_interval_time
        frame_num = len(frame_timestamp)
        
        if videos_list[i] == 'dvSave-2022_03_21_16_11_40':
            event_list = []
            try:
                for event in f['events'].numpy():
                    event_list.append(event)
            except RuntimeError as e:
                # error_flag = True
                print(f"Error reading events: {e}")
                events_back = np.hstack([packet for packet in f['events'].numpy()])
            events = np.hstack(event_list)
            events = np.hstack((events, events_back))
        else:
            events = np.hstack([packet for packet in f['events'].numpy()])
            
        t_all = torch.tensor(events['timestamp']).unsqueeze(1).to(device)
        x_all = torch.tensor(events['x']).unsqueeze(1).to(device)
        y_all = torch.tensor(events['y']).unsqueeze(1).to(device)
        p_all = torch.tensor(events['polarity']).unsqueeze(1).to(device)
        # all_events = torch.cat((x_all, y_all, t_all, p_all), dim=1)
        
        
        W, H = (346, 260) # coesot
        count_IMG = 0
        for frame_no in range(0, int(frame_num) - 1):
            start_idx = np.searchsorted(events['timestamp'], frame_timestamp[frame_no][0], side='left')
            end_idx = np.searchsorted(events['timestamp'], frame_timestamp[frame_no][1], side='left')
            sub_event = events[start_idx:end_idx]
            t = t_all[start_idx: end_idx]
            
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
            else:
                if event_sub.numel() == 0 or event_sub.shape[0] < 105:
                    index, event_sub = for_in_bbox(sub_event, ratio=4)
                    event_sub = event_sub[index.squeeze(1),:]
            
            if event_sub.numel() == 0:
                t_sub = torch.tensor(sub_event['timestamp']).unsqueeze(1).to(device)
                x_sub = torch.tensor(sub_event['x']).unsqueeze(1).to(device)
                y_sub = torch.tensor(sub_event['y']).unsqueeze(1).to(device)
                p_sub = torch.tensor(sub_event['polarity']).unsqueeze(1).to(device)
                event_sub = torch.cat((x_sub, y_sub, t_sub, p_sub), dim=1)
            
            p = 1
            while len(t)==0:
                start_idx = np.searchsorted(events['timestamp'], frame_timestamp[frame_no][0], side='left')
                end_idx = np.searchsorted(events['timestamp'], frame_timestamp[frame_no+p][1], side='left')
                sub_event = events[start_idx:end_idx]
                t = torch.tensor(sub_event['timestamp']).unsqueeze(1).to(device)
                x_sub = torch.tensor(sub_event['x']).unsqueeze(1).to(device)
                y_sub = torch.tensor(sub_event['y']).unsqueeze(1).to(device)
                p_sub = torch.tensor(sub_event['polarity']).unsqueeze(1).to(device)
                event_sub = torch.cat((x_sub, y_sub, t, p_sub), dim=1)
                p +=1
            
            time_length = t[-1] - t[0]
            # rescale the timestampes to start from 0 up todrft 1000
            t = ((t-t[0]).float() / time_length) * 1000
            all_idx = np.where(event_sub[:,3] != -1)
            neg_idx = np.where(event_sub[:,3] == 0)
            
            t = t[all_idx]
            x = event_sub[:,0][all_idx].unsqueeze(1)
            y = event_sub[:,1][all_idx].unsqueeze(1)
            p = event_sub[:,3][all_idx].unsqueeze(1)
            p[neg_idx] = -1     # negtive voxel change from 0 to -1. because after append 0 operation.
            current_events = torch.cat((x, y, t, p), dim=1) 
    
            current_data = load(current_events)
             
            process_data = pre_transform(current_data)
             
            bin_file = os.path.join(bin_save, 'frame{:0>4d}.bin'.format(count_IMG))
            
            torch.save(process_data.to("cpu"), bin_file)
            count_IMG += 1