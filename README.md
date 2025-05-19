# :dart: Abstract 
Existing tracking algorithms typically rely on low-frame-rate RGB cameras coupled with computationally intensive deep neural network architectures to achieve effective tracking. However, such frame-based methods inherently face challenges in achieving low-latency performance and often fail in resource-constrained environments. Visual object tracking using bio-inspired event cameras has emerged as a promising research direction in recent years, offering distinct advantages for low-latency applications. In this paper, we propose a novel Slow-Fast Tracking paradigm that flexibly adapts to different operational requirements, termed SFTrack. The proposed framework supports two complementary modes, i.e., a high-precision slow tracker for scenarios with sufficient computational resources, and an efficient fast tracker tailored for latency-aware, resource-constrained environments. Specifically, our framework first performs graph-based representation learning from high-temporal-resolution event streams, and then integrates the learned graph-structured information into two FlashAttention-based vision backbones, yielding the slow and fast trackers, respectively. The fast tracker achieves low latency through a lightweight network design and by producing multiple bounding box outputs in a single forward pass. Finally, we seamlessly combine both trackers via supervised fine-tuning and further enhance the fast tracker’s performance through a knowledge distillation strategy. Extensive experiments on public benchmarks, including FE240, COESOT, and EventVOT, demonstrate the effectiveness and efficiency of our proposed method across different real-world scenarios.

# :hammer: Environment 

Install env
```
conda create -n sftrack python=3.10
conda activate sftrack
bash install.sh
pip install flash-attn==2.7.3 --no-build-isolation
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
```

You can modify paths by editing these files
```
Stage1/lib/train/admin/local.py  # paths about training
Stage1/lib/test/evaluation/local.py  # paths about testing

Stage2/lib/train/admin/local.py  # paths about training
Stage2/lib/test/evaluation/local.py  # paths about testing
```

## Train & Test
```
# Stage1 (You can choose the Slow Tracker or the Fast Tracker in "Stage1/experiments/sftrack/**.yaml")
bash train.sh 
bash test.sh

# Stage2 (You can choose the Slow Tracker or the Fast Tracker in "Stage2/experiments/sftrack/**.yaml")
bash train.sh 
bash test.sh
```

### Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX 4090 GPU.
