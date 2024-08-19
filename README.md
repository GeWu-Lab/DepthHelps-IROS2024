## Depth Helps: Improving Pre-trained RGB-based Policy with Depth Information Injection
Authors: [Xincheng Pang](https://github.com/pangxincheng)\*, [Wenke Xia](https://xwinks.github.io/)\*, [Zhigang Wang](https://scholar.google.com/citations?hl=zh-CN&user=cw3EaAYAAAAJ&view_op=list_works&sortby=pubdate), [Bin Zhao](https://scholar.google.com/citations?user=DQB0hqwAAAAJ&hl=zh-CN), [Di Hu](https://dtaoo.github.io/)‡, [Dong Wang](https://scholar.google.es/citations?user=dasL9V4AAAAJ&hl=zh-CN)‡, [Xuelong Li](https://iopen.nwpu.edu.cn/info/1329/1171.htm)

The repo for "Depth Helps: Improving Pre-trained RGB-based Policy with Depth Information Injection", IROS 2024.

Due to deleting a lot of useless code, there may be issues during runtime. If you have any questions, please open an issue or send an email to xinchengpang@ruc.edu.cn.

#### Setup
```shell

# Step0: activate your conda.

# Step1: create a new conda virtual environment.
conda create -n depthHelps python=3.8.13

# Step2: Install torch==1.12.0
# WARNING: please use `nvidia-smi` to check if your nvidia dirver supports cuda11.3, If the `Driver Version>= 11.3`, you can directly use the following command to install torch.
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# Step2: Install the setuptools==57.5.0 as the pyhash in requirements.txt needs it.
pip install setuptools==57.5.0

# Step3: Install the library in requirements.txt
pip install -r requirements.txt

# Step4: Install the `libero` following [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
```

#### Dataset
- Download the LIBERO dataset following [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and put it in `data/LIBERO/v0`
- Preprocess dataset
```shell
python src/rerender_libero.py
```

#### Pretrained model
- Download the pretrained model following [RoboFlamingo](https://github.com/RoboFlamingo/RoboFlamingo) and put it in `RoboFlamingo`


#### Train

Before train, we suggest that you check if your directory is consistent with the directory structure below.
```
.
├── data
│   └──LIBERO
│      ├── v0
│      └── v1
├── README.md
├── metadata
│   └── libero.txt
├── requirements.txt
├── run.sh
├── RoboFlamingo
│   └── models
│       ├── bert-base-cased
│       ├── clip
│       ├── mpt-1b-redpajama-200b
│       ├── open_flamingo
│       │   └── checkpoint.pt
│       └── robo_flamingo
│           └── checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_3b_4.pth
└── src
    ├── convert_vq_ckpt.py
    ├── datasets
    │   ├── __init__.py
    │   ├── depth_libero_dataset.py
    │   └── libero_dataset.py
    ├── eval.py
    ├── main.py
    ├── merge_checkpoint.py
    ├── models
    │   ├── __init__.py
    │   └── roboflamingo
    │       ├── __init__.py
    │       ├── flamingo_lm.py
    │       ├── flamingo_mpt.py
    │       ├── flamingo_mpt_depth.py
    │       ├── flamingo_utils.py
    │       └── helpers.py
    ├── pred_depth.py
    ├── rerender_libero.py
    ├── train_vq.py
    └── utils
        ├── __init__.py
        └── common.py
```
All of our experiments were trained using 4 NVIDIA A100.

Please refernece to the `run.sh`

#### Eval
You need to install the `libero` following [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
```shell
python src/eval.py \
--log_dir runs/roboflamingo-mpt_3b_depth_depth_codebook_ema_finetune-DepthLiberoDataset \
--save_videos
```

#### Citation
```
@misc{pang2024depthhelpsimprovingpretrained,
  title={Depth Helps: Improving Pre-trained RGB-based Policy with Depth Information Injection}, 
  author={Xincheng Pang and Wenke Xia and Zhigang Wang and Bin Zhao and Di Hu and Dong Wang and Xuelong Li},
  year={2024},
  eprint={2408.05107},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2408.05107}, 
}
```