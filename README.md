# A Part Power Set Model for Scale-Free Person Retrieval

By [Yunhang Shen](), [Rongrong Ji](http://mac.xmu.edu.cn/rrji-en.html), [Xiaopeng Hong](https://hongxiaopeng.com/), [Feng Zheng](https://scholar.google.com/citations?user=PcmyXHMAAAAJ), [Xiaowei Guo](), [Yongjian Wu](), [Feiyue Huang]().

IJCAI 2019 Paper

This project is based on [Detectron](https://github.com/facebookresearch/Detectron).


## Introduction

PPS is an end-to-end part power set model with multi-scale features, which captures the discriminative parts of pedestrians from global to local, and from coarse to fine, enabling part-based scale-free person re-ID.
In particular, PPS first factorize the visual appearance by enumerating $k$-combinations for all $k$ of $n$ body parts to exploit rich global and partial information to learn discriminative feature maps.
Then, a combination ranking module is introduced to guide the model training with all combinations of body parts, which alternates between ranking combinations and estimating an appearance model.
To enable scale-free input, we further exploit the pyramid architecture of deep networks to construct multi-scale feature maps with a feasible amount of extra cost in term of memory and time.


## License

PPS is released under the [Apache 2.0 license](https://github.com/shenyunhang/PPS/blob/PPS/LICENSE). See the [NOTICE](https://github.com/shenyunhang/PPS/blob/PPS/NOTICE) file for additional details.


## Citing PPS

If you find PPS useful in your research, please consider citing:

```
@inproceedings{shen2019pps,
    author = {Yunhang Shen and Rongrong Ji and Xiaopeng Hong and Feng Zheng and Xiaowei Guo and Yongjian Wu and Feiyue Huang},
    title = {{A Part Power Set Model for Person Retrieval with Multi-Scale Features}},
    booktitle = {IJCAI},
    year = {2019},
}   
```


## Installation

**Requirements:**

- NVIDIA GPU, Linux, Python2
- Caffe2 in pytorch v1.0.1, various standard Python packages, and the COCO API; Instructions for installing these dependencies are found below

### Caffe2

Clone the pytorch repository:

```
# pytorch=/path/to/clone/pytorch
git clone https://github.com/pytorch/pytorch.git $pytorch
cd $pytorch
git checkout v1.0.1
git submodule update --init --recursive
```

Install Python dependencies:

```
pip install -r $pytorch/requirements.txt
```

Build caffe2:

```
cd $pytorch && mkdir -p build && cd build
cmake ..
sudo make install
```


### Other Dependencies

Install the [COCO API](https://github.com/cocodataset/cocoapi):

```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python setup.py install --user
```

Note that instructions like `# COCOAPI=/path/to/install/cocoapi` indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (`COCOAPI` in this case) accordingly.

Install the [pycococreator](https://github.com/waspinator/pycococreator):

```
pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
```


### PPS

Clone the PPS repository:

```
# PPS=/path/to/clone/PPS
git clone https://github.com/shenyunhang/PPS.git $PPS
cd $PPS
```

Install Python dependencies:

```
pip install -r requirements.txt
```

Set up Python modules:

```
make
```

Build the custom operators library:

```
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_FLAGS="-isystem $pytorch/third_party/eigen -isystem $/pytorch/third_party/cub"
make
```


### Dataset Preparation
Please follow [this](https://github.com/huanghoujing/beyond-part-models/blob/master/README.md#dataset-preparation) to transform the original datasets (Market1501, DukeMTMC-reID and CUHK03) to PCB format.

After that, we assume that your dataset copy at `~/Dataset` has the following directory structure:

```
market1501
|_ images
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ partitions.pkl
|_ train_test_split.pkl
|_ ...
duke
|_ images
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ partitions.pkl
|_ train_test_split.pkl
|_ ...
cuhk03
|_ detected
|  |_ images
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ partitions.pkl
|_ labeled
|  |_ images
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|     |_ partitions.pkl
|_ re_ranking_train_test_split.pkl
|_ ...
...
```

Generate the COCO Json files, which is used in Detectron:
```
cd $PPS
python tools/bpm_to_coco.py
```
You may need to modify the paths of datasets in tools/bpm_to_coco.py if you put datasets in different locations.

After that, check that you have trainval.json and test.json for each datatset in their corresponding locations.

Create symlinks：
```
cd $PPS/detectron/datasets/data/
ln -s ~/Dataset/market1501 market1501
ln -s ~/Dataset/duke duke
ln -s ~/Dataset/cuhk03 cuhk03
```


### Model Preparation

Download ResNet50 model (ResNet-50-model.caffemodel and ResNet-50-deploy.prototxt) from this [link](https://github.com/KaimingHe/deep-residual-networks)
```
cd $PPS
mkdir -p ~/Dataset/model
python tools/pickle_caffe_blobs_keep_bn.py --prototxt /path/to/ResNet-50-deploy.prototxt --caffemodel /path/to/ResNet-50-model.caffemodel --output ~/Dataset/model/R-50_BN.pkl
```

Noted that this requires to instal caffe1 separately, as caffe1 specific proto is removed in pytorch v1.0.1. 
See [this](https://github.com/pytorch/pytorch/commit/40109b16d0df8248bc01ad08c7ab615310c52d67).

You can download what I have transformed for the project from this [link](https://1drv.ms/u/s!AodeRhn8mpxoh0TIj0K7eaUWyREj?e=VuGeGe).

You may also need to modify the below config files to point TRAINING.WEIGHTS to R-50_BN.pkl.


## Quick Start: Using PPS

### market1501

```
CUDA_VISIBLE_DEVICES=0 ./scripts/train_reid.sh --cfg configs/market1501/pps_crm_triplet_R-50_1x.yaml OUTPUT_DIR experiments/pps_crm_triplet_market1501_`date +'%Y-%m-%d_%H-%M-%S'`
```

### duke

```
CUDA_VISIBLE_DEVICES=0 ./scripts/train_reid.sh --cfg configs/duke/pps_crm_triplet_R-50_1x.yaml OUTPUT_DIR experiments/pps_crm_triplet_duke_`date +'%Y-%m-%d_%H-%M-%S'`
```

### cuhk03

```
CUDA_VISIBLE_DEVICES=0 ./scripts/train_reid.sh --cfg configs/cuhk03/pps_crm_triplet_R-50_1x.yaml OUTPUT_DIR experiments/pps_crm_triplet_cuhk03_`date +'%Y-%m-%d_%H-%M-%S'`
```
