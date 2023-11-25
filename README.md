## Attention Deficit is Ordered!
This codebase is the official implementation of the paper 'Attention Deficit is Ordered: Fooling Deformable Vision Transformers with Collaborative Adversarial Patches'. 

## Declaration
The code is provided as part of the supplementary material. Upon acceptance of the paper, the code will be made public in the anonymous Github repository mentioned in the introduction. 

## Content Overview
Apart from the code, we have also included the slightly modified (for parameter extraction) Deformable DeTr and MVDetr codebases with this repo. Pre-trained models are also provided, but the datasets are not included due to their substantial file sizes.

## Environment Setup

### Creating Virtual Environment
We encourage the reviewers to create a separate virtual environment for package installation using the following instructions:
```bash
conda create -n AttentionDeficit python=3.7
conda activate AttentionDeficit
```

### Installing Packages
To install the necessary packages there are two options. One can opt for the manual installation using the following instructions:
```bash
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install pycocotools tqdm cython scipy matplotlib pillow opencv-python kornia==0.5.0 shapely
cd ./AttDefDeTr/models/ops/; bash make.sh
cd ./AttDefMVDeTr/multiview_detector/models/ops; bash make.sh
```
Or, install using the `requirements.txt` file provided with the code.
```bash
pip install -r requirements.txt
```

### Installing CUDA modules
Two CUDA modules need to be installed for DeTr and MVDeTr. From this point onward all directories are assumed with respect to the content root `AttentionDeficit`.
```bash
cd AttDefDeTr/models/ops
bash make.sh
cd AttDefMVDeTr/multiview_detector/models/ops
bash make.sh
```

### Placing data inside correct folders
The dataset are not provided with the code, they need to be downloaded and placed in the correct folder to run the code 
without any modifications. The MS COCO dataset can be downloaded from [https://cocodataset.org](https://cocodataset.org) 
and the Wildtrack dataset can be downloaded from [https://www.epfl.ch/labs/cvlab/data/data-wildtrack/](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/). 