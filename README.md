## Attention Deficit is Ordered!
This codebase is the official implementation of the paper 'Attention Deficit is Ordered: Fooling Deformable Vision 
Transformers with Collaborative Adversarial Patches'. 

## Declaration
The code is provided as part of the supplementary material. Upon acceptance of the paper, the code will be made public 
in the anonymous Github repository mentioned in the introduction. 

## Content Overview
Apart from the code, we have also included the slightly modified (for parameter extraction) Deformable DeTr and MVDetr 
codebases with this repo. Pre-trained models are also provided, but the datasets are not included due to their 
substantial file sizes.

## Environment Setup

### Creating Virtual Environment
We encourage the reviewers to create a separate virtual environment for package installation using the following 
instructions:
```bash
conda create -n AttentionDeficit python=3.7
conda activate AttentionDeficit
```

### Installing Packages
To install the necessary packages there are two options. One can opt for the manual installation using the following 
instructions:
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
Two CUDA modules need to be installed for DeTr and MVDeTr. From this point onward all directories are assumed with 
respect to the content root `AttentionDeficit`.
```bash
cd AttDefDeTr/models/ops
bash make.sh
cd AttDefMVDeTr/multiview_detector/models/ops
bash make.sh
```

### Placing data inside correct folders
The dataset are not provided with the code, they need to be downloaded and placed in the correct folder to run the code 
without any modifications. 
The MS COCO dataset can be downloaded from 
[https://cocodataset.org](https://cocodataset.org) 
and the Wildtrack dataset can be downloaded from 
[https://www.epfl.ch/labs/cvlab/data/data-wildtrack/](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/). 
The dataset should be placed into the `data\dataset` folder, where `dataset` is the name of the dataset (e.g., 
`coco` or `wildtrack`). For example, the following directory structure is recommended for MS COCO:
```markdown
data
|___coco
    |___annotations
    |___train2017
    |___val2017
```
And, the following directory structure is recommended for Wildtrack:
```markdown
data
|___wildtrack
    |___annotations_positions
    |___calibrations
    |___image_subsets
    |___gt.txt
    |___README.txt
    |___rectangles.pom
    |___WILDTRACK.pdf
```

### Placing models into correct folders
The pretrained 6 layers Deformable DeTr model and 3 layers MVDeTr model is provided with the code. Additionally,
other variants of these models can be downloaded from the author websites. The models need to be placed in 
the correct folder to run the code without any modifications. The models should be placed in the `model/dataset`
folder, where `dataset` is the name of the dataset (e.g.,`coco` or `wildtrack`). For example, the following directory 
structure is recommended for Deformable DeTr:
```markdown
model
|___coco
    |___detr_6
```
And, the following directory structure is recommended for MVDeTr:
```markdown
model
|___wildtrack
    |___mvdetr_3
```

## Single-view Attention Deficit
The single-view attention deficit can be run with the default parameters by executing the following instruction:
```bash
python att_def_single_view.py
```
For information about the default parameters please consult the `arguments.py` file inside `AttDefDeTr` folder.
Additionally, the following options are available:
```markdown
--output: name of the output folder
--src_num: number of source patches
--src_size: size of source patches
--tar_num: number of target patches
--tar_size: size of target patches
--src_sampler: location sampler for source patches, options: uniformg, uniforml, random, custom
--tar sampler: location sampler for target patches, same options as above
--smoother: True/False, whether to use a continuous smoother function for loss
--aggregator: type of aggregator function to use on source patches, options: mean, norm, None
--epsilon: noise budget for patches
--constrainer: type of constrainer function to use on patches, options: l2, linf, None
--layers: number of layers used in the model
--attack: name of the attack, options: IP, OP, SP, CP, Att, Clean
--epoch: number of epochs
--num_batches_train: size of training batches
--num_batches_val: size of test batches
--visualize: True/False, whether to save visualization files
```
The `--src_sampler` and `--tar_sampler` can be assigned the following values:
```markdown
uniformg: uniform locations on a grid
uniforml: uniform locations on a line
random: random locations on a grid
custom: user-defined location
```
The `--aggregator` function can be of following types:
```markdown
mean: average of all the gradients
norm: gradient with the maximum L2 norm
None: no aggregation
```
The `--constrainer` function coupled with the noise budget `--epsilon` can be of the following types:
```markdown
l2: L2 norm
linf: L-infinity norm
None: no constrainer
```
Finally, the `--attack` can be of followin types:
```markdown
IP: Inward Pointer attack
OP: Outward Pointer attack
SP: Standalone Patch attack
CP: Collaborative Patch attack
Att: previous attention-based attacks
Clean: no attack
```

## Multi-view Attention Deficit
The multi-view attention deficit can be run with the default parameters by executing the following instruction:
```bash
python att_def_multi_view.py
```
For information about the default parameters please consult the `arguments.py` file inside `AttDefMVDeTr` folder.
Additionally, the following options are available:
```markdown
--output: name of the output folder
--src_cams: camera ids hosting source patches, example: 0 1 2 3 4 5 6
--src_nums: number of patches for each src_cam, example: 1 1 1 1 1 1 1
--src_size: size of the source patches
--tar_cams: camera ids hosting target patches, example: 0 1 2 3 4 5 6
--tar_nums: number of patches for each tar_cam, example: 1 1 1 1 1 1 1
--tar_size: size of the target patches
--src_sampler: location sampler for source patches, options: uniformg, uniforml, random, custom
--tar_sampler: location sampler for target patches, same options as above
--smoother: True/False, whether to use a continuous smoother function for loss
--aggregator: type of aggregator function to use on source patches, options: mean, norm, None
--epsilon: noise budget for patches
--constrainer: type of constrainer function to use on patches, options: l2, linf, None
--layers: number of layers used in the model
--attack:  name of the attack, options: IP, OP, SP, CP, Clean
--epoch: number of epochs
```



