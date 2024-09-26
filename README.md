# TypeFSL  

## Introduction 

This is the repository for TypeFSL: Type Prediction from Binaries via Inter-procedural Data-flow Analysis and Few-shot Learning.

We will actively update it.  


## Requirements:  

- Python >= 3.7
- PyTorch == 1.1.0
- numpy == 1.15.4
- torchtext == 0.4.0
- pytorch-transformers == 1.1.0
- termcolor == 1.1.0
- tqdm == 4.32.2
- Binary Ninja (business)


## Installation

We recommend `conda` to setup the environment and install the required packages. Conda installation instructions can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html). The following setup assumes Conda is installed and is running on Linux system (though Windows should work too).

First, create the conda environment,

`conda create -n TypeFSL python=3.7`

and activate the conda environment:

`conda activate TypeFSL`

If you want to train the model with your gpu (assume CUDA 10.0 is used):

`conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch`

Depending on your CUDA version, you need to select different versions of `cudatoolkit` and `cudnn`. Details can be seen in [Pytorch](https://pytorch.org/get-started/previous-versions/).

If you want to use CPU version, we can:

`conda install pytorch==1.1.0`

## Dataset Preparation (optional)

In this repository, we provided raw datasets and source code datasets in [`Dataset/`](./Dataset/). 

We also prepare a pre-processed x64 dataset in O0 optimization in [`FSL_train/DS/data.inter/`](./FSL_train/DS/data.inter), which can be directly used for few-shot learning evaluation. Details are shown in next section.

### Pre-process the Dataset

Scripts are provide in [`Dataset_collection/`](./Dataset_collection/).

0. Unpack the packages in [`Dataset/Source_Dataset/Package`](./Dataset/Source_Dataset/Package).
1. Run **./use_ctags.sh** and **python collect_source_type.py**.
   - To generate ctags file, with function prototype definition, then obtain the ground truth (i.e., data type of arguments) for each project.
   - Please make sure the codes are in [`Dataset/Source_Dataset/Code/`](./Dataset/Source_Dataset/Code/) and prepare the result folder [`Dataset/Source_Dataset/Index/`](./Dataset/Source_Dataset/Index/).

2. Run **batch_slices.sh [normal/inter]**.
   - Start program slicing.
   - The "normal" option means do not apply inter-procedural analysis (i.e., withou IPDA).
   - The "inter" option means apply the inter-procedural analysis (i.e., with IPDA).
   - If you want to process a specific dataset, try generate_slice.py

3. Run **batch_label.sh [normal/inter]**.
   - To generate training and testing samples for machine learning.
   - The "normal" option means do not apply inter-procedural analysis (i.e., withou IPDA).
   - The "inter" option means apply the inter-procedural analysis (i.e., with IPDA).
   - If you want to process a specific dataset, try generate_label.py

4. Move the **asm_[arch]_[opt].json** files into [`FSL_train/DS/`](./FSL_train/DS/)
   - Depending on the used module (normal/inter), dest folder is [`data.normal/`](./FSL_train/DS/data.normal) or [`data.inter/`](./FSL_train/DS/data.inter).

## FSL Model Training and Testing

Utilizing the few-shot learning for type prediction. Scripts are provided in [`FSL_train/`](./FSL_train/).


To train and test a FSL model with the dataset in [`data.inter/`](./FSL_train/DS/data.inter):

```
bash train_inter_asm.sh [arch] [opt] [way] [shot] 
```

For example, you can try to analyze the x64 datasets with O0 optimitzation in 20-way 10-shot classification:

```
bash train_inter_asm.sh x86-64 O0 20 10
```

The results will be stored at [`FSL_train/FSL_results/`](./FSL_train/FSL_results/).