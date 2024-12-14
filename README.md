# Enhancing Cross-modal Semantic Consistency via Key Token Alignment for Image-text Retrieval

## Introduction

The goal of cross-modal matching is to establish a strong link between vision and speech so that computers can better understand the semantic relationships between images and texts. As a key multimodal task, it improves the overall understanding of the model by optimising the matching between images and texts. Traditional fine-grained alignment methods typically rely on pre-trained target detectors to extract features from image regions and then match these regions to texts. While effective, this approach suffers from high computational complexity and error propagation during region detection and multi-stage training, which affects the efficiency and robustness of the model.

![Image text](https://github.com/ICME2025ITR/SCTA/blob/main/imgs/image-1.jpg)

In this paper, we find that some modifiers or redundant descriptions in the text can negatively affect the final alignment results. According to our observation, word redundancy in each text tends to bring unwanted interference, which hinders accurate alignment between key features of images and texts, leading to biased or distorted understanding of the semantics of the corresponding images. Therefore, we propose SCTA to improve cross-modal semantic consistency of image-text retrieval through key token alignment.SCTA addresses both image patch redundancy and text word redundancy, thus achieving fine-grained patch-word alignment through key tokens.

![Image text](https://github.com/ICME2025ITR/SCTA/blob/main/imgs/image-2.jpg)

## Preparation

### Environments
We recommended the following dependencies:

* python >= 3.9
* torch >= 1.12.0
* torchvision >= 0.13.0
* transformers >=4.32.0
* opencv-python
* tensorboard

### Datasets
We have prepared the caption files for two datasets in data/ folder, hence you just need to download the images of the datasets. The Flickr30K (f30k) images can be downloaded in [flickr30k-images](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). The MSCOCO (coco) images can be downloaded in [train2014](http://images.cocodataset.org/zips/train2014.zip), and [val2014](http://images.cocodataset.org/zips/val2014.zip). We hope that the final data are organized as follows:

```
data
├── coco
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── testall_ids.txt
│   ├── testall_caps.txt
│   └── id_mapping.json
│
├── f30k
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── test_ids.txt
│   ├── test_caps.txt
│   └── id_mapping.json
│
├── flickr30k-images
│
├── coco-images
│   ├── train2014
│   └── val2014

```

### Model Weights

Our framework needs to get the pre-trained weights for [BERT-base](https://huggingface.co/bert-base-uncased) and [ViT-base](https://huggingface.co/google/vit-base-patch16-224-in21k) models. You also can choose the weights downloaded by [transformers](https://github.com/huggingface/transformers) automatically (the weights will be downloaded at ```~/.cache```).

## Train

First, we set up the arguments, detailed information about the arguments is shown in ```arguments.py```.
* ```--dataset```: the chosen datasets, e.g., ```f30k``` and ```coco```.
* ```--data_path```: the root path of datasets, e.g., ```data/```.
* ```--multi_gpu```: whether to use the multiple GPUs (DDP) to train the models.
* ```--gpu-id```, the chosen GPU number, e.g., 0-3.
* ```--logger_name```, the path of logger files, e.g., ```runs/f30k_test``` or ```runs/coco_test```
We then run train.py to train the model. For batch size = 64, the model requires about 20,000 GPU memory (a 3090 GPU); for batch size = 128, the model requires about 40,000 GPU memory (an A40 GPU). You will need to modify the batch size depending on your hardware, and we also support multi-GPU training.


```
## single GPU

### vit + f30k 
python train.py --dataset f30k --gpu-id 0 --logger_name runs/f30k_vit --batch_size 128 --vit_type vit --embed_size 512 --img_sparse_ratio 0.5 --img_aggr_ratio 0.3 --cap_aggr_ratio 0.75

### vit + coco 
python train.py --dataset coco --gpu-id 0 --logger_name runs/coco_vit --batch_size 64 --vit_type vit --embed_size 512 --img_sparse_ratio 0.5 --img_aggr_ratio 0.3 --cap_aggr_ratio 0.75

## multiple GPUs

### vit + f30k
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 train.py --dataset f30k --multi_gpu 1 --logger_name runs/f30k_vit --batch_size 128 --vit_type vit --embed_size 512 --img_sparse_ratio 0.5 --img_aggr_ratio 0.3 --cap_aggr_ratio 0.75

### vit + coco
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 train.py --dataset coco --multi_gpu 1 --logger_name runs/coco_vit --batch_size 64 --vit_type vit --embed_size 512 --img_sparse_ratio 0.5 --img_aggr_ratio 0.3 --cap_aggr_ratio 0.75

```
## Evaluation

Run ```eval.py``` to evaluate the trained models on f30k or coco datasets, and you need to specify the model paths.
```
python eval.py --dataset f30k --data_path data/ --gpu-id 0
python eval.py --dataset coco --data_path data/ --gpu-id 1
```

## Performances

The following tables show the reproducing results of cross-modal retrieval on **MSCOCO** and **Flickr30K** datasets. We provide the training logs, checkpoints, performances, and hyper-parameters.

| Datasets | Visual encoders | I2T R@1 | I2T R@5 | T2I R@1 | T2I R@5 |
| :--- | :--- | :--- | :--- | :--- | :--- |  
| <p align="center">Flickr30K</p> | <p align="center">ViT</p> | <p align="center">78.5</p> | <p align="center">96.2</p> | <p align="center">65.6</p> | <p align="center">91.4</p> |
| <p align="center">MSCOCO-1K</p> | <p align="center">ViT</p> | <p align="center">82.3</p> | <p align="center">96.7</p> | <p align="center">68.7</p> | <p align="center">90.9</p> |
| <p align="center">MSCOCO-5K</p> | <p align="center">ViT</p> | <p align="center">55.8</p> | <p align="center">84.0</p> | <p align="center">43.7</p> | <p align="center">73.6</p> |
