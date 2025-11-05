# Mask Architecture Anomaly Segmentation for Road Scenes Course Project Repository

## Overview

This is a starting repository for you project.

## Requiremetns Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation for training

Download the datasets below depending on which datasets you plan to use.  
You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training

To train EoMT from scratch (don't do it, it will be impossible to do it in Colab due to resource contraints):

```bash
python3 main.py fit \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This command trains the `EoMT-L` model with a 640×640 input size on COCO panoptic segmentation using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16. Switch to ```dinov3``` in the configuration path to enable the corresponding DINOv3 model.

✅ Make sure the total batch size is `devices × batch_size = 16`  
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

> This configuration takes ~6 hours on 4×NVIDIA H100 GPUs, each using ~26GB VRAM.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes.

> **DINOv3 Models**: When using DINOv3-based configurations, the code expects delta weights relative to DINOv3 weights by default. To disable this behavior and use absolute weights instead, add `--model.delta_weights False`. 

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This command evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

A [notebook](inference.ipynb) is available for quick inference and visualization with auto-downloaded pre-trained models.

> **DINOv3 Models**: When using DINOv3-based configurations, the code expects delta weights relative to DINOv3 weights by default. To disable this behavior and use absolute weights instead, add `--model.delta_weights False`. 

## Model Zoo

We provide pre-trained weights for both DINOv2- and DINOv3-based EoMT models.

- **[DINOv2 Models](model_zoo/dinov2.md)** - Original published results and pre-trained weights.
- **[DINOv3 Models](model_zoo/dinov3.md)** - New DINOv3-based models and pre-trained weights.

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccol\`{o} and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and {de Geus}, Daan},
  title     = {{Your ViT is Secretly an Image Segmentation Model}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## Acknowledgements

This project builds upon code from the following libraries and repositories:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache-2.0 License)  
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Apache-2.0 License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)
