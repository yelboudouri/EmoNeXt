# EmoNeXt: an Adapted ConvNeXt for facial Emotion Recognition

*This repository contains the code referenced in the paper: ["EmoNeXt: an Adapted ConvNeXt for facial Emotion Recognition"](https://ieeexplore.ieee.org/abstract/document/10337732).*

## Abstract
Facial expressions play a crucial role in human communication serving as a powerful and impactful means to express a wide range of emotions. With advancements in artificial intelligence and computer vision, deep neural networks have emerged as effective tools for facial emotion recognition. In this paper, we propose EmoNeXt, a novel deep learning framework for facial expression recognition based on an adapted ConvNeXt architecture network. We integrate a Spatial Transformer Network (STN) to focus on feature-rich regions of the face and Squeeze-and-Excitation blocks to capture channel-wise dependencies. Moreover, we introduce a self-attention regularization term, encouraging the model to generate compact feature vectors. We demonstrate the superiority of our model over existing state-of-the-art deep learning models on the FER2013 dataset regarding emotion classification accuracy.

## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
   
        pip install -r requirements.txt

5. Download the data and run training:

        python scripts/download_dataset.py
        python train.py \
            --dataset-path='FER2013' \
            --batch-size=64 --lr=0.0001 \
            --epochs=300 \
            --amp \
            --in_22k \
            --num-workers=1 \
            --model-size='tiny'

## Comments
Our codebase builds heavily on Facebook's [ConvNeXt](https://github.com/facebookresearch/ConvNeXt). Thanks for open-sourcing!

## Citation
Please use the following bibtex entry:

    @INPROCEEDINGS{10337732,
      author={El Boudouri, Yassine and Bohi, Amine},
      booktitle={2023 IEEE 25th International Workshop on Multimedia Signal Processing (MMSP)}, 
      title={EmoNeXt: an Adapted ConvNeXt for Facial Emotion Recognition}, 
      year={2023},
      volume={},
      number={},
      pages={1-6},
      doi={10.1109/MMSP59012.2023.10337732}}
