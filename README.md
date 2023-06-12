# AcerNet: an Adapted ConvNeXt for facial Emotion Recognition

## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the data and run training:
```bash
python train.py \
    --dataset-path='FER2013' \
    --batch-size=64 --lr=0.0001 \
    --epochs=300 \
    --amp \
    --in_22k \
    --num-workers=1 \
    --model-size='tiny'