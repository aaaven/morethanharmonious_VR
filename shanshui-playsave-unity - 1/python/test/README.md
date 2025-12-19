

## Stable Install
```shell
conda create -n linz pytorch torchvision torchaudio pytorch-cuda diffusers ipykernel transformers opencv huggingface_hub accelerate -c pytorch -c nvidia -c conda-forge
```

## Nightly Install
```shell
conda create -n linz pytorch torchvision torchaudio pytorch-cuda diffusers ipykernel transformers opencv accelerate pytest expecttest fire -c pytorch-nightly -c nvidia -c conda-forge
conda activate linz
conda install tokenizers -c conda-forge
conda install transformers -c conda-forge
conda install huggingface_hub -c conda-forge
pip install torchao
```
