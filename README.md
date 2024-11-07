
## 1. Requirements

```bash
pip install -r requirements.txt
```

## 2. Data Preparation

```bash
python dataset/generate_training_data.py
```

## 3. Training the ISNet Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 main.py
```
