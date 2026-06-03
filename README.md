# REFACT: Fact Restatement for Compact and Faithful Chain-of-Thought Reasoning

# Overview
![](figs/main.png)

Click the links below to view our papers, checkpoints:

<a href='https://arxiv.org/abs/2506.10822'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a><a href='https://huggingface.co/jinpu666/ReCUT-Qwen'><a href='https://huggingface.co/jinpu666/ReCUT-Llama'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a>

# Set up
**Use `git clone` to download this project**
```
git clone https://github.com/NEUIR/REFACT.git
cd REFACT
```
**To prevent conflicts between packages, we mainly use two virtual environment management packages, one for constructive data and evaluate、 one for model rl training.**

```
for constructive data and evaluate, please:
conda create -n eval python==3.10.0
conda activate eval
pip install -r requirements.txt

for model training, please:
conda create -n verl python==3.10.0
conda activate verl
pip install -r requirements_rl.txt

```

# Data
Our corresponding generated training data is placed under the data folder

Download the files from [here]()
Use the downloaded data to synthesize the data using the following scripts
```

```


# GRPO
Our GRPO training uses verl. Before use, please modify the model path and dataset path in the script below, as well as the output path for saving checkpoints. You can use our data[here]() for training.
```
conda activate verl
cd verl
bash ./example/grpo_trainer/longtext/run_8b_ruler_cite.sh
```

# Evaluate
Our evaluation uses LongBench's and LVEval's evaluation methodology and we provide nothing but good test datasets. If you want to make any changes, please refer to the files under config.
```
cd LongBench
python pred.py --model mode_name
python eval.py --model mode_name
```

Before evaluation, please first go to the [LVEval](https://github.com/infinigence/LVEval) official website to download the corresponding dataset.
```
cd LVEval
bash pred_vllm.sh model_path output_dir
bash eval.sh output_dir  
```





































