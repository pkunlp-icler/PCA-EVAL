<h1 align="center">PCA-Bench</h1>

<p align="center">

<a href="https://pca-eval.github.io/">
<img alt="Static Badge" src="https://img.shields.io/badge/Documentation-Online-green">
</a>

<a href="https://arxiv.org/abs/2310.02071">
<img alt="Static Badge" src="https://img.shields.io/badge/Paper-PCAEVAL-red">

<a href="https://huggingface.co/datasets/PCA-Bench/PCA-Bench-V1">
<img alt="Static Badge" src="https://img.shields.io/badge/Datasets-HuggingFace-yellow">
</a>
</p>




*PCA-Bench is an innovative benchmark for evaluating and locating errors in Multimodal LLMs when conducting embodied decision making tasks, specifically focusing on perception, cognition, and action.*

<div align=center>
<img width="300" src="./imgs/pca-chain.png"/>
</div>


## News
- PCA-Bench-V1 is released in HuggingFace Datasets.

## Leaderboard

[Leaderboard with Full Metrics](【腾讯文档】PCA-Bench-V1-Leaderboard PCA-Eval Results
https://docs.qq.com/sheet/DVUd4WUpGRHRqUnNV?tab=mqeh4c)


## Run Evaluation on Accuracy

```python
#pip install datasets
from datasets import load_dataset
dataset_ad = load_dataset("PCA-Bench/PCA-Bench-V1","Autonomous Driving")
dataset_dr = load_dataset("PCA-Bench/PCA-Bench-V1","Domestic Robot")
dataset_og = load_dataset("PCA-Bench/PCA-Bench-V1","Open-World Game")

# use your model to inference on the test_open/close split with "question_prompt" and "image" given in the datasets and extract the answers.
# compute the acc regarding the action groundtruth of open track. 

```

📢 For close track data, please follow [this file](https://github.com/pkunlp-icler/PCA-EVAL/blob/main/pca-eval/results/chatgpt_holmes_outputs/Autonomous%20Driving.json) to organize your model output. Submit **three JSON files** from different domains, along with your **model name** and **organization** to us via [email](mailto:leo.liang.chen@stu.pku.edu.cn). Ensure you use the dataset's provided prompt as the default input for fair comparison (You should mention in the email if custom prompts were used).

We will send the closed track PCA-Eval results of your model to you.




## Run PCA Evaluation


```bash
git clone https://github.com/pkunlp-icler/PCA-EVAL.git
cd PCA-EVAL
```




### End2End Method

In the End2End method, the prompt utilized for each instance, along with its corresponding image name, is provided in JSON format within the data directory specific to each domain. For example:

```bash
pca-eval/data/v1.0/Autonomous Driving/end2end_prompts.json
pca-eval/data/v1.0/Domestic Robot/end2end_prompts.json
pca-eval/data/v1.0/Open-World Game/end2end_prompts.json
```

You can seamlessly pass both the image and the prompt to your multimodal model to obtain results.

The output for each instance should be saved in json file, in the format of
```json
[
    {"index":0,"model_output":"xxxxx"},
    {"index":1,"model_output":"xxxxx"}, 
]
```


### HOLMES Method

For HOLMES method using LLM, we provide jupyter notebooks for OPENAI model tested in our paper. By changing the openai key and data path, you could reproduce the results easily.

```bash
pca-eval/evaluation/HOLMES_Autonomous_Driving.ipynb
pca-eval/evaluation/HOLMES_Domestic_Robot.ipynb
pca-eval/evaluation/HOLMES_Game.ipynb
```


The output for each instance should be saved in json file, in the format of
```json
[
    {"index":0,"model_output":"xxxxx"},
    {"index":1,"model_output":"xxxxx"},
]
```

### Automatic Scoring


```bash
python ./pca-eval/evaluation/pca_auto_scoring.py \ 
    --meta_data  pca-eval/data/v1.0/Open-World Game/meta_data.json \  # path to the meta data
    --model_output chatgpt_output.json \  # model output file in json format
    --openai_key sk-xxxxxxxxxx \  # your openai key
    --output_path  chatgpt_result.json \  # path to save the result
```


**Evaluation Rule: To make fair evaluation and comparison among different models, make sure you use the same LLM evaluation model as ours (GPT4) for all the models you want to evaluate.**



## Benchmark Overview

<div align=center>
<img width="400" src="./imgs/sun.jpg"/>
      
Domain and required ability distribution of PCA-EVAL.
</div>

### Examples

- Traffic Domain

<div align=center>
<img width="600" src="./imgs/traffic_example.png"/>
</div>


- Domestic Robot Domain

<div align=center>
<img width="600" src="./imgs/alfred_example.png"/>
</div>


- Game Domain

<div align=center>
<img width="600" src="./imgs/mc_example.png"/>
</div>

## Citation
```bib
@article{chen2023endtoend,
      title={Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond}, 
      author={Liang Chen and Yichi Zhang and Shuhuai Ren and Haozhe Zhao and Zefan Cai and Yuchi Wang and Peiyi Wang and Tianyu Liu and Baobao Chang},
      year={2023},
      journal={ArXiv},
}
```



