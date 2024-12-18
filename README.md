# Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning

![image](https://github.com/Xiaohao-Xu/Customizable-VLM/assets/147964359/3bc6c6ab-b5c2-4b2d-8530-577ea95b9a0f)

[ArXiv-Paper](https://arxiv.org/pdf/2403.11083.pdf)


## Setup environment:

Install packages for your virtual environment:
```
pip install -r requirements.txt
```
Set up API keys for openai  (Non-Free GPT-4Vision API Usage) and google (for Free Gemini Vision API Usage) on terminal or .bashrc:
```
export OPENAI_API_KEY=<your key>
export GOOGLE_API_KEY=<your key>
```

## Dataset Preparation:

### MVTecAD
- Download link: [MVTecAD-Website](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- Please put the dataset under the folder [datasets/MVTecAD](./datasets/MVTecAD)
## Evaluation Scripts:

Eval on Gemini
```
python main_gemini.py --dataset "datasets/MVTecAD/vlm_for_ad_dataset.json" --cache "./output/answer_genmini.json" --output "./output/answer_5.json" --google_api_key 'ADD_YOUR_GOOLE_API_HERE’ --prompt_template “./prompt_template/ad_prompt.txt”
```

Eval on GPT4-Vision
```
python main_gpt.py --dataset "datasets/MVTecAD/vlm_for_ad_dataset.json" --cache "./output/answer_gpt4v.json" --output "./output/answer_gpt4v.json" --openai_api_key ‘ADD_YOUR_OPENAI_API_HERE’ --prompt_template “./prompt_template/ad_prompt.txt”
```

Eval on InternVL2
- Follow the [official guidance](https://internvl.readthedocs.io/en/latest/get_started/installation.html) to set up the environment for `InternVL2` and download the checkpoints. (By default, we used `InternVL2-8B`.)
```
python main_internvl2.py --model "~/path/to/InternVL2-8B" --dataset "datasets/MVTecAD/vlm_for_ad_dataset.json" --cache "./output/answer_internvl2_8b.json" --output "./output/answer_internvl2_8b.json"
```

Eval on Qwen2VL
- Follow the [official repo](https://github.com/QwenLM/Qwen2-VL) to set up the environment for `Qwen2VL` and download the checkpoints. (By default, we used `Qwen2-VL-7B-Instruct`.)
```
python main_qwenvl2.py --model "~/path/to/Qwen2-VL-7B-Instruct" --dataset "datasets/MVTecAD/vlm_for_ad_dataset.json" --cache "./output/answer_qwenvl2_7b.json" --output "./output/answer_qwenvl2_7b.json"
```

## Citation

Please cite our paper if you find this repo useful! :yellow_heart: :blue_heart: :yellow_heart: :blue_heart:

```bibtex
@article{xu2024custimizing,
  title={Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning},
  author={Xu, Xiaohao and Cao, Yunkang and Chen, Yongqi and Shen, Weiming and Huang, Xiaonan},
  journal={arXiv preprint arXiv:2403.11083},
  year={2024}
}
```

## Contact
If you have any question about this project, please feel free to contact xiaohaox@umich.edu
