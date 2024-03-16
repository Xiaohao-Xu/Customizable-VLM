<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">Customizing Visual-Language Foundation Models for Anomaly Detection and Reasoning</h1>
    <p align="center">
    <strong>UMich</strong>
    ·
    <strong>HUST</strong>
  </p>
  <h3 align="center"><a href="">Preprint Paper</a></h3>
  <div align="center"></div>
</p>

<br>

## Setup environment:

Install packages for your virtual environment:
```
pip install -r requirements.txt
```
Set up API keys for openai and google on terminal or .bashrc:
```
export OPENAI_API_KEY=<your key>
export GOOGLE_API_KEY=<your key>
```

## Dataset Preparation:

### MVTecAD

Download link: [MVTecAD-Website](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## Evaluation Scripts:

Eval on GeMini
```
python main_gemini.py --dataset "datasets/MVTecAD/vlm_for_ad_dataset.json" --cache "./output/answer_genmini.json" --output "./output/answer_5.json" --google_api_key 'ADD_YOUR_GOOLE_API_HERE’ --prompt_template “./prompt_template/ad_prompt.txt”
```

Eval on GPT4-Vision
```
python main_gpt.py --dataset "datasets/MVTecAD/vlm_for_ad_dataset.json" --cache "./output/answer_gpt4v.json" --output "./output/answer_gpt4v.json" --openai_api_key ‘ADD_YOUR_OPENAI_API_HERE’ --prompt_template “./prompt_template/ad_prompt.txt”
```

## Citation

Please cite our paper if you find this repo useful! :yellow_heart: :blue_heart: :yellow_heart: :blue_heart:

```bibtex
@article{xu2024custimizing,
  title={Customizing Visual-Language Foundation Models for Anomaly Detection and Reasoning},
  author={Xu, Xiaohao and Cao, Yunkang and Chen, Yongqi and Shen, Weiming and and Huang, Xiaonan},
  journal={arXiv preprint},
  year={2024}
}
```

## Contact
If you have any question about this project, please feel free to contact xiaohaox@umich.edu
