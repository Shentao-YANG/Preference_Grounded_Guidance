# Preference-grounded Token-level Guidance for Language Model Training [NeurIPS '23]

Source codes for the main experiments in *Preference-grounded Token-level Guidance for Language Model Fine-tuning*.
[[Paper]](https://arxiv.org/abs/2306.00398), [[Poster]](https://github.com/Shentao-YANG/Preference_Grounded_Guidance/blob/main/NeurIPS_2023_tokel_level_reward_poster_40x60.pdf).

Bibtex:
```angular2html
@inproceedings{yang2023preferencegrounded,
    title={Preference-grounded Token-level Guidance for Language Model Fine-tuning},
    author={Shentao Yang and Shujian Zhang and Congying Xia and Yihao Feng and Caiming Xiong and Mingyuan Zhou},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://arxiv.org/pdf/2306.00398}
}
```
## Update (01/07/2025)
In our follow-up work, we developed new techniques to successfully scale up the token-level RLHF framework in this paper/repo to PPO + LLMs.
We again observed strong gain over the classical bandit RLHF.

- $\small \textsf{The judge model is GPT-4o}$

| Models                               |  AlpacaEval 2 (LC)  |  Arena-Hard   |  MT-Bench   |
|--------------------------------------|:----------:|:-----:|:-----:|
| ***Token PPO*** Llama-3.1 Instruct 8B |   45.81    | 49.3  | 7.93  |
| Bandit PPO Llama-3.1 Instruct 8B     |   40.77    | 36.6  | 7.76  |
| ***Token PPO*** Llama-3 SFT 8B         |   23.84    | 26.0  | 7.13  |
| Bandit PPO Llama-3 SFT 8B            |   21.20    | 18.7  | 7.11  |

Our follow-up work adopts the framework in this paper and addresses the over-granular issue of token-level RLHF.
Our proposal is to ${\color{lightgreen} \textsf{assign reward to each semantically complete text segment}}$, rather than per-token.
Details are available on:

- ArXiv: [Segmenting Text and Learning Their Rewards for Improved RLHF in Language Model](https://arxiv.org/abs/2501.02790)

- GitHub Repo and Model Releases: [DenseRewardRLHF-PPO](https://github.com/yinyueqin/DenseRewardRLHF-PPO)

## Dependency

To install the required packages, please run the following command:
```angular2html
bash install_packages.sh
```

## Experiments

### Prompt Experiments

As an example, to run the prompt experiments on the `sst-2` dataset under `dataset_seed=0` and `random_seed=0`, 
please use the following commands
```angular2html
cd prompt_task/examples/few-shot-classification
python run_fsc.py 
```
- The above commands is a minimal example. Please change `dataset_seed` and `random_seed` according to your experiment setting.

- Please check `fsc_config.yaml` for available flags.

- For experiments on other datasets, *e.g.*, `agnews` and `yelp-2`, please change the corresponding flags in `fsc_config.yaml` accordingly. 

#### Prompt Examples

Examples of good generated text-prompt and their classification accuracy on the corresponding test set are as follows.
If you want to directly use them, please pay attention to the spacing. You may directly copying from the source code of `README.md`.

|               **SST-2**               |    **SST-2**   |                    **Yelp P.**                   |   **Yelp P.**  |              **AG News**              |   **AG News**  |
|-------------------------------------|:--------------:|------------------------------------------------|:--------------:|-------------------------------------|:--------------:|
|              **_Prompt_**             | **_Accuracy_** |                   **_Prompt_**                   | **_Accuracy_** |              **_Prompt_**             | **_Accuracy_** |
| guys filmmaker filmmaker rated Grade  |      94.18     | done Absolutely Absolutely absolutecompletely    |      96.14     | newsIntroduction Comments Tags Search |      85.78     |
| MovieMovieFilm rated Grade            |      94.18     | passionately Absolutely utterly absolutely to... |      95.25     | newsTopic Blog Support Category       |      85.55     |
| Rated CinemaScoreReporting Grade      |      94.01     | distinctly absolutely utterly Absolutely utterly |      95.15     | news RecentRecentPhotosIntroduction   |      84.53     |
| employment theater rated Oscars Grade |      93.96     | loosely Absolutely absolutely utterly totally    |      95.14     | news Recent Brief LatestExample       |      84.51     |
| scene filmmaking rated comedian Grade |      93.85     | markedly Absolutely utterly utterly utterly      |      95.10     | newsVirtualBlogBlogNet                |      84.33     |

### Text Summarization Experiments
As an example, to run the summarization experiments under random seed `0`, please use the following commands
```angular2html
cd sum_task/

# for the "cnn_dailymail" dataset 
python run_sum.py --dataset_name="cnn_dailymail" --dataset_config="3.0.0" --seed=0

# for the "xsum" dataset
python run_sum.py --dataset_name="xsum" --seed=0
```
- The above commands is a minimal example. Please change `--seed` according to your experiment setting.
- Please check `parse_args.py` for available flags.

Note: `--model_name_or_path` and `--rew_model_name_or_path` need not be the same. In particular, one may use a smaller 
pretrained LM for the reward model to save compute. 

## Acknowledgement

This codebase builds on the following codebases:
* [**RLPrompt**](https://github.com/mingkaid/rl-prompt)
* [**RL4LMs**](https://github.com/allenai/RL4LMs)
* [**Hugging Face**](https://github.com/huggingface/transformers/tree/main/)

## License
MIT License.


















