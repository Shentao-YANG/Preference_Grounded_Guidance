# Preference-grounded Token-level Guidance for Language Model Fine-tuning

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

### Summarization Experiments
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

## Acknowledgement

This codebase builds on the following codebases:
* [**RLPrompt**](https://github.com/mingkaid/rl-prompt)
* [**Hugging Face**](https://github.com/huggingface/transformers/tree/main/)




















