import os, sys
from os.path import dirname
sys.path.append(dirname(dirname(os.path.dirname(os.path.abspath("__file__")))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import torch
import hydra
import subprocess
from omegaconf import DictConfig, OmegaConf

from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model, make_reward_model)
from rlprompt.modules import SQLModuleConfig, make_reinforce_module, ReinforceModuleConfig
from rlprompt.trainers import TrainerConfig, make_reward_model_trainers, RewardTrainerConfig, make_reward_policy_trainer, RewardPolicyTrainerConfig
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir, print_dict)

from fsc_helpers import (PromptedClassificationRewardConfig,
                         FewShotClassificationDatasetConfig,
                         make_prompted_classification_reward,
                         make_few_shot_classification_dataset,
                         get_prompts_for_eval)


# Compose default config
config_list = [PromptedClassificationRewardConfig,
               FewShotClassificationDatasetConfig, LMAdaptorModelConfig,
               SinglePromptModelConfig, SQLModuleConfig, TrainerConfig,
               RewardTrainerConfig, RewardPolicyTrainerConfig, ReinforceModuleConfig]
cs = compose_hydra_config_store('base_fsc', config_list)


@hydra.main(version_base=None, config_path="./", config_name="fsc_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    assert config.output_folder != ""
    output_dir = config.output_folder

    (train_dataset, val_dataset, test_dataset,
     num_classes, verbalizers, template) = \
        make_few_shot_classification_dataset(config)

    print('Train Size:', len(train_dataset), flush=True)
    print('Examples:', flush=True)
    print_dict(train_dataset[:2])
    print('Val Size', len(val_dataset), flush=True)
    print('Examples:', flush=True)
    print_dict(val_dataset[:2])

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    reward = make_prompted_classification_reward(num_classes, verbalizers, 
                                                 template, config)

    reward_model = make_reward_model(config)
    algo_module = make_reinforce_module(prompt_model, reward, reward_model, config)

    # Hack for few-shot classification - Each batch contains all examples
    config.train_batch_size = len(train_dataset)
    config.eval_batch_size = len(val_dataset)
    config.save_dir = os.path.join(output_dir, config.save_dir)
    print("\nSave directory: ", config.save_dir, flush=True)

    reward_model_trainer = make_reward_model_trainers(reward_model, algo_module, train_dataset, val_dataset, config)

    trainer = make_reward_policy_trainer(algo_module, reward_model_trainer, train_dataset, val_dataset, config)

    trainer.train(config=config)

    ##############################
    # START AUTO EVALUATION
    ##############################
    torch.cuda.empty_cache()
    prompts_scores = get_prompts_for_eval(output_dir)

    # evaluate each candidate prompts
    eval_acc = []
    for prompt in prompts_scores["prompts"]:
        cmd = f"cd evaluation; python run_eval.py dataset='{config.dataset}' prompt='{prompt}'; cd ../"
        try:
            output = subprocess.check_output(cmd, shell=True)
            eval_acc.append(float(str(output).split("accuracy: ")[1].split("\\")[0]))
        except Exception:
            eval_acc.append(-1.)
            print(f"\nException on {prompt} !!!\n")

    # add the eval_acc list to the prompts_scores dataframe
    prompts_scores["eval_acc"] = eval_acc
    prompts_scores = prompts_scores[prompts_scores['eval_acc'] >= 0.]
    # sort the resulting prompts_scores dataframe by scores
    prompts_scores = prompts_scores.sort_values(by="scores", ascending=False)
    print(f"\nprompts_scores (shape={prompts_scores.shape}) with evaluation accuracy:")
    print(prompts_scores.head())
    # store the prompts_scores dataframe
    save_loc = os.path.join(output_dir, "prompts_scores.csv")
    prompts_scores.to_csv(save_loc, index=False)
    print(f"\nMean of Top 3 accuracy: {prompts_scores['eval_acc'].values[:3].mean() * 100.:.2f}\n")
    ##############################
    # END AUTO EVALUATION
    ##############################


if __name__ == "__main__":
    main()
