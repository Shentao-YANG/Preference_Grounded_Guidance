from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Optional

from rlprompt.modules import BaseModule
from rlprompt.models import BaseModel
from rlprompt.trainers import RewardModelTrainer, RewardPolicyTrainer


def make_reward_model_trainers(reward_model: BaseModel,
                               policy_module: BaseModule,
                               train_dataset: Optional[Dataset],
                               eval_dataset: Optional[Dataset],
                               config: "DictConfig") -> RewardModelTrainer:
    return RewardModelTrainer(reward_model, policy_module, train_dataset, config.train_batch_size, config.train_shuffle,
                              config.train_drop_last, eval_dataset, config.eval_batch_size, config.save_dir, config.gradient_clip,
                              config.gradient_clip_norm, config.checkpoint_path,
                              # reward-learning specific params
                              config.rew_num_train_epochs, config.rew_max_train_steps, config.rew_learning_rate,
                              config.reward_learning_samples, config.reward_learning_batch_size, config.rew_gradient_accumulation_steps,
                              config.lr_decay, config.weight_decay_count, config.early_stop_count, config.agg_func, config.soft_maxmin_temp, config.compute_rew_batch_mode)


def make_reward_policy_trainer(module: BaseModule,
                               reward_trainer: RewardModelTrainer,
                               train_dataset: Optional[Dataset],
                               eval_dataset: Optional[Dataset],
                               config: "DictConfig") -> RewardPolicyTrainer:
    return RewardPolicyTrainer(module, reward_trainer, train_dataset, config.train_batch_size, config.train_shuffle,
                               config.train_drop_last, config.num_train_epochs, config.max_train_steps, config.do_eval,
                               eval_dataset, config.eval_batch_size, config.eval_steps, config.do_save, config.save_dir,
                               config.save_steps, config.learning_rate, config.gradient_clip, config.gradient_clip_norm,
                               config.checkpoint_path, config.random_seed, config.report_to_wandb, config.project_name, config.run_name,
                               # Reward-Policy joint-training specific params
                               config.reward_retrain_period, config.early_stop_eval_period, config.num_validation_prompts,
                               config.lr_decay, config.weight_decay_count, config.early_stop_count)


@dataclass
class TrainerConfig:
    # Train params
    train_batch_size: int = 16
    train_shuffle: bool = True
    train_drop_last: bool = True
    num_train_epochs: int = 1
    max_train_steps: int = -1
    # Eval params
    do_eval: bool = True
    eval_batch_size: int = 16
    eval_steps: int = -1
    # Save params
    do_save: bool = True
    save_dir: str = './outputs'
    save_steps: int = -1
    # Optimizer params
    learning_rate: float = 1e-4
    gradient_clip: bool = True
    gradient_clip_norm: float = 5.0
    # Checkpoint params
    checkpoint_path: Optional[str] = None
    # Random seed
    random_seed: Optional[int] = None
    # exp idx for logging and saving
    output_folder: str = ""
    # Wandb reporting
    report_to_wandb: bool = False   # True
    project_name: Optional[str] = 'rl-prompt'
    run_name: Optional[str] = None
    # early-stopping callbacks for both reward training and policy learning
    lr_decay: float = 0.8
    weight_decay_count: int = 2
    early_stop_count: int = 7


@dataclass
class RewardTrainerConfig:
    # reward model training params
    reward_learning_samples: int = 3
    reward_learning_batch_size: int = 8
    rew_gradient_accumulation_steps: int = 8
    rew_num_train_epochs: int = 100
    rew_max_train_steps: int = 10000
    rew_learning_rate: float = 5e-5
    agg_func: str = "sum"
    soft_maxmin_temp: float = 1.
    compute_rew_batch_mode: int = 1


@dataclass
class RewardPolicyTrainerConfig:
    # reward model training params
    reward_retrain_period: int = 1000
    early_stop_eval_period: int = 200
    num_validation_prompts: int = 25
