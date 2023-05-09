import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List
import os
import wandb
import json
from transformers import WEIGHTS_NAME
import time
from rlprompt.modules import BaseModule
from rlprompt.utils import utils
from .trainer_utils import set_random_seed


class RewardPolicyTrainer:
    def __init__(
        self,
        module: BaseModule,
        reward_trainer,
        # Train params
        train_dataset: Optional[Dataset],
        train_batch_size: int,
        train_shuffle: bool,
        train_drop_last: bool,
        num_train_epochs: int,
        max_train_steps: int,
        # Eval params
        do_eval: bool,
        eval_dataset: Optional[Dataset],
        eval_batch_size: int,
        eval_steps: int,
        # Save params
        do_save: bool,
        save_dir: str,
        save_steps: int,
        # Optimizer params
        learning_rate: float,
        gradient_clip: bool,
        gradient_clip_norm: float,
        # Checkpoint params
        checkpoint_path: Optional[str],
        # Random seed
        random_seed: Optional[int],
        # Wandb reporting
        report_to_wandb: bool,
        project_name: Optional[str],
        run_name: Optional[str],
        # RewardPolicyTrainer specific param
        reward_retrain_period: int,
        early_stop_eval_period: int,
        num_validation_prompts: int,
        lr_decay: float,
        weight_decay_count: int,
        early_stop_count: int
    ):
        assert not do_eval or eval_dataset is not None, "Need to have eval_dataset if do_eval is True"
        self.module = module
        self.reward_trainer = reward_trainer
        self.device = self.reward_trainer.device

        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.train_drop_last = train_drop_last
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps

        self.do_eval = do_eval
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps

        self.do_save = do_save
        self.save_dir = save_dir
        self.save_steps = save_steps

        self.reward_retrain_period = reward_retrain_period
        self.early_stop_eval_period = early_stop_eval_period
        self.num_validation_prompts = num_validation_prompts

        self.gradient_clip = gradient_clip
        self.gradient_clip_norm = gradient_clip_norm
        self.lr_decay = lr_decay
        self.weight_decay_count = weight_decay_count
        self.early_stop_count = early_stop_count
        self.report_interval = min(max(self.early_stop_eval_period // 10, 1), 20)

        self.optimizer = torch.optim.Adam(self.module._model.parameters(), lr=learning_rate)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, load_rew_ckpt=True)

        if random_seed is not None:
            set_random_seed(random_seed)

        self.report_to_wandb = report_to_wandb
        self.project_name = project_name
        self.run_name = run_name

        self.print_prefix = "[PolicyLearning]"

        print(
            f"\n{self.print_prefix} train_batch_size: {self.train_batch_size}; eval_batch_size: {self.eval_batch_size}; num_train_epochs: {self.num_train_epochs}; "
            f"\nmax_train_steps: {self.max_train_steps}; reward_retrain_period: {self.reward_retrain_period}; early_stop_eval_period: {self.early_stop_eval_period}; "
            f"num_validation_prompts: {num_validation_prompts} \n",
            flush=True)

        self.start_time = time.time()

    def _load_checkpoint(self, checkpoint_path: str, load_rew_ckpt: bool) -> None:
        # checkpoint_path: path to the **policy** checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.module.load_state_dict(checkpoint["model_state_dict"])

        reward_model_path = os.path.join(checkpoint_path.split("ckpt")[0], "model_ckpt", WEIGHTS_NAME)
        if load_rew_ckpt:
            self.reward_trainer._load_checkpoint(checkpoint_path=reward_model_path)

        print_info = f"{self.print_prefix} Loaded module from {checkpoint_path}; " \
                     f"Loaded reward trainer from {reward_model_path if load_rew_ckpt else 'NO LOADING'} !!!"
        print("^" * len(print_info), flush=True)
        print(print_info, flush=True)
        print("^" * len(print_info), flush=True)

    def train_op(self):
        if self.gradient_clip:
            nn.utils.clip_grad_norm_(self.module._model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          shuffle=self.train_shuffle,
                          batch_size=self.train_batch_size,
                          drop_last=self.train_drop_last)

    def _get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
        return DataLoader(eval_dataset, batch_size=self.eval_batch_size)

    def _train_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        # one gradient-update step
        model = self.module.train()

        loss, batch_log = model(batch)
        loss.backward()
        self.train_op()

        return batch_log

    def _print_batch_log(self, batch_log, total_steps):
        print(f"{self.print_prefix} [{total_steps}/{self.max_train_steps}] loss: {batch_log['model/rew_loss']:.4f}, "
              f"model-rews-each-step: {batch_log['model/rews']}, trainset rewards: {batch_log['model/rewards/raw']:.2f}, "
              f"acc: {batch_log['model/rewards/acc'] * 100.:.2f}, time: {(time.time() - self.start_time) / 60:.1f} min !!!",
              flush=True)

    def train(self,
              report_to_wandb: Optional[bool] = None,
              project_name: Optional[str] = None,
              run_name: Optional[str] = None,
              config: Optional["DictConfig"] = None) -> None:
        # Configure Wandb reporting
        if report_to_wandb is None:
            report_to_wandb = self.report_to_wandb
        if project_name is None:
            project_name = self.project_name
        if run_name is None: 
            run_name = self.run_name
        if config is not None: 
            config = eval(str(config))
        if report_to_wandb:
            wandb.init(project=project_name, name=run_name, config=config)
            wandb.watch(self.module, log=None)

        # Create saving path
        eval_save_dir = os.path.join(self.save_dir, "eval")
        ckpt_save_dir = os.path.join(self.save_dir, "ckpt")
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)
        validation_save_loc = os.path.join(ckpt_save_dir, WEIGHTS_NAME)

        train_dataloader = self._get_train_dataloader()
        if self.max_train_steps < 0:
            total_train_epochs = self.num_train_epochs
        else:
            num_batches_per_epoch = len(train_dataloader)
            total_train_epochs = \
                (self.max_train_steps // num_batches_per_epoch
                 + int(self.max_train_steps % num_batches_per_epoch > 0))

        # Determine whether to evaluate by epoch or steps
        eval_by_steps = self.eval_steps > 0
        # Determine whether to save by epoch or steps
        save_by_steps = self.save_steps > 0

        print("*" * 80)
        print(f"{self.print_prefix} START Policy-Reward Joint Learning! "
              f"\neval_save_dir={eval_save_dir}, ckpt_save_dir={ckpt_save_dir}, "
              f"\nlen(train_dataloader)={len(train_dataloader)}, total_train_epochs={total_train_epochs} !!!")

        prev_max_score = -1e10
        early_stop_count = self.early_stop_count

        # lr scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.lr_decay,
            patience=self.weight_decay_count,
            min_lr=0.000001,
            verbose=True
        )

        # train the reward model before starting the policy learning
        self.reward_trainer.train()
        # start policy learning
        total_steps = 0
        for epoch in range(total_train_epochs):
            for step, batch in enumerate(train_dataloader):
                batch_log = self._train_step(batch=batch)
                total_steps += 1
                if report_to_wandb:
                    wandb.log(batch_log)
                else:
                    if total_steps % self.report_interval == 0:
                        self._print_batch_log(batch_log=batch_log, total_steps=total_steps)

                if self.do_eval and eval_by_steps \
                    and ((total_steps >= (self.max_train_steps - 2 * self.early_stop_eval_period)) or (early_stop_count <= 2)) \
                        and total_steps % self.eval_steps == 0:
                    output_save_path = os.path.join(eval_save_dir, f'outputs.step.{total_steps}.json')
                    eval_log = self.evaluate(output_save_path=output_save_path)
                    if report_to_wandb:
                        wandb.log(eval_log)
                    else:
                        print_info = f"{self.print_prefix} [Eval: {total_steps}/{self.max_train_steps}], acc: {eval_log['eval/rewards/acc'] * 100.:.2f}; " \
                                     f"score: {eval_log['eval/score']:.2f}; " \
                                     f"out_len: {eval_log['eval/output_length']:.0f}, time: {(time.time() - self.start_time) / 60:.1f} min !!!"
                        print(print_info, flush=True)
                        print("^" * len(print_info), flush=True)

                if self.do_save and save_by_steps and (total_steps % self.save_steps == 0):
                    torch.save({"steps": total_steps, "model_state_dict": self.module.state_dict()},
                               os.path.join(ckpt_save_dir, f"ckpt.step.{total_steps}.pth"))

            if total_steps > self.max_train_steps:
                break

            # 1 <= total_steps <= self.max_train_steps
            # early stopping
            if total_steps % self.early_stop_eval_period == 0:
                # don't save the prompts generated at the validation process
                if total_steps == self.max_train_steps:
                    self.module._reward._counter = self.max_train_steps - 1
                # this is effectively the end of one epoch
                valid_score = self.evaluate(output_save_path=None, num_prompt_samples=self.num_validation_prompts)["eval/score"]
                scheduler.step(valid_score)
                print_info = f"{self.print_prefix} [{total_steps}/{self.max_train_steps}] valid score: {valid_score:.4f}, " \
                             f"prev_max_score: {prev_max_score:.4f}, time: {(time.time() - self.start_time) / 60:.1f} min"
                print("-" * len(print_info), flush=True)
                print(print_info, flush=True)
                if valid_score > prev_max_score * (1 + 1e-4):
                    early_stop_count = self.early_stop_count
                    prev_max_score = valid_score
                    torch.save({"steps": total_steps, "model_state_dict": self.module.state_dict()}, validation_save_loc)
                    print(f'{self.print_prefix} [{total_steps}/{self.max_train_steps}] Model saved to {validation_save_loc} !!!', flush=True)
                else:
                    early_stop_count -= 1
                    print(f'{self.print_prefix} [{total_steps}/{self.max_train_steps}] early stop countdown {early_stop_count}/{self.early_stop_count} !!!', flush=True)

                print("-" * len(print_info), flush=True)
                if early_stop_count == 0:
                    break

            # retrain the reward model
            # only retrain the reward model during the first half of policy-learning
            if (total_steps <= (self.max_train_steps // 2)) and (total_steps % self.reward_retrain_period == 0):
                print_info = f"{self.print_prefix} [{total_steps}/{self.max_train_steps}] Retrain Reward Model ({total_steps // self.reward_retrain_period + 1}-th reward model) !!!"
                print("*" * len(print_info), flush=True)
                print(print_info, flush=True)
                print("*" * len(print_info), flush=True)
                self.reward_trainer.train()

            if self.do_eval and not eval_by_steps:
                output_save_path = os.path.join(eval_save_dir, f'outputs.epoch.{epoch+1}.json')
                eval_log = self.evaluate(output_save_path=output_save_path)
                if report_to_wandb:
                    wandb.log(eval_log)
                else:
                    print(eval_log, flush=True)

            if self.do_save and not save_by_steps:
                torch.save({"steps": total_steps, "model_state_dict": self.module.state_dict()},
                           os.path.join(ckpt_save_dir, f"ckpt.epoch.{epoch+1}.pth"))

        # load the best checkpoint during the training process
        # need to generate candidate prompts based on the loaded checkpoint
        self.module._reward._counter = self.max_train_steps
        # load the saved (early stopped) checkpoint
        self._load_checkpoint(validation_save_loc, load_rew_ckpt=False)
        batch_log = self._train_step(batch=batch)
        self._print_batch_log(batch_log=batch_log, total_steps=total_steps)

        print("*" * 80)
        return

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        output_save_path: Optional[str] = None,
        num_prompt_samples: int = 1
    ) -> Dict[str, np.number]:

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_dataloader = self._get_eval_dataloader(eval_dataset)

        model = self.module.eval()
        hypos = []
        scores: List[List[str]] = []
        score_list = []

        for _ in range(num_prompt_samples // len(eval_dataloader)):
            for batch in eval_dataloader:
                if num_prompt_samples < 2:
                    infer_outputs = model.infer(batch)
                else:
                    infer_outputs = {'sample_tokens': model._decode_sampling(batch=batch, return_gs_action=False)[1]}
                hypos += infer_outputs['sample_tokens']

                score, score_log = model.compute_rewards(
                    batch=batch,
                    output_tokens=infer_outputs['sample_tokens'],
                    mode="infer"
                )
                scores += score.detach().tolist()
                score = float(score.detach().mean().item())
                score_list.append(score)

        if (output_save_path is not None) and num_prompt_samples < 2:
            json.dump({'output_tokens': hypos, 'scores': scores}, open(output_save_path, 'w'))

        utils.add_prefix_to_dict_keys_inplace(score_log, prefix=f"eval/rewards/")

        if num_prompt_samples > 1:
            assert len(score_list) == num_prompt_samples
            score = float(np.mean(score_list))

        self.module.train()
        return utils.unionize_dicts([
            score_log,
            {
                f"eval/score": score,
                f"eval/output_length": float(np.mean([len(tokens) for tokens in hypos]))
            }
        ])
