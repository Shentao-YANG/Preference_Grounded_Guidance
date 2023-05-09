import torch
from torch import optim, nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, Union, List
from transformers import WEIGHTS_NAME
from functools import partial
import os
import time

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.losses import listMLELoss


def logsumexp(x: torch.tensor, temp: float):
    avg_trajectory_len = x.shape[1]
    return temp * torch.logsumexp(x / temp, dim=-1) * avg_trajectory_len


def tensor_sum(x: torch.tensor):
    return torch.sum(x, dim=-1)


def tensor_average(x: torch.tensor):
    avg_trajectory_len = x.shape[1]
    return torch.mean(x, dim=-1) * avg_trajectory_len


class RewardModelTrainer:
    def __init__(
        self,
        reward_model: BaseModel,
        policy_module: BaseModule,
        # Train params
        train_dataset: Optional[Dataset],
        train_batch_size: int,
        train_shuffle: bool,
        train_drop_last: bool,
        # Eval params
        eval_dataset: Optional[Dataset],
        eval_batch_size: int,
        # Save params
        save_dir: str,
        # Optimizer params
        gradient_clip: bool,
        gradient_clip_norm: float,
        # Checkpoint params
        checkpoint_path: Optional[str],
        # reward model training params
        rew_num_train_epochs: int,
        rew_max_train_steps: int,
        rew_learning_rate: float,
        reward_learning_samples,
        reward_learning_batch_size,
        rew_gradient_accumulation_steps,
        lr_decay,
        weight_decay_count,
        early_stop_count,
        agg_func: str,
        soft_maxmin_temp: float,
        compute_rew_batch_mode: int
    ):
        assert eval_dataset is not None, \
            "Need to have a eval_dataset"
        self.module = reward_model
        self.policy_module = policy_module
        self.tokenizer = self.policy_module._model._model.tokenizer
        self.reward_learning_samples = reward_learning_samples
        self.true_reward_func = self.policy_module._reward
        self.device = self.module.transformer.device
        self.reward_learning_batch_size = reward_learning_batch_size
        self.gradient_accumulation_steps = rew_gradient_accumulation_steps
        self.rew_learning_rate = rew_learning_rate

        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.train_drop_last = train_drop_last
        self.num_train_epochs = rew_num_train_epochs
        self.max_train_steps = rew_max_train_steps
        self.steps_per_epoch = max(self.max_train_steps // self.num_train_epochs, 1)
        self._num_valid_steps = max(self.steps_per_epoch // 4, 1)       # valid:train = 1:4

        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size

        self.save_dir = save_dir

        self.gradient_clip = gradient_clip
        self.gradient_clip_norm = gradient_clip_norm

        self.lr_decay = lr_decay
        self.weight_decay_count = weight_decay_count
        self.early_stop_count = early_stop_count

        self.optimizer = optim.Adam(self.module.parameters(), lr=rew_learning_rate)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self._num_train_steps = 0
        self.report_interval = max(self.steps_per_epoch // 5, 1)
        self.print_prefix = "[RewardLearning]"

        if agg_func == "max":
            self.agg_func = partial(logsumexp, temp=soft_maxmin_temp)
        elif agg_func == "min":
            self.agg_func = partial(logsumexp, temp=((-1.) * soft_maxmin_temp))
        elif agg_func == "sum":
            self.agg_func = tensor_sum
        elif agg_func == "avg":
            self.agg_func = tensor_average
        else:
            raise NotImplementedError

        assert compute_rew_batch_mode in (0, 1)
        self.compute_rew_batch_mode = compute_rew_batch_mode == 1

        print(f"\n{self.print_prefix} reward_learning_samples: {self.reward_learning_samples}; batch_size: {self.reward_learning_batch_size}; "
              f"gradient_accumulation_steps: {self.gradient_accumulation_steps}; \nnum_train_epochs: {self.num_train_epochs}; "
              f"max_train_steps: {self.max_train_steps}; steps_per_epoch: {self.steps_per_epoch}; num_valid_steps: {self._num_valid_steps}; "
              f"\ntokenizer: {self.tokenizer.__class__}; "
              f"\ntrain_batch_size: {self.train_batch_size}; eval_batch_size: {self.eval_batch_size}, agg_func={agg_func}, "
              f"soft_maxmin_temp={soft_maxmin_temp}, compute_rew_batch_mode={self.compute_rew_batch_mode} \n",
              flush=True)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.module.load_state_dict(checkpoint["model_state_dict"])
        print_info = f"{self.print_prefix} Loaded module from {checkpoint_path} !!!"
        print("^" * len(print_info), flush=True)
        print(print_info, flush=True)
        print("^" * len(print_info), flush=True)

    def train_op(self):
        if self.gradient_clip:
            nn.utils.clip_grad_norm_(self.module.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          shuffle=self.train_shuffle,
                          batch_size=self.train_batch_size,
                          drop_last=self.train_drop_last)

    def _get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
        return DataLoader(eval_dataset, batch_size=self.eval_batch_size)

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self.tokenizer.convert_tokens_to_string(s)
                for s in tokens]

    def get_sampled_prompts(self, batch: Dict[str, Any]):
        with torch.no_grad():
            (output_tokens, _, output_ids, _) = self.policy_module._model.generate(**batch,
                                                               do_sample=True,
                                                               top_k=self.policy_module._top_k,
                                                               top_p=self.policy_module._top_p,
                                                               num_beams=self.policy_module._num_beams).values()
        prompt_strings = self._convert_tokens_to_string(output_tokens)
        return prompt_strings, output_ids

    def get_infer_prompts(self, batch: Dict[str, Any]):
        with torch.no_grad():
            (output_tokens, _, output_ids, _) = self.policy_module.infer(batch).values()
        prompt_strings = self._convert_tokens_to_string(output_tokens)
        return prompt_strings, output_ids

    def compute_rewards(self, batch: Dict[str, Any], prompt_strings: List[str]):
        """Simplify version of `fsc_reward.forward`"""
        source_texts = batch['source_texts']
        class_labels = batch['class_labels']
        batch_size = len(source_texts)
        rewards: List[torch.Tensor] = []
        for i, prompt in enumerate(prompt_strings):
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self.true_reward_func._format_prompts(source_texts, current_prompts)
            all_logits = self.true_reward_func._get_logits(formatted_templates)
            class_probs = torch.softmax(all_logits[:, self.true_reward_func.verbalizer_ids], -1)

            # Get label and maximum not-label probabilities
            label_probs = class_probs[range(batch_size), class_labels]
            not_label_probs = torch.where(
                class_probs == label_probs.unsqueeze(1),
                torch.Tensor([-1]).to(self.device), class_probs)
            max_not_label_probs, _ = torch.max(not_label_probs, -1)
            # Compute piecewise gap reward
            gap = (label_probs - max_not_label_probs)
            correct = (gap > 0).long()
            gap_rewards = gap * (self.true_reward_func.correct_coeff * correct \
                                 + self.true_reward_func.incorrect_coeff * (1 - correct))
            reward = gap_rewards.mean().detach()
            rewards.append(reward)

        rewards_tensor = torch.stack(rewards)

        return rewards_tensor.to(self.device)

    def compute_rewards_multiple_prompts(self, batch: Dict[str, Any], prompt_strings: List[str]):
        """batch_version of `compute_rewards`"""
        num_prompts = len(prompt_strings)
        source_texts = batch['source_texts']
        class_labels = batch['class_labels'].repeat(num_prompts)
        batch_size = num_prompts * len(source_texts)
        all_formatted_templates: List[str] = []
        # all_formatted_templates: [prompt1's formatted strings, prompt2's formatted strings,...]
        for prompt in prompt_strings:
            # Compute LM logits
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self.true_reward_func._format_prompts(source_texts, current_prompts)
            all_formatted_templates.extend(formatted_templates)

        all_logits = self.true_reward_func._get_logits(all_formatted_templates)
        class_probs = torch.softmax(all_logits[:, self.true_reward_func.verbalizer_ids], -1)

        # Get label and maximum not-label probabilities
        label_probs = class_probs[range(batch_size), class_labels]
        not_label_probs = torch.where(
            class_probs == label_probs.unsqueeze(1),
            torch.Tensor([-1]).to(self.device), class_probs)
        max_not_label_probs, _ = torch.max(not_label_probs, -1)
        # Compute piecewise gap reward
        gap = (label_probs - max_not_label_probs)
        correct = (gap > 0).long()
        gap_rewards = gap * (self.true_reward_func.correct_coeff * correct \
                             + self.true_reward_func.incorrect_coeff * (1 - correct))

        rewards_tensor = gap_rewards.reshape(num_prompts, -1).mean(dim=-1)

        return rewards_tensor

    def get_loss(self, batch: Dict[str, Any], mode='train'):

        # get sampled prompts
        if mode == 'train':
            prompt_strings_candidates, output_ids_candidates = self.get_sampled_prompts(batch)
        elif mode == 'test':
            prompt_strings_candidates, output_ids_candidates = self.get_infer_prompts(batch)
        else:
            raise NotImplementedError

        total_num_prompts = len(prompt_strings_candidates)
        assert total_num_prompts > 1

        # get the corresponding true rewards
        train_batch_size = len(batch['source_texts'])
        sample_batch_idx = np.random.choice(train_batch_size, size=np.ceil(train_batch_size * .8).astype('int'), replace=False)
        sampled_batch = {
            'source_texts': [batch['source_texts'][idx] for idx in sample_batch_idx],
            'class_labels': batch['class_labels'][sample_batch_idx]
        }

        if self.compute_rew_batch_mode:
            true_rewards_candidates = self.compute_rewards_multiple_prompts(batch=sampled_batch, prompt_strings=prompt_strings_candidates)
        else:
            true_rewards_candidates = self.compute_rewards(batch=sampled_batch, prompt_strings=prompt_strings_candidates)

        # get the reward-model predictions
        model_rewards_candidates = self.module.forward(input_ids=output_ids_candidates,
                                              attention_mask=torch.ones(output_ids_candidates.shape, dtype=torch.long, device=output_ids_candidates.device))

        model_rewards_candidates = self.agg_func(model_rewards_candidates)

        # create batch of training samples from `true_rewards_candidates` and `model_rewards_candidates`
        true_rewards = []
        model_rewards = []
        for _ in range(self.reward_learning_batch_size):
            # sample `reward_learning_samples` prompts
            sampled_idx = np.random.choice(total_num_prompts, size=self.reward_learning_samples, replace=(total_num_prompts < self.reward_learning_samples))
            true_rewards.append(true_rewards_candidates[sampled_idx].unsqueeze(0))
            model_rewards.append(model_rewards_candidates[sampled_idx].unsqueeze(0))

        true_rewards = torch.cat(true_rewards, dim=0)
        model_rewards = torch.cat(model_rewards, dim=0)

        assert true_rewards.shape == model_rewards.shape == (self.reward_learning_batch_size, self.reward_learning_samples)

        # calculate the reward loss
        loss = listMLELoss(y_pred=model_rewards, y_true=true_rewards, listmle_temp=1.)

        return loss

    def get_valid_loss(self, eval_dataloader, num_steps):
        self.module.eval()
        log_valid_loss = 0.
        len_eval_dataloader = len(eval_dataloader)
        with torch.no_grad():
            for _ in range(num_steps // len_eval_dataloader):
                for batch in eval_dataloader:
                    loss = self.get_loss(batch=batch, mode='train')
                    log_valid_loss += float(loss.item()) / num_steps

        self.module.train()
        return log_valid_loss

    def train_step(self, batch):
        loss_val = 0.
        for _ in range(self.gradient_accumulation_steps):
            loss = self.get_loss(batch=batch, mode='train') / self.gradient_accumulation_steps
            loss.backward()
            loss_val += float(loss.item())

        self.train_op()
        self._num_train_steps += 1

        return loss_val

    def train_one_epoch(self, train_dataloader, num_steps, epoch_count):
        self.module.train()
        log_train_loss = 0.
        start_time = time.time()
        len_train_dataloader = len(train_dataloader)

        for idx in range(num_steps // len_train_dataloader):
            for step, batch in enumerate(train_dataloader):
                loss_one_step = self.train_step(batch)
                log_train_loss += loss_one_step
                if self._num_train_steps % self.report_interval == 0:
                    step_curr_epoch = idx * len_train_dataloader + (step + 1)
                    print(f"{self.print_prefix} [E:{epoch_count}|{step_curr_epoch}/{num_steps}] "
                          f"total iter: {self._num_train_steps}/{self.max_train_steps}, minibatch loss: {loss_one_step:.4f}, "
                          f"average train loss: {log_train_loss / step_curr_epoch:.4f}, time: {(time.time() - start_time) / 60:.1f} min",
                          flush=True
                          )

        return log_train_loss / num_steps

    def train(self):
        self.optimizer = optim.Adam(self.module.parameters(), lr=self.rew_learning_rate)
        self.optimizer.zero_grad()
        self._num_train_steps = 0
        # Create saving path
        ckpt_save_dir = os.path.join(self.save_dir, "model_ckpt")
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)
        save_loc = os.path.join(ckpt_save_dir, WEIGHTS_NAME)

        train_dataloader = self._get_train_dataloader()
        eval_dataloader = self._get_eval_dataloader(self.eval_dataset)

        print("*" * 60, flush=True)
        print(f'{self.print_prefix} [{self._num_train_steps}/{self.max_train_steps}] START Reward Training! save_loc = {save_loc}'
              f'\nlen(train_dataloader)={len(train_dataloader)}, len(eval_dataloader)={len(eval_dataloader)} !!!', flush=True)
        print("*" * 30, flush=True)

        prev_min_loss = 1e10
        early_stop_count = self.early_stop_count

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_decay,
            patience=self.weight_decay_count,
            min_lr=0.000001,
            verbose=True
        )

        start_time = time.time()
        for epoch in range(self.num_train_epochs):
            _ = self.train_one_epoch(train_dataloader, self.steps_per_epoch, epoch)
            valid_loss = self.get_valid_loss(eval_dataloader, num_steps=self._num_valid_steps)
            scheduler.step(valid_loss)
            valid_info = f"{self.print_prefix} [Epoch {epoch + 1}/{self.num_train_epochs}|Step {self._num_train_steps}/{self.max_train_steps}] " \
                         f"valid loss: {valid_loss:.4f}, prev_min_loss: {prev_min_loss:.4f}, time: {(time.time() - start_time) / 60:.1f} min"
            print("-" * len(valid_info), flush=True)
            print(valid_info, flush=True)

            if valid_loss < prev_min_loss * (1 - 1e-4):
                early_stop_count = self.early_stop_count
                prev_min_loss = valid_loss
                torch.save({"steps": self._num_train_steps, "model_state_dict": self.module.state_dict()}, save_loc)
                print(f'{self.print_prefix} [Epoch {epoch + 1}] Model saved to {save_loc} !!!', flush=True)
            else:
                early_stop_count -= 1
                print(f'{self.print_prefix} [Epoch {epoch + 1}] early stop countdown {early_stop_count}/{self.early_stop_count} !!!', flush=True)

            print("-" * len(valid_info), flush=True)
            if (early_stop_count == 0) or (self._num_train_steps >= self.max_train_steps):
                break

        # load the best reward model
        self._load_checkpoint(save_loc)
        valid_loss = self.get_valid_loss(eval_dataloader, num_steps=self._num_valid_steps)
        print_info = f"{self.print_prefix} [Loaded Model] total iter: {self._num_train_steps} " \
                     f"valid loss: {valid_loss:.4f}, time: {(time.time() - start_time) / 60:.1f} min !!!"
        print(print_info, flush=True)
        print("*" * len(print_info), flush=True)

        return
