import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import WEIGHTS_NAME
from functools import partial
from math import ceil
import os
import time
import evaluate
from utils import (
    _build_one_layer_mlp,
    _init_weights,
    prints,
    print_banner,
    get_decoded_preds_labels_from_batch
)
from reward_losses import listMLELoss


def logsumexp(x: torch.tensor, x_mask, replace_pad_value_to, temp: float, reward_learning_samples):
    x = torch.where(x_mask == 1, x, replace_pad_value_to)
    rew = temp * torch.logsumexp(x / temp, dim=-1)
    final_rew = multiply_rew_with_avg_traj_len(rew, x_mask, reward_learning_samples)
    return final_rew


def tensor_sum(x: torch.tensor, x_mask, reward_learning_samples):
    x = x * x_mask.float()
    rew = torch.sum(x, dim=-1)
    final_rew = rew.reshape(-1, reward_learning_samples)
    return final_rew


def tensor_average(x: torch.tensor, x_mask, reward_learning_samples):
    x = x * x_mask.float()
    x_sum = torch.sum(x, dim=-1)
    num_non_masked = torch.sum(x_mask, dim=-1).float()
    rew = x_sum / num_non_masked
    final_rew = multiply_rew_with_avg_traj_len(rew, x_mask, reward_learning_samples)
    return final_rew


def multiply_rew_with_avg_traj_len(rew, rew_mask, reward_learning_samples):
    rew = rew.reshape(-1, reward_learning_samples)
    trajectory_len = torch.sum(rew_mask, dim=-1).float()
    trajectory_len = trajectory_len.reshape(-1, reward_learning_samples)
    avg_trajectory_len = trajectory_len.mean(-1, keepdim=True)
    return rew * avg_trajectory_len


class RewardModel(nn.Module):
    def __init__(
            self,
            args,
            reward_transformer,
    ):
        super().__init__()
        self.args = args
        self.device = self.args.device

        self.transformer = reward_transformer
        model_dim = self.transformer.config.hidden_size
        self.mlp = _build_one_layer_mlp(in_dim=model_dim, out_dim=1, hidden_size=2048).to(self.device)
        self.mlp.apply(_init_weights)

        assert args.use_softplus in (0, 1)
        self.use_softplus = args.use_softplus == 1

        if self.use_softplus:
            self.max_r = torch.tensor(1., device=self.device)
            self.min_r = torch.tensor(1e-8, device=self.device)

        self.print_prefix = "[RewardModel]"
        prints(f"\n{self.print_prefix} Init RewardModel use_softplus={self.use_softplus}, "
               f"(min_r, max_r)={(self.min_r.item(), self.max_r.item()) if self.use_softplus else (0, 1)}")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
    ):
        last_hidden_states = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        ).last_hidden_state
        logits = self.mlp(last_hidden_states)

        if self.use_softplus:
            logits = self.max_r - F.softplus(self.max_r - logits)
            logits = self.min_r + F.softplus(logits - self.min_r)
        else:
            logits = torch.sigmoid(logits)

        logits = logits.nan_to_num(nan=0.0)
        logits = logits.squeeze(-1)
        return logits

    def get_embeddings(self, inputs):
        pass


class RewardModelTrainer(object):
    def __init__(
            self,
            args,
            reward_model,
            policy,
            tokenizer,
            metric,
            rew_train_dataloader,
            rew_eval_dataloader,
    ):
        self.module = reward_model
        self.policy = policy
        self.tokenizer = tokenizer
        self.metric = evaluate.load('meteor')
        self.train_dataloader = rew_train_dataloader
        self.eval_dataloader = rew_eval_dataloader

        self.args = args
        self.device = self.args.device

        ######## Training Params ########
        self.save_dir = os.path.join(args.output_dir, "reward_model")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # optimization param
        self.reward_learning_batch_size = args.reward_learning_batch_size
        self.rew_gradient_accumulation_steps = args.rew_gradient_accumulation_steps
        self.rew_learning_rate = args.rew_learning_rate
        assert args.gradient_clip in (0, 1)
        self.gradient_clip = args.gradient_clip == 1
        self.gradient_clip_norm = args.gradient_clip_norm
        self.lr_decay = args.lr_decay
        self.weight_decay_count = args.weight_decay_count
        self.early_stop_count = args.early_stop_count

        ##### reward-learning specific param #####
        self.reward_learning_samples = args.reward_learning_samples
        self.agg_func = args.agg_func
        self.soft_maxmin_temp = args.soft_maxmin_temp

        ########################################
        # training controls
        self.num_updates_per_epoch = (
                len(self.train_dataloader) // self.rew_gradient_accumulation_steps +
                int(len(self.train_dataloader) % self.rew_gradient_accumulation_steps != 0)
        )

        self.rew_num_train_epochs = args.rew_num_train_epochs
        self.rew_eval_period = max(ceil(args.rew_eval_period * self.num_updates_per_epoch), 1)
        self.num_valid_batch = max(int(0.2 * len(rew_eval_dataloader)), 1)
        # use 20% of the `rew_eval_dataloader` to compute the validation loss (which saves time)

        ######## Training Setups ########
        self.optimizer = optim.Adam(self.module.parameters(), lr=self.rew_learning_rate)
        self.num_train_steps = 0
        self.report_interval = max(self.num_updates_per_epoch // 100, 1)

        self.gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else self.module.transformer.config.max_length,
            "num_beams": args.num_beams,
        }

        ######## Aggregate Func. Setups ########
        if args.agg_func == "max":
            # select `max` so replace pad value to `-inf`
            replace_pad_value_to = torch.tensor(-float("Inf"), device=self.device)
            self.agg_func = partial(logsumexp, replace_pad_value_to=replace_pad_value_to, temp=args.soft_maxmin_temp, reward_learning_samples=self.reward_learning_samples)
        elif args.agg_func == "min":
            # select `min` so replace pad value to `inf`
            replace_pad_value_to = torch.tensor(float("Inf"), device=self.device)
            self.agg_func = partial(logsumexp, replace_pad_value_to=replace_pad_value_to, temp=((-1.) * args.soft_maxmin_temp), reward_learning_samples=self.reward_learning_samples)
        elif args.agg_func == "sum":
            self.agg_func = partial(tensor_sum, reward_learning_samples=self.reward_learning_samples)
        elif args.agg_func == "avg":
            self.agg_func = partial(tensor_average, reward_learning_samples=self.reward_learning_samples)
        else:
            raise NotImplementedError

        ########## Load checkpoint ###############
        if args.rew_checkpoint_path is not None:
            self._load_checkpoint(args.rew_checkpoint_path)

        #####################################
        self.print_prefix = "[RewardLearning]"
        prints(f"\n{self.print_prefix} len(train_dataloader): {len(self.train_dataloader)}, len(eval_dataloader): {len(self.eval_dataloader)}, rew_num_train_epochs: {self.rew_num_train_epochs}, "
               f"\nrew_learning_rate: {self.rew_learning_rate}, reward_learning_samples: {self.reward_learning_samples}, reward_learning_batch_size: {self.reward_learning_batch_size}, "
               f"\nrew_gradient_accumulation_steps: {self.rew_gradient_accumulation_steps}, num_updates_per_epoch: {self.num_updates_per_epoch}, \nagg_func: {self.agg_func}, \nsoft_maxmin_temp: {self.soft_maxmin_temp}, "
               f"rew_eval_period: {self.rew_eval_period}, num_valid_batch: {self.num_valid_batch}, report_interval: {self.report_interval}, env reward: {self.metric.name} \n")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.module.load_state_dict(checkpoint["model_state_dict"])
        print_info = f"{self.print_prefix} Loaded module from {checkpoint_path} !!!"
        print_banner(print_info, "^", front=True, back=True)

    def get_sample_summaries(self, batch):
        decoded_preds, decoded_labels, generated_tokens, generated_tokens_mask = get_decoded_preds_labels_from_batch(
            self.args, batch, self.policy, self.tokenizer,
            greedy_decoding=False, num_samples=self.reward_learning_samples, gen_kwargs=self.gen_kwargs,
            for_reward_training=True
        )
        return decoded_preds, decoded_labels, generated_tokens, generated_tokens_mask

    def compute_rewards(self, decoded_preds, decoded_labels):
        final_score = []
        for preds, labels in zip(decoded_preds, decoded_labels):
            final_score.append(self.metric.compute(predictions=[preds], references=[labels])['meteor'])

        final_score = torch.tensor(final_score).to(self.device)

        return final_score

    def get_loss(self, batch):

        # get sampled summaries
        decoded_preds, decoded_labels, generated_tokens, generated_tokens_mask = self.get_sample_summaries(batch)
        assert len(decoded_preds) == len(decoded_labels) == generated_tokens.shape[0] == generated_tokens_mask.shape[0] == batch['input_ids'].shape[0] * self.reward_learning_samples

        # get the corresponding true rewards
        true_rewards = self.compute_rewards(decoded_preds, decoded_labels)
        assert true_rewards.shape == (batch['input_ids'].shape[0] * self.reward_learning_samples,)
        true_rewards = true_rewards.reshape(-1, self.reward_learning_samples)
        assert true_rewards.shape == (batch['input_ids'].shape[0], self.reward_learning_samples)

        model_rewards = self.module.forward(
            input_ids=batch["input_ids"].repeat_interleave(self.reward_learning_samples, dim=0),
            attention_mask=batch["attention_mask"].repeat_interleave(self.reward_learning_samples, dim=0),
            decoder_input_ids=generated_tokens,
        )
        assert model_rewards.shape == (batch['input_ids'].shape[0] * self.reward_learning_samples, generated_tokens.shape[1])
        model_rewards = self.agg_func(model_rewards, generated_tokens_mask)
        assert model_rewards.shape == (batch['input_ids'].shape[0], self.reward_learning_samples)

        # calculate the reward loss
        loss = listMLELoss(y_pred=model_rewards, y_true=true_rewards, listmle_temp=1.)

        return loss

    def get_valid_loss(self):
        self.module.eval()
        log_valid_loss = 0.
        with torch.no_grad():
            for step, batch in enumerate(self.eval_dataloader):
                if step >= self.num_valid_batch:
                    continue
                else:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    loss = self.get_loss(batch=batch)
                    log_valid_loss += float(loss.item()) / self.num_valid_batch

        self.module.train()
        return log_valid_loss

    def train(self):
        self.module.train()
        self.policy.eval()
        self.optimizer = optim.Adam(self.module.parameters(), lr=self.rew_learning_rate)
        self.optimizer.zero_grad(set_to_none=True)
        self.num_train_steps = 0
        max_train_steps = ceil(self.num_updates_per_epoch * self.rew_num_train_epochs)
        save_loc = os.path.join(self.save_dir, WEIGHTS_NAME)
        info = f'{self.print_prefix} [{self.num_train_steps}/{max_train_steps}] START Reward Training! \nsave_loc = {save_loc}' \
               f'\nlen(train_dataloader): {len(self.train_dataloader)}, len(eval_dataloader): {len(self.eval_dataloader)} ' \
               f'rew_gradient_accumulation_steps: {self.rew_gradient_accumulation_steps}, rew_num_train_epochs: {self.rew_num_train_epochs} !!!'
        print_banner(info, "*", True, True)
        prev_min_loss = 1e10
        early_stop_count = self.early_stop_count
        early_stopped = False
        train_dataloader_last_idx = len(self.train_dataloader) - 1

        # lr scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_decay,
            patience=self.weight_decay_count,
            min_lr=0.000001,
            verbose=True
        )
        start_time = time.time()

        for epoch in range(ceil(self.rew_num_train_epochs)):
            self.module.train()
            epoch_total_loss = 0
            epoch_start_time = time.time()
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.get_loss(batch) / self.rew_gradient_accumulation_steps
                loss.backward()
                minibatch_loss = float(loss.item()) * self.rew_gradient_accumulation_steps
                epoch_total_loss += minibatch_loss

                if ((step + 1) % self.rew_gradient_accumulation_steps == 0) or (step == train_dataloader_last_idx):
                    if self.gradient_clip:
                        nn.utils.clip_grad_norm_(self.module.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # `self.num_train_steps` count the number of gradient updates
                    self.num_train_steps += 1

                    # callbacks
                    if (self.num_train_steps % self.rew_eval_period == 0) or (self.num_train_steps == max_train_steps):
                        valid_loss = self.get_valid_loss()
                        scheduler.step(valid_loss)
                        valid_info = f"{self.print_prefix} [E:{epoch+1}/{ceil(self.rew_num_train_epochs)}|{step+1}/{train_dataloader_last_idx+1}" \
                                     f"||S:{self.num_train_steps}/{max_train_steps}] " \
                                     f"valid loss: {valid_loss:.4f}, prev_min_loss: {prev_min_loss:.4f}, " \
                                     f"epoch: {(time.time() - epoch_start_time) / 60:.1f} min, total: {(time.time() - start_time) / 60:.1f} min"
                        prints("-" * len(valid_info))
                        prints(valid_info)

                        if valid_loss < prev_min_loss * (1 - 1e-4):
                            early_stop_count = self.early_stop_count
                            prev_min_loss = valid_loss
                            torch.save({"steps": self.num_train_steps, "model_state_dict": self.module.state_dict()}, save_loc)
                            prints(f'{self.print_prefix} [E:{epoch+1}/{ceil(self.rew_num_train_epochs)}|{step+1}/{train_dataloader_last_idx+1}||{self.num_train_steps}/{max_train_steps}] '
                                   f'Model saved to {save_loc} !!!')
                        else:
                            early_stop_count -= 1
                            prints(f'{self.print_prefix} [E:{epoch+1}/{ceil(self.rew_num_train_epochs)}|{step+1}/{train_dataloader_last_idx+1}||{self.num_train_steps}/{max_train_steps}]'
                                   f' early stop countdown {early_stop_count}/{self.early_stop_count} !!!')

                        prints("-" * len(valid_info))

                    if (self.num_train_steps % self.report_interval == 0) or (step == train_dataloader_last_idx) or (self.num_train_steps == max_train_steps):
                        prints(f"{self.print_prefix} [E:{epoch+1}/{ceil(self.rew_num_train_epochs)}|{step+1}/{train_dataloader_last_idx+1}] "
                               f"iter: {self.num_train_steps}/{max_train_steps}, batch loss: {minibatch_loss:.4f}, epoch avg loss: {epoch_total_loss / (step+1):.4f}, "
                               f"epoch: {(time.time() - epoch_start_time) / 60:.1f} min, total: {(time.time() - start_time) / 60:.1f} min")

                    if (early_stop_count == 0) or (self.num_train_steps >= max_train_steps):
                        early_stopped = True
                        break

            if early_stopped:
                info = f"{self.print_prefix} [E:{epoch+1}/{ceil(self.rew_num_train_epochs)}|{step+1}/{train_dataloader_last_idx+1}||{self.num_train_steps}/{max_train_steps}] " \
                       f"STOPPED!!! early_stop_count: {early_stop_count}"
                print_banner(info, "-", True, True)
                break

        del self.optimizer      # next call to `train` will initialize a new `self.optimizer` so this one is not needed any more
        self.module.eval()      # will not train the reward model, set it to eval mode
        self.policy.train()     # may need to continue training policy, set it to train mode
        return
