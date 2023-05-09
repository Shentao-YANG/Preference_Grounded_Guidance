import torch
from torch import nn, optim
import torch.nn.functional as F
import os
from transformers import WEIGHTS_NAME, get_scheduler
import time
from utils import (
    prints,
    print_banner,
    get_decoded_preds_labels_from_batch
)


def reward_to_Q_value(x: torch.tensor):
    return torch.flip(torch.cumsum(torch.flip(x, dims=[-1]), dim=-1), dims=[-1])


def reward_to_reward(x: torch.tensor):
    return x


class RewardPolicyTrainer(object):
    def __init__(
            self,
            args,
            policy,
            reward_trainer,
            tokenizer,
            metric,
            train_dataloader,
            eval_dataloader,
    ):
        self.args = args
        self.policy = policy
        self.reward_trainer = reward_trainer
        self.reward_model = self.reward_trainer.module
        self.tokenizer = tokenizer
        self.metric = metric
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = self.args.device
        self.nll_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.label_pad_token = torch.tensor(-100, device=self.device)
        self.tokenizer_pad_id = torch.tensor(self.tokenizer.pad_token_id, device=self.device)

        self.save_dir = os.path.join(args.output_dir, "policy_model")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        ############ loss parameters ############
        self.max_entropy_coeff = args.max_entropy_coeff
        self.reinforce_coeff = args.reinforce_coeff
        self.num_reinforce_samples = args.num_reinforce_samples

        assert args.use_q_for_weight in (0, 1)
        if args.use_q_for_weight == 1:  # use q for weight
            self.rew_to_weight_func = reward_to_Q_value
        else:  # use r for weight
            self.rew_to_weight_func = reward_to_reward

        assert args.exp_in_wmle in (0, 1)
        self.exp_in_wmle = args.exp_in_wmle == 1

        ############ Optimization parameters ############
        self.learning_rate = args.learning_rate

        self.train_batch_size = args.per_device_train_batch_size
        self.eval_batch_size = args.per_device_eval_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        assert args.gradient_clip in (0, 1)
        self.gradient_clip = args.gradient_clip == 1
        self.gradient_clip_norm = args.gradient_clip_norm
        self.lr_decay = args.lr_decay
        self.weight_decay_count = args.weight_decay_count
        self.early_stop_count = args.early_stop_count

        ########################################
        # training controls
        self.num_updates_per_epoch = (
                len(self.train_dataloader) // self.gradient_accumulation_steps +
                int(len(self.train_dataloader) % self.gradient_accumulation_steps != 0)
        )

        self.num_train_epochs = args.num_train_epochs
        self.eval_period = max(int(args.policy_eval_period * self.num_updates_per_epoch), 1)
        self.num_valid_batch = max(int(1. * len(self.eval_dataloader)), 1)
        self.reward_retrain_period = max(int(args.reward_retrain_period * self.num_updates_per_epoch), 1)

        ######## Training Setups ########
        assert args.reset_optim in (0, 1)
        self.reset_optim = args.reset_optim == 1
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        self.optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.policy.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.policy.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optim.AdamW(self.optimizer_grouped_parameters, lr=self.learning_rate)
        self.num_train_steps = 0
        self.report_interval = max(self.num_updates_per_epoch // 100, 1)

        self.gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else self.policy.config.max_length,
            "num_beams": args.num_beams,
        }

        ########## Load checkpoint ###############
        if args.policy_checkpoint_path is not None:
            self._load_checkpoint(args.policy_checkpoint_path, load_rew_ckpt=False)
        ################################################
        self.print_prefix = "[PolicyLearning]"

        prints(f"\n{self.print_prefix} max_entropy_coeff={self.max_entropy_coeff}, reinforce_coeff={self.reinforce_coeff}, num_reinforce_samples={self.num_reinforce_samples}, use_q_for_weight={args.use_q_for_weight == 1}, "
               f"\nlearning_rate={self.learning_rate}, save_dir={self.save_dir}"
               f"\ntrain_batch_size={self.train_batch_size}, eval_batch_size={self.eval_batch_size}, gradient_accumulation_steps={self.gradient_accumulation_steps}, num_updates_per_epoch={self.num_updates_per_epoch}, "
               f"num_train_epochs={self.num_train_epochs} \nnum_valid_batch={self.num_valid_batch}, eval_period={self.eval_period}, "
               f"reward_retrain_period={self.reward_retrain_period}, report_interval={self.report_interval}, exp_in_wmle={self.exp_in_wmle}, weight_decay={args.weight_decay}, reset_optim={self.reset_optim} \n")

    def _load_checkpoint(self, checkpoint_path: str, load_rew_ckpt: bool):
        # checkpoint_path: path to the **policy** checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])

        reward_model_path = os.path.join(self.args.output_dir, "reward_model", WEIGHTS_NAME)
        if load_rew_ckpt:
            self.reward_trainer._load_checkpoint(checkpoint_path=reward_model_path)

        print_info = f"{self.print_prefix} Loaded module from {checkpoint_path}; " \
                     f"Loaded reward model from {reward_model_path if load_rew_ckpt else 'NO LOADING'} !!!"
        print_banner(print_info, "^", True, True)

    def get_weighted_mle_loss(self, batch):
        with torch.no_grad():
            model_reward = self.reward_model.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=torch.where(batch["labels"] < 0, self.tokenizer_pad_id, batch["labels"]),
            )
        if self.exp_in_wmle:
            model_reward = model_reward.exp()

        # get the per-label-token NLL
        policy_logits = self.policy(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=batch['decoder_input_ids']
        ).logits

        loss = self.nll_loss_fct(policy_logits.view(-1, policy_logits.shape[-1]), batch["labels"].view(-1))
        loss = loss.reshape(batch["labels"].shape[0], -1)

        model_reward_with_mask = model_reward * batch["labels_mask"]
        model_reward_with_mask = self.rew_to_weight_func(model_reward_with_mask)
        model_reward_with_mask = model_reward_with_mask / model_reward_with_mask.sum(dim=-1, keepdims=True)

        reward_weighted_loss = model_reward_with_mask * loss

        final_loss = reward_weighted_loss.sum(dim=-1).mean()

        assert final_loss.requires_grad
        assert final_loss.isfinite()

        return final_loss, [round(x, 4) for x in model_reward_with_mask.detach()[:, :5].median(dim=0).values.cpu().tolist()]

    def sample_summaries_with_scores(self, batch):
        # generated sequences starts from index 1
        gen = self.policy.generate_with_grad(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=True,
            num_return_sequences=self.num_reinforce_samples,
            return_dict_in_generate=True,
            output_scores=True,
            **self.gen_kwargs,
        )
        gen['sequences'] = gen.sequences[:, 1:]
        gen['scores'] = torch.cat([logits.unsqueeze(dim=1) for logits in gen.scores], dim=1)
        gen['scores'] = gen['scores'].nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)
        gen['sequences_mask'] = (gen.sequences.detach() != self.tokenizer_pad_id).float()

        assert len(gen.scores.shape) == 3 and gen.scores.shape[:2] == gen.sequences.shape == gen.sequences_mask.shape
        assert gen.scores.requires_grad

        return gen

    def get_reinforce_loss(self, batch, gen):
        with torch.no_grad():
            model_reward = self.reward_model.forward(
                input_ids=batch["input_ids"].repeat_interleave(self.num_reinforce_samples, dim=0),
                attention_mask=batch["attention_mask"].repeat_interleave(self.num_reinforce_samples, dim=0),
                decoder_input_ids=gen.sequences,
            )

        assert model_reward.shape == (batch["input_ids"].shape[0] * self.num_reinforce_samples, gen.sequences.shape[1])
        model_reward = model_reward * gen.sequences_mask
        model_reward = self.rew_to_weight_func(model_reward)

        gen_scores_flatten = gen.scores.reshape(-1, gen.scores.shape[-1])

        NLL = self.nll_loss_fct(gen_scores_flatten, gen.sequences.reshape(-1)).reshape(batch["input_ids"].shape[0] * self.num_reinforce_samples, -1)
        assert NLL.shape == gen.sequences.shape

        loss = (model_reward * NLL).sum() / gen.sequences_mask.sum()
        assert loss.requires_grad
        assert loss.isfinite()

        return loss, [round(x, 4) for x in model_reward.detach()[:, :5].median(dim=0).values.cpu().tolist()]

    def get_neg_entropy(self, batch, gen):
        output_probs = F.softmax(gen.scores, dim=-1) + 1e-8
        neg_output_entropy = output_probs * torch.log(output_probs)
        neg_output_entropy = neg_output_entropy.sum(dim=-1)

        loss = (neg_output_entropy * gen.sequences_mask).sum() / gen.sequences_mask.sum()
        assert loss.requires_grad
        assert loss.isfinite()

        return loss

    def get_loss(self, batch):
        batch["labels_mask"] = (batch["labels"] != self.label_pad_token).float()
        gen = None
        log_dict = {}

        loss, label_rew = self.get_weighted_mle_loss(batch)
        log_dict['WMLE'] = round(float(loss.detach().cpu().item()), 4)
        log_dict["label_rew"] = label_rew

        if self.reinforce_coeff > 0.:
            if gen is None:
                gen = self.sample_summaries_with_scores(batch)
            reinforce_loss, gen_rew = self.get_reinforce_loss(batch, gen)
            loss = loss + self.reinforce_coeff * reinforce_loss
            log_dict['reinforce'] = round(float(reinforce_loss.detach().cpu().item()), 4)
            log_dict['gen_rew'] = gen_rew

        if self.max_entropy_coeff > 0.:
            if gen is None:
                gen = self.sample_summaries_with_scores(batch)
            neg_entropy = self.get_neg_entropy(batch, gen)
            loss = loss + self.max_entropy_coeff * neg_entropy
            log_dict['neg_ent'] = round(float(neg_entropy.detach().cpu().item()), 4)

        assert loss.requires_grad

        return loss, log_dict

    def get_valid_scores(self):
        self.policy.eval()
        with torch.no_grad():
            for step, eval_batch in enumerate(self.eval_dataloader):
                if step >= self.num_valid_batch:
                    continue
                else:
                    eval_batch = {k: v.to(self.device) for k, v in eval_batch.items()}
                    decoded_preds, decoded_labels = get_decoded_preds_labels_from_batch(
                        self.args, eval_batch, self.policy, self.tokenizer, greedy_decoding=True,
                        num_samples=1, gen_kwargs=self.gen_kwargs, for_reward_training=False
                    )
                    self.metric.add_batch(
                        predictions=decoded_preds,
                        references=decoded_labels,
                    )

        result = self.metric.compute(use_stemmer=True)
        score = result['rouge1'] + 2 * result['rouge2'] + result['rougeLsum']

        self.policy.train()
        return score * 100., {'rouge1': round(result['rouge1'] * 100., 2), 'rouge2': round(result['rouge2'] * 100., 2), 'rougeLsum': round(result['rougeLsum'] * 100., 2)}

    def train(self):

        self.policy.train()
        max_train_steps = int(self.num_updates_per_epoch * self.num_train_epochs)
        save_loc = os.path.join(self.save_dir, WEIGHTS_NAME)

        print("*" * 80)
        print(f"{self.print_prefix} START Policy-Reward Joint Learning! "
              f"\nlen(train_dataloader)={len(self.train_dataloader)}, len(self.eval_dataloader)={len(self.eval_dataloader)}, max_train_steps={max_train_steps}, "
              f"\nsave_loc={save_loc}, \ngradient_accumulation_steps: {self.gradient_accumulation_steps}, num_train_epochs: {self.num_train_epochs} !!!")

        prev_max_score = -1e10
        early_stop_count = self.early_stop_count
        early_stopped = False
        train_dataloader_last_idx = len(self.train_dataloader) - 1

        # lr scheduler
        scheduler = get_scheduler(
            name="constant",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps
        )

        start_time = time.time()

        # train the reward model before starting the policy learning
        self.reward_trainer.train()
        num_reward_model = 1

        for epoch in range(self.num_train_epochs):
            self.policy.train()
            epoch_total_loss = 0
            epoch_start_time = time.time()
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, log_dict = self.get_loss(batch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                minibatch_loss = float(loss.item()) * self.gradient_accumulation_steps
                epoch_total_loss += minibatch_loss

                if ((step + 1) % self.gradient_accumulation_steps == 0) or (step == train_dataloader_last_idx):
                    # one gradient-update operation
                    if self.gradient_clip:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    # `self.num_train_steps` count the number of gradient updates
                    self.num_train_steps += 1

                    # callbacks
                    # everytime we reach here, `self.num_train_steps` will be `+1`, so we will not have wrong repetition
                    # 1 <= self.num_train_steps <= max_train_steps
                    if (self.num_train_steps % self.eval_period == 0) or (self.num_train_steps == max_train_steps):
                        self.policy.eval()
                        valid_score, valid_dict = self.get_valid_scores()
                        self.policy.train()
                        valid_info = f"{self.print_prefix} [E:{epoch+1}/{self.num_train_epochs}|{step+1}/{train_dataloader_last_idx+1}" \
                                     f"||S:{self.num_train_steps}/{max_train_steps}] " \
                                     f"valid score: {valid_score:.4f}, prev_max_score: {prev_max_score:.4f}, " \
                                     f"epoch: {(time.time() - epoch_start_time) / 60:.1f} min, total: {(time.time() - start_time) / 60:.1f} min"
                        valid_info = valid_info + f"\nMetric Scores: {valid_dict}"
                        prints("-" * len(valid_info))
                        prints(valid_info)
                        prev_max_score = max(prev_max_score, valid_score)

                        if self.num_train_steps == max_train_steps:
                            torch.save({"steps": self.num_train_steps, "model_state_dict": self.policy.state_dict()}, save_loc)
                            prints(
                                f'{self.print_prefix} [E:{epoch + 1}/{self.num_train_epochs}|{step + 1}/{train_dataloader_last_idx+1}||{self.num_train_steps}/{max_train_steps}] '
                                f'Model saved to {save_loc} !!!'
                            )

                        prints("-" * len(valid_info))

                    if (self.num_train_steps % self.report_interval == 0) or (step == train_dataloader_last_idx):
                        info = f"{self.print_prefix} [E:{epoch+1}/{self.num_train_epochs}|{step+1}/{train_dataloader_last_idx+1}||{self.num_train_steps}/{max_train_steps}], " \
                               f"epoch: {(time.time() - epoch_start_time) / 60:.1f} min, total: {(time.time() - start_time) / 60:.1f} min "
                        info = info + f"\nbatch loss: {minibatch_loss:.4f}, epoch avg loss: {epoch_total_loss / (step+1):.4f}, "
                        info = info + ", ".join(f"{k}: {v}" for k, v in log_dict.items() if "rew" not in k) + "\n"
                        info = info + "-" * 40 + "\n"
                        info = info + ", ".join(f"{k}: {v}" for k, v in log_dict.items() if "rew" in k)
                        info = info + "\n" + "-" * 40
                        prints(info)

                    if self.num_train_steps >= max_train_steps:
                        early_stopped = True
                        break

                    # 1 <= self.num_train_steps < max_train_steps and early_stop_count > 0
                    # retrain the reward model, only on the first two epochs (epoch=0,1)
                    if (epoch < 2) and (self.num_train_steps % self.reward_retrain_period == 0):
                        num_reward_model += 1
                        print_info = f"{self.print_prefix} [E:{epoch+1}/{self.num_train_epochs}|{step+1}/{train_dataloader_last_idx+1}||{self.num_train_steps}/{max_train_steps}] " \
                                     f"Retrain Reward Model ({num_reward_model}-th reward model) !!!"
                        print_banner(print_info, "*", True, True)
                        self.reward_trainer.train()
                        # set the policy model to train mode
                        self.policy.train()
                        if self.reset_optim:
                            info = f"{self.print_prefix} [E:{epoch + 1}/{self.num_train_epochs}|{step + 1}/{train_dataloader_last_idx + 1}||{self.num_train_steps}/{max_train_steps}] " \
                                   f"Reset Policy Optimizer !!!"
                            print_banner(info, "-", True, True)
                            self.optimizer = optim.AdamW(self.optimizer_grouped_parameters, lr=self.learning_rate)
                            scheduler = get_scheduler(
                                name="constant",
                                optimizer=self.optimizer,
                                num_warmup_steps=0,
                                num_training_steps=(max_train_steps - self.num_train_steps)
                            )

            if early_stopped:
                info = f"{self.print_prefix} [E:{epoch + 1}/{self.num_train_epochs}|{step + 1}/{train_dataloader_last_idx + 1}||{self.num_train_steps}/{max_train_steps}] " \
                       f"STOPPED!!! early_stop_count: {early_stop_count}"
                print_banner(info, "-", True, True)
                break

        print("*" * 80)

        del self.optimizer  # we will not need to train the model any more, so delete `optimizer`
        self.policy.eval()      # we will not need to train the policy, set it to eval mode
        return
