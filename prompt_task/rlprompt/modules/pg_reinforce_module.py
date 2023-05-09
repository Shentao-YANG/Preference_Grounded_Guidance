import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Union, Tuple

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.rewards import BaseReward
from rlprompt.utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reward_to_Q_value(x: torch.tensor):
    return torch.flip(torch.cumsum(torch.flip(x, dims=[-1]), dim=-1), dims=[-1])


def reward_to_reward(x: torch.tensor):
    return x


class ReinforceModule(BaseModule):
    def __init__(
        self,
        model: BaseModel,
        reward: Optional[BaseReward],
        reward_model: BaseModel,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
        num_samples: int,
        max_entropy_coeff: float,
        use_q_for_weight: int      # default: use reward itself as the weight
    ):
        super().__init__()
        # Initialize self._model and self._reward
        assert not (top_k is not None and top_p < 1.0), "Only one of top_k or top_p should be selected"

        self._model = model
        self.reward_model = reward_model
        self._reward = reward

        self._top_k = top_k
        self._top_p = top_p
        self._num_beams = num_beams
        self.num_samples = num_samples
        self.max_entropy_coeff = max_entropy_coeff

        self.nll_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        if use_q_for_weight == 1:   # use q for weight
            self.rew_to_weight_func = reward_to_Q_value
        else:   # use r for weight
            self.rew_to_weight_func = reward_to_reward

        print(f"\nUse ReinforceModule for Policy Training !!! max_entropy_coeff={self.max_entropy_coeff}, "
              f"use_q_for_weight={use_q_for_weight == 1} \n", flush=True)

    def forward(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict]:
        """Calculate the REINFORCE loss for policy training"""

        (output_logits, output_tokens, output_ids, _) = self._decode_sampling(batch=batch, return_gs_action=False, num_samples=self.num_samples)
        assert len(output_logits.shape) == 3 and output_logits.requires_grad
        assert len(output_ids.shape) == 2

        attention_mask = torch.ones(output_ids.shape, dtype=torch.long, device=output_ids.device)

        output_logits_flatten = output_logits.reshape(-1, output_logits.shape[-1])

        with torch.no_grad():
            rewards = self.reward_model.forward(
                input_ids=output_ids,
                inputs_embeds=None,
                attention_mask=attention_mask
            )

        rewards = self.rew_to_weight_func(rewards)

        NLL = self.nll_loss_fct(output_logits_flatten, output_ids.view(-1)).reshape(output_logits.shape[0], -1)

        output_probs = F.softmax(output_logits, dim=-1) + 1e-8
        neg_output_entropy = output_probs * torch.log(output_probs)
        neg_output_entropy = neg_output_entropy.sum(dim=-1)

        # loss is REINFORCE loss + coeff * neg_entropy
        reward_loss = (rewards * NLL).mean() + self.max_entropy_coeff * neg_output_entropy.mean()
        assert reward_loss.requires_grad

        ###### For logging purpose, can be removed if too slow ######
        raw_rewards, rewards_log = self.compute_rewards(batch=batch, output_tokens=output_tokens[:16], mode="train")
        utils.add_prefix_to_dict_keys_inplace(rewards_log, prefix=f"model/rewards/")

        loss_log = {
            "model/rew_loss": float(reward_loss.detach().cpu().item()),
            "model/rews": [round(x, 4) for x in rewards.detach().median(dim=0).values.cpu().tolist()],
            f"model/rewards/raw": float(raw_rewards.detach().mean().item()),
        }
        loss_log = utils.unionize_dicts([loss_log, rewards_log])

        return reward_loss, loss_log

    def compute_rewards(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        to_tensor: bool = True,
        mode: str = "infer"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rewards_tensor, rewards_log = self._reward(
            **batch,
            output_tokens=output_tokens,
            to_tensor=to_tensor,
            mode=mode)

        rewards_tensor = rewards_tensor.to(device)            
        return rewards_tensor, rewards_log

    def infer(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:
        # greedy decoding
        return self._model.generate(**batch,
                                    do_sample=False,
                                    top_k=self._top_k,
                                    top_p=self._top_p,
                                    num_beams=self._num_beams,
                                    infer=True)

    def _decode_sampling(
        self,
        batch: Dict[str, Any],
        return_gs_action: bool = False,
        num_samples: int = None
    ) -> Tuple[torch.Tensor, List[List[str]], torch.tensor, torch.LongTensor]:
        # stochastic decoding
        outputs = self._model.generate(**batch,
                                       do_sample=True,
                                       top_k=self._top_k,
                                       top_p=self._top_p,
                                       num_beams=self._num_beams,
                                       return_gs_action=return_gs_action,
                                       num_samples=num_samples)

        return (outputs['sample_logits'].contiguous(),
                outputs['sample_tokens'],
                outputs['sample_ids'].contiguous(),
                outputs['sample_lengths'])
