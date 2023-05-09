from dataclasses import dataclass
from typing import Optional

from rlprompt.modules import ReinforceModule
from rlprompt.models import BaseModel
from rlprompt.rewards import BaseReward


def make_reinforce_module(model: BaseModel,
                   reward: BaseReward,
                   reward_model: BaseModel,
                   config: "DictConfig") -> ReinforceModule:
    return ReinforceModule(model, reward, reward_model, config.top_k, config.top_p, config.num_beams,
                           config.num_reinforce_samples, config.max_entropy_coeff, config.use_q_for_weight)


@dataclass
class SQLModuleConfig:
    # Prompt generation setting
    top_k: Optional[int] = None
    top_p: float = 1.0
    num_beams: int = 1


@dataclass
class ReinforceModuleConfig:
    num_reinforce_samples: int = 64
    max_entropy_coeff: float = 0.1
    use_q_for_weight: int = 0       # default: do not use q for weight -> use r for weight
