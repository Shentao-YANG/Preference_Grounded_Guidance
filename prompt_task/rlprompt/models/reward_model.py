from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from .base_model import BaseModel
from .lm_adaptor_model import _build_one_layer_mlp, SUPPORTED_LMS, LM_HIDDEN_SIZES
from typing import Optional


class RewardModel(BaseModel):
    def __init__(
        self,
        # MLP-specific parameters
        policy_lm,
        hidden_size,
        use_softplus
    ):
        super().__init__()
        assert policy_lm in SUPPORTED_LMS
        model = policy_lm
        self.device = 0
        self.transformer = GPT2Model.from_pretrained(model).to(self.device)

        model_dim = LM_HIDDEN_SIZES[model]

        self.mlp = _build_one_layer_mlp(in_dim=model_dim, out_dim=1, hidden_size=hidden_size).to(self.device)       # r(s_t, a_t) is a scalar

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)

        self.mlp.apply(_init_weights)

        assert use_softplus in (0, 1)
        self.use_softplus = use_softplus == 1

        if self.use_softplus:
            self.max_r = torch.tensor(1., device=self.device)
            self.min_r = torch.tensor(1e-8, device=self.device)

        print(f"\n[RewardModel] use_softplus={self.use_softplus}, "
              f"(min_r, max_r)={(self.min_r.item(), self.max_r.item()) if self.use_softplus else (0, 1)}",
              flush=True)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        last_hidden_states = self.transformer(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

        logits = self.mlp(last_hidden_states)

        # scale rewards to [0,1]
        if self.use_softplus:
            logits = self.max_r - F.softplus(self.max_r - logits)
            logits = self.min_r + F.softplus(logits - self.min_r)
        else:   # use sigmoid
            logits = torch.sigmoid(logits)

        logits = logits.nan_to_num(nan=0.0)                         # replace nan with 0.0

        logits = logits.squeeze(-1)                                 # (batch_size, seq_len)

        return logits

    def get_embeddings(self, inputs):
        if len(inputs.shape) == 2:
            # inputs are token ids (integer)
            # inputs: (batch_size, seq_len)
            assert not torch.is_floating_point(inputs)
            input_embeds = self.transformer.wte(inputs)
        elif len(inputs.shape) == 3:
            # inputs are one-hot tensor over vocab (float)
            # inputs: (batch_size, seq_len, vocab_size)
            assert torch.is_floating_point(inputs)
            input_embeds = inputs @ self.transformer.wte.weight
        else:
            raise ValueError(f"inputs should have dimension 2 or 3, received shape {inputs.shape}")

        return input_embeds
