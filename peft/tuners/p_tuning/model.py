# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on https://github.com/huggingface/peft/blob/main/src/peft/tuners/p_tuning/model.py
# with some refactor
#
# Modifications by Sergio Burdisso (Idiap Research Institute © 2025):
#   - Added support for P_PROJECTOR method
#   - Added support for user-provided embedding initialization (virtual_token_embs)
#   - Added RESIDUAL encoder reparameterization type
# These modifications are licensed under the MIT License.
# SPDX-License-Identifier: Apache-2.0 AND MIT
import warnings

import torch

from .config import PromptEncoderConfig, PromptEncoderReparameterizationType
from peft.utils import PeftType


class PromptEncoder(torch.nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for p-tuning or prompt projector (pp).

    Args:
        config ([`PromptEncoderConfig`]): The configuration of the prompt encoder.

    Example:

    ```py
    >>> from peft import PromptEncoder, PromptEncoderConfig

    >>> config = PromptEncoderConfig(
    ...     peft_type="P_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_reparameterization_type="MLP",
    ...     encoder_hidden_size=768,
    ... )

    >>> prompt_encoder = PromptEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt encoder.
        - **mlp_head** (`torch.nn.Sequential`) -- The MLP head of the prompt encoder if `inference_mode=False`.
        - **lstm_head** (`torch.nn.LSTM`) -- The LSTM head of the prompt encoder if `inference_mode=False` and
        `encoder_reparameterization_type="LSTM"`.
        - **token_dim** (`int`) -- The hidden embedding dimension of the base transformer model.
        - **input_size** (`int`) -- The input size of the prompt encoder.
        - **output_size** (`int`) -- The output size of the prompt encoder.
        - **hidden_size** (`int`) -- The hidden size of the prompt encoder.
        - **total_virtual_tokens** (`int`): The total number of virtual tokens of the
        prompt encoder.
        - **encoder_type** (Union[[`PromptEncoderReparameterizationType`], `str`]): The encoder type of the prompt
          encoder.


    Input shape: (`batch_size`, `total_virtual_tokens`)

    Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config):
        super().__init__()
        self.token_dim = config.token_dim
        self.input_size = self.token_dim
        self.output_size = self.token_dim
        self.hidden_size = config.encoder_hidden_size
        self.total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.encoder_type = config.encoder_reparameterization_type

        # embedding
        if config.virtual_token_embs is None:
            self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(config.virtual_token_embs,
                                                                freeze=config.peft_type == PeftType.P_PROJECTOR)
            config.virtual_token_embs = None # Tensor is not JSON serializable, setting to None for config to be saved without errors

        if not config.inference_mode:
            if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
                lstm_dropout = config.encoder_dropout
                num_layers = config.encoder_num_layers
                # LSTM
                self.lstm_head = torch.nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    bidirectional=True,
                    batch_first=True,
                )

                self.mlp_head = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size * 2, self.output_size),
                )

            elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
                encoder_num_layers_default = PromptEncoderConfig.encoder_num_layers
                if config.encoder_num_layers != encoder_num_layers_default:
                    warnings.warn(
                        f"for {self.encoder_type.value}, the argument `encoder_num_layers` is ignored. "
                        f"Exactly {encoder_num_layers_default} MLP layers are used."
                    )
                layers = [
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),  
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.output_size),
                ]
                self.mlp_head = torch.nn.Sequential(*layers)
            elif self.encoder_type == PromptEncoderReparameterizationType.RESIDUAL:
                encoder_num_layers_default = PromptEncoderConfig.encoder_num_layers
                if config.encoder_num_layers != encoder_num_layers_default:
                    warnings.warn(
                        f"for {self.encoder_type.value}, the argument `encoder_num_layers` is ignored. "
                        f"Exactly {encoder_num_layers_default} MLP layers are used."
                    )
                layers = [
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.output_size)
                ]
                self.mlp_head = torch.nn.Sequential(*layers)
            elif self.encoder_type == PromptEncoderReparameterizationType.EMB:
                encoder_num_layers_default = PromptEncoderConfig.encoder_num_layers
                if config.encoder_num_layers != encoder_num_layers_default:
                    warnings.warn(
                        f"for {self.encoder_type.value}, the argument `encoder_num_layers` is ignored. "
                        f"Exactly {encoder_num_layers_default} MLP layers are used."
                    )
                self.mlp_head = torch.nn.Identity()
            else:
                raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended), LSTM, or RESIDUAL.")

    def forward(self, indices):
        input_embeds = self.embedding(indices)
        if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderReparameterizationType.RESIDUAL:
            output_embeds = self.mlp_head(input_embeds)
            output_embeds += input_embeds
        elif self.encoder_type == PromptEncoderReparameterizationType.EMB:
            output_embeds = self.mlp_head(input_embeds)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

        return output_embeds
