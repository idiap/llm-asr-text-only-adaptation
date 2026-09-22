# SPDX-FileCopyrightText: Copyright (c) 2024 SLAM-LLM contributors
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Modifications by Idiap Research Institute (© 2025):
#   - Added support for p-tuning and p-projector PEFT methods
#   - Added virtual token handling for prompt-based learning
#   - Added prompt token initialization and configuration
#   - Added support for batches containing text-only (audio-free) items

import os
import types
import copy
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.train_utils import print_module_size, print_model_size
from peft import PeftModel, PeftConfig

# from torch.nn import CrossEntropyLoss
from slam_llm.utils.metric import compute_accuracy

import pathlib
import numpy as np

import logging

logger = logging.getLogger(__name__)


def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    # If P-Tunning add prompt token...
    if train_config.use_peft and train_config.peft_config.peft_method.lower() in ["p-tuning", "p-projector"] and not train_config.prompt_num_virtual_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [train_config.prompt_token]})
        train_config.prompt_token_id = tokenizer.get_vocab()[train_config.prompt_token]
        logger.info(f"Adding prompt token to the tokenizer, prompt token id is {train_config.prompt_token_id}")

    # Processor added for llama3
    processor = setup_processor(model_config, **kwargs)

    # Speech Encoder
    encoder = setup_encoder(train_config, model_config, **kwargs)

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # projector
    encoder_projector = setup_encoder_projector(train_config, model_config, **kwargs)
    model = slam_model(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get(
        "ckpt_path", None
    )  # FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
        logger.info("Loading Speech Projector from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer, processor


# Added this funciton to get a speccial processor for prompt formatting in llama3
def setup_processor(model_config, **kwargs):
    processor = None
    return processor


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    if "vallex" in model_config.llm_name.lower():
        return None
    elif "mupt" in model_config.llm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.llm_path, trust_remote_code=True, use_fast=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    encoder_list = (
        model_config.encoder_name.split(",") if model_config.encoder_name else []
    )
    if len(encoder_list) == 0:
        return None
    if len(encoder_list) == 1:
        encoder_name = encoder_list[0]
        if encoder_name == "whisper" or encoder_name == "qwen-audio":
            from slam_llm.models.encoder import WhisperWrappedEncoder

            encoder = WhisperWrappedEncoder.load(model_config)
        if encoder_name == "beats":
            from slam_llm.models.encoder import BEATsEncoder

            encoder = BEATsEncoder.load(model_config)
        if encoder_name == "eat":
            from slam_llm.models.encoder import EATEncoder

            encoder = EATEncoder.load(model_config)
        if encoder_name == "SpatialAST":
            from slam_llm.models.encoder import SpatialASTEncoder

            encoder = SpatialASTEncoder.load(model_config)
        if encoder_name == "wavlm":
            from slam_llm.models.encoder import WavLMEncoder

            encoder = WavLMEncoder.load(model_config)
        if encoder_name == "av_hubert":
            from slam_llm.models.encoder import AVHubertEncoder

            encoder = AVHubertEncoder.load(model_config)
        if encoder_name == "hubert":
            from slam_llm.models.encoder import HubertEncoder

            encoder = HubertEncoder.load(model_config)
        if encoder_name == "musicfm":
            from slam_llm.models.encoder import MusicFMEncoder

            encoder = MusicFMEncoder.load(model_config)

        if "llama" in encoder_name.lower():
            from slam_llm.models.encoder import HfTextEncoder

            encoder = HfTextEncoder.load(model_config)

    print_module_size(
        encoder,
        encoder_name,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )

    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
    print_module_size(
        encoder,
        encoder_name,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return encoder


def setup_llm(train_config, model_config, **kwargs):
    from pkg_resources import packaging

    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None
    if (
        train_config.enable_fsdp or train_config.enable_ddp
    ) and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        # v = packaging.version.parse(torch.__version__)
        # verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        # if not verify_latest_nightly:
        #     raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
        #                     "please install latest nightly.")
        rank = int(os.environ["RANK"])
        if rank == 0:
            if "vallex" in model_config.llm_name.lower():
                from src.slam_llm.models.vallex.vallex_config import VallexConfig
                from src.slam_llm.models.vallex.vallex_model import VALLE

                vallex_config = VallexConfig(**model_config)
                model = VALLE(vallex_config)
            elif "aya" in model_config.llm_name.lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.llm_path,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.llm_path,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                )
        else:
            llama_config = AutoConfig.from_pretrained(model_config.llm_path)
            llama_config.use_cache = use_cache
            # with torch.device("meta"):
            if "aya" in model_config.llm_name.lower():
                model = AutoModelForSeq2SeqLM(llama_config)
            else:
                model = AutoModelForCausalLM(
                    llama_config
                )  # (FIX:MZY): torch 2.0.1 does not support `meta`

    else:
        if "vallex" in model_config.llm_name.lower():
            from src.slam_llm.models.vallex.vallex_config import VallexConfig
            from src.slam_llm.models.vallex.vallex_model import VALLE

            vallex_config = VallexConfig(**model_config)
            model = VALLE(vallex_config)
        elif "aya" in model_config.llm_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
    if (
        train_config.enable_fsdp or train_config.enable_ddp
    ) and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer

            model = BetterTransformer.transform(model)
        except ImportError:
            logger.warning(
                "Module 'optimum' not found. Please install 'optimum' it before proceeding."
            )

    print_module_size(
        model,
        model_config.llm_name,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if (
        train_config.freeze_llm
    ):  # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()

    print_module_size(
        model,
        model_config.llm_name,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model


def setup_encoder_projector(train_config, model_config, **kwargs):
    if model_config.encoder_projector == "linear":
        from slam_llm.models.projector import EncoderProjectorConcat

        encoder_projector = EncoderProjectorConcat(model_config)
    elif model_config.encoder_projector == "cov1d-linear":
        from slam_llm.models.projector import EncoderProjectorCov1d

        encoder_projector = EncoderProjectorCov1d(model_config)
    elif model_config.encoder_projector == "q-former":
        from slam_llm.models.projector import EncoderProjectorQFormer

        encoder_projector = EncoderProjectorQFormer(model_config)
    else:
        return None
    print_module_size(
        encoder_projector,
        model_config.encoder_projector,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return encoder_projector


class slam_model(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__()
        # modality encoder
        self.encoder = encoder

        # llm
        self.llm = llm

        # projector
        self.encoder_projector = encoder_projector

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

        if self.model_config.vq_encoder_out:
            logger.info(f"Enabling VQ on encoder_outs")

        if self.model_config.ctc_weight > 0:
            logger.info(
                f"Enabling CTC on encoder_outs with weight: {self.model_config.ctc_weight}"
            )

        self.total_align_cosine_sim_matrix_saved = 0

        if train_config.get("enable_deepspeed", False):

            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)

            for item in self.modules():
                if isinstance(item, nn.LayerNorm):
                    item.forward = types.MethodType(new_forward, item)

    def pairwise_cosine_sim(self, query, base, return_argmax=False):
        # query: [N, T, dim]
        # base: [V, dim]
        max_sim_ind = []

        for i in range(query.shape[0]):
            max_sim_ind_cur = []
            for j in range(query.shape[1]):
                inp_query = query[i, j].repeat(base.shape[0], 1)
                cos_sim = F.cosine_similarity(inp_query, base)
                if return_argmax:
                    cos_sim = torch.argmax(cos_sim)
                cos_sim = cos_sim.cpu().detach()
                max_sim_ind_cur.append(cos_sim)
            max_sim_ind.append(torch.stack(max_sim_ind_cur))
        return max_sim_ind

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get(
            "audio_mel_post_mask", None
        )  # 2x downsample for whisper

        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)
        visual = kwargs.get("visual", None)
        visual_mask = kwargs.get("visual_mask", None)

        # for text encoder
        instruct_ids = kwargs.get("instruct_ids", None)
        instruct_mask = kwargs.get("instruct_mask", None)

        modality_mask = kwargs.get("modality_mask", None)

        zh_data = kwargs.get("zh", None)
        en_data = kwargs.get("en", None)

        encoder_outs = None
        if audio_mel is not None or audio is not None or visual is not None:
            if self.train_config.freeze_encoder:  # freeze encoder
                self.encoder.eval()

            if self.model_config.encoder_name == "whisper":
                encoder_outs = self.encoder.extract_variable_length_features(
                    audio_mel.permute(0, 2, 1)
                )  # bs*seq*dim
            if self.model_config.encoder_name == "beats":
                encoder_outs, audio_mel_post_mask = self.encoder.extract_features(
                    audio_mel, audio_mel_mask
                )  # bs*seq*dim
            if self.model_config.encoder_name == "eat":
                encoder_outs = self.encoder.model.extract_features(
                    audio_mel.unsqueeze(dim=1),
                    padding_mask=None,
                    mask=False,
                    remove_extra_tokens=False,
                )["x"]
            if self.model_config.encoder_name == "SpatialAST":
                encoder_outs = self.encoder(
                    audio
                )  # output: [bs, seq_len=3+512, dim=768]
            if self.model_config.encoder_name == "wavlm":
                missing_mask = (audio_mask.sum(dim=1) == 0)                 
                if missing_mask.any(): # If any element in the batch has no audio data
                    valid_idx = (~missing_mask).nonzero(as_tuple=True)[0]
                    
                    if valid_idx.numel() == 0: #means all samples are missing
                        # Create a placeholder tensor for missing samples
                        encoder_outs = torch.zeros((audio.shape[0],1,self.encoder.config.encoder_embed_dim), dtype=audio.dtype, device =audio.device) 

                    elif valid_idx.numel() > 0 :
                        # Process only valid audio samples, a few samples in the batch
                        encoder_outs_valid = self.encoder.extract_features(
                            audio[valid_idx],
                            (1 - audio_mask[valid_idx])
                        )
                                                
                        encoder_outs = torch.zeros(
                            (audio.shape[0],) + encoder_outs_valid.shape[1:],  # (B, seq_len, hidden_dim)
                            dtype=encoder_outs_valid.dtype,
                            device=encoder_outs_valid.device,
                        )
                        encoder_outs[valid_idx] = encoder_outs_valid
                else:
                    encoder_outs = self.encoder.extract_features(
                        audio, 1 - audio_mask
                    )  # (FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
            if self.model_config.encoder_name == "hubert":
                results = self.encoder(source=audio, padding_mask=1 - audio_mask)
                if self.model_config.encoder_type == "pretrain":
                    encoder_outs, audio_mel_post_mask = (
                        results["x"],
                        results["padding_mask"],
                    )
                if self.model_config.encoder_type == "finetune":
                    encoder_outs, audio_mel_post_mask = (
                        results["encoder_out"],
                        results["padding_mask"],
                    )
                    encoder_outs = encoder_outs.transpose(0, 1)
            if self.model_config.encoder_name == "av_hubert":
                results = self.encoder(
                    source={"video": visual, "audio": audio}, padding_mask=visual_mask
                )  # bs*seq*dim
                encoder_outs, audio_mel_post_mask = (
                    results["encoder_out"],
                    results["padding_mask"],
                )
                encoder_outs = encoder_outs.transpose(0, 1)
                audio_mel_post_mask = (~audio_mel_post_mask).float()
            if self.model_config.encoder_name == "musicfm":
                encoder_outs = self.encoder.extract_features(
                    audio, padding_mask=None
                )  # MusicFM doesn't support padding mask
            if self.encoder is None:
                encoder_outs = audio_mel if audio_mel is not None else audio

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs_all = self.encoder_projector(encoder_outs)
                if isinstance(encoder_outs_all, tuple):
                    encoder_outs = encoder_outs_all[0]
                    encoder_outs_llm = encoder_outs_all[-1]
                else:
                    encoder_outs = encoder_outs_all
            if self.model_config.encoder_projector == "cov1d-linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        if (
            str(self.model_config.save_align_cosine_sim_matrix) != "none"
            and self.total_align_cosine_sim_matrix_saved < 30
        ):
            if not self.model_config.save_align_cosine_sim_matrix.exists():
                self.model_config.save_align_cosine_sim_matrix.mkdir(
                    parents=True, exist_ok=True
                )
            self.total_align_cosine_sim_matrix_saved += 1
            input_lens = (modality_mask == True).sum(dim=1)
            input_lens = torch.sub(input_lens, 1)
            if self.train_config.use_peft:
                out_sim_mat = self.pairwise_cosine_sim(
                    encoder_outs, self.llm.base_model.model.model.embed_tokens.weight
                )
            else:
                out_sim_mat = self.pairwise_cosine_sim(
                    encoder_outs, self.llm.model.embed_tokens.weight
                )
            assert len(out_sim_mat) == len(
                kwargs["targets"]
            ), f"{len(out_sim_mat)}, {len(kwargs['targets'])}"

            for tmp_i in range(len(kwargs["targets"])):
                torch.save(
                    {
                        "cosine_sim": out_sim_mat[tmp_i][0 : input_lens[tmp_i], :],
                        "label_ids": self.tokenizer.encode(kwargs["targets"][tmp_i])[
                            1:
                        ],
                        "text": kwargs["targets"][tmp_i],
                        "key": kwargs["keys"][tmp_i],
                    },
                    str(
                        self.model_config.save_align_cosine_sim_matrix
                        / f"{kwargs['keys'][tmp_i]}.pt"
                    ),
                )

        # VQ - encoder_outs after projector
        if self.model_config.vq_encoder_out:
            # VQ
            # import pdb

            # pdb.set_trace()
            N, T, D = encoder_outs.shape
            encoder_outs_flat = encoder_outs.view(-1, D)
            hard_assign_vq = kwargs.get("inference_mode", False) or True
            #hard_assign_vq = kwargs.get("inference_mode", False) and False
            if self.train_config.use_peft:
                codebook = self.llm.base_model.model.model.embed_tokens.weight
            else:
                codebook = self.llm.model.embed_tokens.weight
            # encoder_outs_embed_logits = torch.matmul(encoder_outs_flat, codebook.t())
            # instead simple matmul, do cosine dist
            encoder_outs_embed_logits = torch.matmul(encoder_outs_flat, codebook.t())
            encoder_outs_embed_logits = encoder_outs_embed_logits / torch.clamp(
                encoder_outs_flat.norm(dim=-1).unsqueeze(-1), min=1e-8
            )
            encoder_outs_embed_logits = encoder_outs_embed_logits / torch.clamp(
                codebook.norm(dim=-1).unsqueeze(0), min=1e-8
            )
            soft_assignment = F.gumbel_softmax(
                encoder_outs_embed_logits, tau=1.0, hard=hard_assign_vq, dim=-1
            )
            encoder_outs_quantized = torch.matmul(soft_assignment, codebook)
            encoder_outs = encoder_outs_quantized.view(N, T, D)

            # passing average of frames
            # input_lens = (modality_mask == True).sum(dim=1)
            # input_lens = torch.sub(input_lens, 1)
            # encoder_outs_meaned = torch.full_like(encoder_outs, fill_value=0)
            # for tmp_utt_ind in range(encoder_outs.shape[0]):
            #     mean_vec_cur = torch.mean(
            #         encoder_outs[tmp_utt_ind, 0 : input_lens[tmp_utt_ind], :], dim=0
            #     )
            #     encoder_outs_meaned[tmp_utt_ind, ::2, :] = mean_vec_cur
            #     # encoder_outs[tmp_utt_ind, 0 : input_lens[tmp_utt_ind], :] = mean_vec_cur
            # encoder_outs = encoder_outs_meaned

        if (
            not kwargs.get("inference_mode", False)
        ) and self.model_config.ctc_weight > 0:
            encoder_outs_permuted = encoder_outs_llm.permute(
                1, 0, 2
            )  # (N, T, C) -> (T, N, C)
            input_lens = (modality_mask == True).sum(dim=1)
            input_lens = torch.sub(input_lens, 1)

            with torch.no_grad():
                references_raw = self.tokenizer(
                    kwargs["targets"]
                ).input_ids  # list[list[int]]
                target_lens = []
                references = []
                for tmp_utt_ind in range(len(references_raw)):
                    target_lens.append(len(references_raw[tmp_utt_ind]) - 1)
                    references.extend(references_raw[tmp_utt_ind][1:])

                target_lens = torch.LongTensor(target_lens).to(encoder_outs_llm.device)
                references = torch.LongTensor(references).to(encoder_outs_llm.device)

            loss_ctc = F.ctc_loss(
                encoder_outs_permuted,
                references,
                input_lens,
                target_lens,
                reduction="mean",
                zero_infinity=True,
            )
            loss_ctc = self.model_config.ctc_weight * loss_ctc

        if instruct_ids is not None:
            if self.encoder is not None:
                encoder_outs = self.encoder(
                    input_ids=instruct_ids, attention_mask=instruct_mask
                ).last_hidden_state

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, instruct_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if self.train_config.prompt_token_id:
                virtual_tokens_ixs = (input_ids == self.train_config.prompt_token_id).nonzero(as_tuple=True)
                input_ids[virtual_tokens_ixs] = 0
            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)
            if self.train_config.prompt_token_id:
                input_ids[virtual_tokens_ixs] = self.train_config.prompt_token_id

        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(
                modality_mask.sum(dim=1), max=encoder_outs.shape[1]
            ).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                encoder_outs_pad[
                    i,
                    modality_mask_start_indices[i] : modality_mask_start_indices[i]
                    + modality_lengths[i],
                ] = encoder_outs[i][: modality_lengths[i]]

            inputs_embeds = encoder_outs_pad + inputs_embeds * (
                ~modality_mask[:, :, None]
            )

        pptuning_enabled = self.train_config.use_peft and self.train_config.peft_config.peft_method.lower() in ["p-tuning", "p-projector"]

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        if zh_data is not None and en_data is not None:
            model_outputs, acc = self.llm(zh=zh_data, en=en_data)
        else:
            model_outputs = self.llm(
                input_ids=input_ids if pptuning_enabled else None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
            )

            acc = -1
            if self.metric:
                with torch.no_grad():
                    preds = torch.argmax(model_outputs.logits, -1)
                    if preds.shape[1] > labels.shape[1]:
                        preds = preds[:, preds.shape[1] - labels.shape[1]:]
                    acc = compute_accuracy(
                        preds.detach()[:, :-1],
                        labels.detach()[:, 1:],
                        ignore_label=-100,
                    )

        # return model_outputs, acc
        if self.model_config.ctc_weight > 0:
            return model_outputs, acc, loss_ctc
        return model_outputs, acc

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        kwargs["inference_mode"] = True
        pptuning_enabled = self.train_config.use_peft and self.train_config.peft_config.peft_method.lower() in ["p-tuning", "p-projector"]

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if pptuning_enabled:

            model_kwargs = self.llm.prepare_inputs_for_generation(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                max_new_tokens=kwargs.get("max_new_tokens", 200),
                num_beams=kwargs.get("num_beams", 4),
                do_sample=kwargs.get("do_sample", False),
                min_length=kwargs.get("min_length", 1),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                length_penalty=kwargs.get("length_penalty", 1.0),
                temperature=kwargs.get("temperature", 1.0),
                attention_mask=attention_mask,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            inputs_embeds = model_kwargs["inputs_embeds"]
            attention_mask = model_kwargs["attention_mask"]
            # position_ids = model_kwargs["position_ids"]  # will be generated automatically from `attention_mask` by model's `prepare_inputs_for_generation()` -- https://github.com/huggingface/transformers/blob/4d5b45870411053c9c72d24a8e1052e00fe62ad6/src/transformers/models/llama/modeling_llama.py#L1264

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            # position_ids=position_ids,
            # max_length=kwargs.get("max_length", 200),
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return model_outputs
