# SPDX-FileCopyrightText: Copyright (c) 2024 SLAM-LLM contributors
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Modifications by Idiap Research Institute (© 2025):
#   - Implemented freeze_projector functionality to freeze the projector module during training
#   - Restricted setup_processor to Llama decoders (the prompt template is defined manually)

import torch
import os
import logging
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_processor,
    setup_encoder,
    setup_encoder_projector,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size
from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.train_utils import print_model_size
from peft import PeftModel, get_peft_model

logger = logging.getLogger(__name__)

DTYPES = {"F32": torch.float32}
import json
import mmap
def load_safetensors_file(filename):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)
    size = os.stat(filename).st_size
    storage = torch.ByteStorage.from_file(filename, shared=False, size=size).untyped()
    offset = n + 8
    return {name: create_tensor(storage, info, offset) for name, info in metadata.items() if name != "__metadata__"}
def create_tensor(storage, info, offset):
    dtype = DTYPES[info["dtype"]]
    shape = info["shape"]
    start, stop = info["data_offsets"]
    return torch.asarray(storage[start + offset : stop + offset], dtype=torch.uint8).view(dtype=dtype).reshape(shape)

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    # If P-Tunning add prompt token...
    if train_config.use_peft and train_config.peft_config.peft_method.lower() in ["p-tuning", "p-projector"] and not train_config.prompt_num_virtual_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [train_config.prompt_token]})
        train_config.prompt_token_id = tokenizer.get_vocab()[train_config.prompt_token]
        logger.info(f"Adding prompt token to the tokenizer, prompt token id is {train_config.prompt_token_id}")

    processor=None
    # Processor added for llama3
    if model_config.llm_name == "llama":
        processor = setup_processor(model_config, **kwargs)

    # Speech Encoder
    encoder = setup_encoder(train_config, model_config, **kwargs)

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    if train_config.prompt_embeddings_path:
        logger.info(f"Embeddings: {llm.model.embed_tokens.weight.data.shape}")
        logger.info(f"Embeddings num: {llm.model.embed_tokens.num_embeddings}")

        learned_embs = load_safetensors_file(train_config.prompt_embeddings_path)["prompt_embeddings"]
        logger.info(f"Learned Embeddings {learned_embs.shape}")
        llm.model.embed_tokens = torch.nn.Embedding.from_pretrained(
            torch.cat([llm.model.embed_tokens.weight.data, learned_embs.data]),
            freeze=False
        )
        logger.info(f"New Embeddings: {llm.model.embed_tokens.weight.data.shape}")
        logger.info(f"New Embeddings num: {llm.model.embed_tokens.num_embeddings}")

    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )
    model = slam_model_asr(
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
    )
    if ckpt_path is not None:
        logger.info("Loading Speech Projector from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    if kwargs.get(
        "peft_ckpt", None
    ):
        logger.info("Loading PEFT_CKPT from: {}".format(kwargs.get("peft_ckpt")))
        model.llm = PeftModel.from_pretrained(
            model=model.llm, model_id=kwargs.get("peft_ckpt"), is_trainable=train_config.peft_config.peft_method.lower() not in ["p-tuning", "p-projector"]
        )
        model.llm.print_trainable_parameters()
        logger.info(f"PEFT model loaded with config: {model.llm.active_peft_config}")
        logger.info(f"PEFT is prompt learning? {model.llm.active_peft_config.is_prompt_learning}")
    elif train_config.use_peft:
        logger.info("Setup PEFT...")
        peft_config = generate_peft_config(train_config, data_config=kwargs["dataset_config"], model=model)
        model.llm = get_peft_model(model.llm, peft_config)
        logger.info("Number of Trainable parameters in the PEFT model:")
        model.llm.print_trainable_parameters()

    if train_config.freeze_projector:
        logger.info(f"Frozen Speech Projector. --> ENABLED")
        for _, param in model.encoder_projector.named_parameters():
                param.requires_grad = False

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


class slam_model_asr(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )


    @torch.no_grad()
    def inference(
        self,
        wav_path=None,
        prompt=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        # inference for asr model

        device = kwargs.get("device", "cuda")
        if os.path.exists(wav_path):  # Audio-Text QA
            import whisper

            audio_raw = whisper.load_audio(wav_path)
            audio_raw = whisper.pad_or_trim(audio_raw)

            mel_size = getattr(
                self.dataset_config, "mel_size", 80
            )  # 80 for large v1 and v2, 128 for large v3
            audio_mel = (
                whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
                .permute(1, 0)[None, :, :]
                .to(device)
            )

            encoder_outs = self.encoder.extract_variable_length_features(
                audio_mel.permute(0, 2, 1)
            )

            if self.model_config.encoder_projector == "q-former":
                audio_mel_post_mask = torch.ones(
                    encoder_outs.size()[:-1], dtype=torch.long
                ).to(encoder_outs.device)
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
        else:  # Text QA
            encoder_outs = torch.empty(
                1, 0, self.llm.model.embed_tokens.embedding_dim
            ).to(device)

        prompt = "USER: {}\n ASSISTANT:".format(prompt)

        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)

        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)

        inputs_embeds = torch.cat(
            (encoder_outs, inputs_embeds[None, :, :]), dim=1
        )  # [audio,prompt]

        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
            inputs_embeds.device
        )

        # generate
        model_outputs = self.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

        return model_outputs