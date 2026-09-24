# SPDX-FileCopyrightText: Copyright (c) 2024 SLAM-LLM contributors
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute
#
# SPDX-License-Identifier: MIT
#
# Modifications by Idiap Research Institute (© 2025):
#   - Enhanced prompt parsing to support <speech>, <p:N>, and <p:TEXT> tokens
#   - Added flexible speech token placement anywhere in the prompt
#   - Added support for learnable token insertion and text-initialized learnable tokens
#   - Added the building blocks of the text-only adaptation method: noise functions,
#     multi-view batch composition and the corresponding sampling strategies

import re
import json
import copy
import torch
import logging
import whisper
import random
import numpy as np
import nlpaug.augmenter.char as nac # for text noise augmentation

logger = logging.getLogger(__name__)

TOKEN_SPEECH = "<speech>"


def noise_text(text: str,
            aug_char_p=0.30,
            aug_char_min=1,
            aug_char_max=10,
            aug_word_p=0.15,
            include_upper_case=True,
            include_lower_case=True,
            include_numeric=False,
            swap_mode = "random",
            dup_char_prob = 0.1,    # percentage of positions to duplicate
            dup_char_prob_max_times  = 3, # max times to duplicate selected characters
            ) -> str:
    if not text:
        logger.warning("Empty text received for noise_text function. Returning empty string.")
        return ""

    # TODO: We need to set the seed for reproducibility for NLPaug and random     
    # We create a nlpaug.augmenter object
    char_insert_aug = nac.RandomCharAug(       
            aug_char_p=aug_char_p,          
            aug_char_min=aug_char_min,
            aug_char_max=aug_char_max,
            aug_word_p=aug_word_p,           
            include_upper_case=include_upper_case,
            include_lower_case=include_lower_case,
            include_numeric=include_numeric,
            swap_mode=swap_mode,
            
            )
    # 1) insert random characters
    noisy = char_insert_aug.augment(text)

    if isinstance(noisy, list):   
        noisy = noisy[0]

    # 2) duplicate characters with probability _dup_char_prob
    out_chars = []
    for ch in noisy:
        out_chars.append(ch)
        if ch.isalpha() and random.random() < dup_char_prob:
            n_dup = random.randint(1, dup_char_prob_max_times)
            out_chars.append(ch * n_dup)

    return "".join(out_chars)


class SpeechDatasetJsonl(torch.utils.data.Dataset):

    def __init__(
        self, dataset_config, tokenizer=None, processor=None, split="train", llm_name="vicuna-7b-v1.5", model=None, noise_fn=noise_text, default_item_type=None
    ):
        # default_item_type is None, "audio", "noise_audio", "noise", "echo"
        global stoken_ix
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.processor = processor
        self.llm_name = llm_name
        # data_parallel_size = dist.get_world_size()
        data_parallel_size = 1

        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.AUDIO_TOKEN_ID = -1
        self.mel_size = dataset_config.get(
            "mel_size", 80
        )  # 80 for whisper large v1 and v2, 128 for large v3
        # self.prompt_library = [
        #     "Begin by converting the spoken words into written text. ",
        #     "Can you transcribe the speech into a written format? ",
        #     "Focus on translating the audible content into text. ",
        #     "Transcribe the speech by carefully listening to it. ",
        #     "Would you kindly write down the content of the speech? ",
        #     "Analyze the speech and create a written transcription. ",
        #     "Engage with the speech to produce a text-based version. ",
        #     "Can you document the speech in written form? ",
        #     "Transform the spoken words into text accurately. ",
        #     "How about putting the speech's content into writing? "
        # ]
        self.prompt = dataset_config.get("prompt", None)
        if self.prompt is None:
            # self.prompt = random.choice(self.prompt_library)
            # self.prompt = "Transcribe speech to text. "
            self.prompt = "Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. "
        # if "llama3" in llm_name:
        #     self.prompt_template = [
        #                             {"role": "system",
        #                             "content": "You are a helpful assistant, expertized in transcribing speech to text.",
        #                             },
        #                             {
        #                             "role": "user",
        #                             "content":"Transcribe speech to text: \n",
        #                             }
        #                         ]
        #     #self.prompt_template = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)  
        #     #self.prompt_template = "USER: {}\n ASSISTANT:"
        #     self.prompt = self.processor.apply_chat_template(self.prompt_template, add_generation_prompt=True, tokenize=False)
        # else:
        self.prompt_template = dataset_config.get("prompt_template", None)
        if self.prompt_template:
            self.prompt = self.prompt_template.format(prompt=self.prompt)
        # If p-prompt embeddings initialization...
        logger.info(f"Input prompt: {self.prompt}")
        p_token = dataset_config.prompt_token
        re_token = f"{p_token[:-1]}:.+?{p_token[-1]}"
        m = re.search(re_token, self.prompt, flags=re.IGNORECASE|re.DOTALL)
        if m:
            logger.info(f"  Embedding initialization detected.")
            prompt_new = []
            for piece in re.split(f"({re_token})", self.prompt, flags=re.IGNORECASE|re.DOTALL):
                if re.match(re_token, piece, flags=re.IGNORECASE|re.DOTALL):
                    init_text = re.match(f"{p_token[:-1]}:(.+?){p_token[-1]}", piece, flags=re.IGNORECASE|re.DOTALL).group(1)
                    input_ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]
                    logger.info(f"    -> Segment '{init_text}' initialized with embeddings for {input_ids} tokens.")
                    prompt_new.append(p_token * len(input_ids))
                else:
                    prompt_new.append(piece)
            self.prompt = "".join(prompt_new)
            logger.info(f"New input prompt: {self.prompt}")

        if dataset_config.prompt_embeddings_path:
            stoken_ix = 0
            def token_number(_):
                global stoken_ix
                stoken_ix = stoken_ix + 1
                return f"<p{stoken_ix - 1}>"
            self.prompt = re.sub(p_token, token_number, self.prompt)
            logger.info(f"Prompt embeddings path detected -> New input prompt: {self.prompt}")
            new_tokens = [f"<p{ix}>" for ix in range(stoken_ix)]
            for new_token in new_tokens:
                # one by one to force the right order, to match the indexes in the Embedding matrix
                tokenizer.add_special_tokens({'additional_special_tokens': [new_token]})
            logger.info(f"Adding new tokens to tokenizer: {new_tokens}")

        self.speech_token_ix = 0
        # If prompt_template contains <speech>, set the speech tokens flag and add token to tokenizer
        if TOKEN_SPEECH not in self.prompt:
            logger.info(f"No speech token ('{TOKEN_SPEECH}') was found in the input prompt. "
                        "Speech embeddings will be automatically prepend to the input.")
            self.prompt_ids = self.tokenizer.encode(self.prompt)
            self.speech_token_prefix = True
        else:
            tokenizer.add_special_tokens({'additional_special_tokens': [TOKEN_SPEECH]})
            speech_token_id = tokenizer.get_vocab()[TOKEN_SPEECH]
            self.prompt_ids = self.tokenizer.encode(self.prompt)
            self.speech_token_ix = self.prompt_ids.index(speech_token_id)
            del self.prompt_ids[self.speech_token_ix]
            self.speech_token_prefix = False
            logger.info(f"Speech token found ('{TOKEN_SPEECH}') at index {self.speech_token_ix}.")

        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.target_lowercase = dataset_config.get("target_lowercase", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]"

        self.data_list = []
        data_path = dataset_config.train_data_path if split == "train" else dataset_config.val_data_path
        with open(data_path, encoding="utf-8") as fin:
            for line in fin:
                data_dict = json.loads(line.strip())
                self.data_list.append(data_dict)

        self.default_item_type = default_item_type
        self.noise_fn = noise_fn
        self.model = model

    def noise_from_audio(self, audio) -> str:        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio = audio.unsqueeze(0).to(device)
        with torch.no_grad():    
            encoder_outs = self.model.module.encoder.extract_features(audio, None)
            projected_embeddings = self.model.module.encoder_projector(encoder_outs)

        noise_tokens = [torch.argmin(torch.matmul(self.model.module.llm.base_model.model.model.embed_tokens.weight, projected_embeddings[0,j].T))
                        for j in range(projected_embeddings.shape[1])]
        return self.tokenizer.decode(noise_tokens)

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):

        return data_dict["target_len"] if "target_len" in data_dict else 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index, item_type=None):
        item_type = item_type or self.default_item_type or "audio"
        data_dict = self.data_list[index]
        target = data_dict.get("target", None)
        key = data_dict.get("key", None)
        # Text-only items carry no audio; inference always loads it.
        audio_raw = None
        audio_mel = None
        audio_length = 0
        if self.inference_mode or "audio" in item_type:
            audio_path = data_dict.get("source")
            audio_raw = whisper.load_audio(audio_path, sr=16000)  ## sr added to make sure all is in 16KHz
            if self.input_type == "raw":
                audio_raw = torch.from_numpy(audio_raw)
                if self.normalize:
                    audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
                audio_length = len(audio_raw) // 320  # ad-hoc for fairseq 320x downsample
                audio_length = audio_length // 5  # ad-hoc for 5x fc downsample
            elif self.input_type == "mel":
                audio_raw = whisper.pad_or_trim(audio_raw)
                audio_mel = whisper.log_mel_spectrogram(
                    audio_raw, n_mels=self.mel_size
                ).permute(1, 0)
                audio_length = (
                    audio_mel.shape[0] + 1
                ) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
                audio_length = audio_length // 5  # ad-hoc for 5x fc downsample
            if self.fix_length_audio > 0:
                audio_length = self.fix_length_audio

        prompt_length = len(self.prompt_ids)

        if self.inference_mode:
            example_ids = self.prompt_ids[:self.speech_token_ix] + ([self.AUDIO_TOKEN_ID] * audio_length) + self.prompt_ids[self.speech_token_ix:]
            example_ids = torch.tensor(example_ids, dtype=torch.int64)
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
            }

        example_ids = self.tokenizer.encode(self.prompt + self.answer_template.format(target))  # [prompt,answer]
        if not self.speech_token_prefix:
            del example_ids[self.speech_token_ix]
        if item_type == "audio":
            audio_tokens_ids = [self.AUDIO_TOKEN_ID] * audio_length  # To be replaced by real audio (speech embeddings)
        else:
            if item_type == "noise" and self.noise_fn is not None:
                noise_target = self.noise_fn(target)  # Noised target as pseudo audio tokens
            elif item_type == "noise_audio":
                noise_target = self.noise_from_audio(audio_raw)  # Projector-based noised target from audio
            elif item_type == "echo":
                noise_target = target  # Echo the target as pseudo audio tokens
            else:
                raise ValueError(f"Unknown item_type: {item_type}")

            audio_tokens_ids = self.tokenizer.encode(noise_target, add_special_tokens=False)

            audio_length = 0
            audio_raw = None
            audio_mel = None
            prompt_length += len(audio_tokens_ids)

        example_ids = example_ids[:self.speech_token_ix] + audio_tokens_ids + example_ids[self.speech_token_ix:] + [self.tokenizer.eos_token_id]
        example_ids = torch.tensor(example_ids, dtype=torch.int64)

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[: audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
            "target": target,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (
                        sequence,
                        torch.full(
                            ([max_length - len(sequence)] + list(sequence.size())[1:]),
                            padding_idx,
                        ),
                    )
                )
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (
                        sequence,
                        np.full(
                            (max_length - len(sequence),) + sequence.shape[1:],
                            padding_idx,
                        ),
                    )
                )
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat(
                            (
                                torch.full(
                                    ([padding_length] + list(sequence.size())[1:]),
                                    padding_idx,
                                ),
                                sequence,
                            )
                        )
                    else:
                        sequence = torch.cat(
                            (
                                sequence,
                                torch.full(
                                    ([padding_length] + list(sequence.size())[1:]),
                                    padding_idx,
                                ),
                            )
                        )
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (
                        sequence,
                        np.full((padding_length,) + sequence.shape[1:], padding_idx),
                    )
                )
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None
        input_prompt_lengths = [
            s["audio_length"] + s["prompt_length"] for s in samples
        ]  # [120, 48, 82, 42]
        input_answer_lengths = [
            len(s["input_ids"]) - s["audio_length"] - s["prompt_length"]
            for s in samples
        ]  # [0, 0, 0, 0]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)

        input_ids = torch.stack(
            [
                self.padding(
                    self.padding(
                        samples[index]["input_ids"],
                        input_prompt_max_length - input_prompt_lengths[index],
                        self.tokenizer.pad_token_id,
                        padding_side="left",
                    ),
                    input_answer_max_length - input_answer_lengths[index],
                    self.tokenizer.pad_token_id,
                )
                for index in range(len(samples))
            ]
        )

        attention_mask = torch.stack(
            [
                self.padding(
                    self.padding(
                        samples[index]["attention_mask"],
                        input_prompt_max_length - input_prompt_lengths[index],
                        False,
                        padding_side="left",
                    ),
                    input_answer_max_length - input_answer_lengths[index],
                    False,
                )
                for index in range(len(samples))
            ]
        )

        if self.input_type == "raw":
            # Filter out None audio samples and get max length
            valid_audio_shapes = [s["audio"].shape[0] for s in samples if s["audio"] is not None]
            audio_raw_max_length = max(valid_audio_shapes) if valid_audio_shapes else 1
            # audio_missing_flag = [s["audio"] is None for s in samples] # Added to control when there is no audio input in the batch

            audio_raw_list = []
            for s in samples:
                if s["audio"] is not None:
                    audio_raw_list.append(self.pad(s["audio"], audio_raw_max_length, 0))
                else:
                    # Create zero tensor for missing audio
                    audio_raw_list.append(torch.zeros(audio_raw_max_length))

            audio_raw = torch.stack(audio_raw_list)
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                if sample["audio"] is not None:
                    audio_mask[line, : sample["audio"].shape[0]] = 1
        elif self.input_type == "mel":
            # Filter out None audio_mel samples and get max length
            valid_audio_mel_shapes = [s["audio_mel"].shape[0] for s in samples if s["audio_mel"] is not None]
            audio_mel_max_length = max(valid_audio_mel_shapes) if valid_audio_mel_shapes else 1
            
            audio_mel_list = []
            for s in samples:
                if s["audio_mel"] is not None:
                    audio_mel_list.append(self.pad(s["audio_mel"], audio_mel_max_length, 0))
                else:
                    # Create zero tensor for missing audio_mel, preserve original shape dimensions
                    if valid_audio_mel_shapes:
                        # Use the same number of features as valid samples
                        sample_audio_mel = next(s["audio_mel"] for s in samples if s["audio_mel"] is not None)
                        feature_dim = sample_audio_mel.shape[1] if len(sample_audio_mel.shape) > 1 else 1
                        audio_mel_list.append(torch.zeros(audio_mel_max_length, feature_dim))
                    else:
                        # All samples have None audio_mel, create minimal tensor
                        audio_mel_list.append(torch.zeros(audio_mel_max_length, self.mel_size))
            
            audio_mel = torch.stack(audio_mel_list)
            audio_mel_post_mask = torch.zeros(
                len(samples), (audio_mel_max_length + 1) // 2
            )  # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                if sample["audio_mel"] is not None:
                    audio_mel_post_mask[line, : (sample["audio_mel"].shape[0] + 1) // 2] = 1
        else:
            raise ValueError("input_type must be one of [raw, mel]")

        # later, in the forward pass, embeddings masked by this mask will be replaced by speech embeddings
        modality_mask = input_ids == self.AUDIO_TOKEN_ID

        position_ids = None
        if attention_mask is not None:
            # To use the right position embedding when left padding is used
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        if self.inference_mode:
            keys = [s["key"] for s in samples]
            targets = [s["target"] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": (
                    audio_mel_post_mask if self.input_type == "mel" else None
                ),
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets,
            }

        labels = torch.stack(
            [
                self.padding(
                    self.padding(
                        samples[index]["labels"],
                        input_prompt_max_length - input_prompt_lengths[index],
                        self.IGNORE_INDEX,
                        padding_side="left",
                    ),
                    input_answer_max_length - input_answer_lengths[index],
                    self.IGNORE_INDEX,
                )
                for index in range(len(samples))
            ]
        )
        targets = [s["target"] for s in samples]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": (
                audio_mel_post_mask if self.input_type == "mel" else None
            ),
            "modality_mask": modality_mask,
            "targets": targets,
        }

    def merge_batch(self, batch1, batch2):
        """
        Merge two batches by concatenating tensors and applying proper padding.
        Follows the same padding logic as the collator method.
        
        Args:
            batch1: First batch (main dataloader batch)
            batch2: Second batch (auxiliary dataloader batch)
            
        Returns:
            merged_batch: Combined batch with proper padding
        """
        merged_batch = {}
        
        # Handle each key in the batch
        for key in batch1.keys():
            if key not in batch2:
                merged_batch[key] = batch1[key]
                continue
                
            value1 = batch1[key]
            value2 = batch2[key]

            # Handle None values first
            if value1 is None:
                merged_batch[key] = value2
            elif value2 is None:
                merged_batch[key] = value1
            elif isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
                if key in ["input_ids", "labels", "attention_mask", "position_ids", "modality_mask"]:
                    # These use left padding for prompt/audio part, right padding for answer part
                    merged_batch[key] = self._merge_sequence_tensors(value1, value2, key)
                else:
                    # All other tensors (audio, audio_mel, audio_mask, audio_mel_post_mask, etc.)
                    # use right padding (standard pad method)
                    merged_batch[key] = self._merge_right_padded_tensors(value1, value2)
            elif isinstance(value1, list) and isinstance(value2, list):
                # Concatenate lists (e.g., targets)
                merged_batch[key] = value1 + value2
            elif hasattr(value1, '__iter__') and hasattr(value2, '__iter__'):
                merged_batch[key] = list(value1) + list(value2)
            else:
                raise ValueError(f"Unsupported data type for key '{key}': {type(value1)} and {type(value2)}")

        return merged_batch

    def _merge_sequence_tensors(self, tensor1, tensor2, key):
        """Merge sequence tensors (input_ids, labels, attention_mask, position_ids) with proper padding"""
        # Get appropriate padding value
        if key == "input_ids":
            pad_value = self.tokenizer.pad_token_id
        elif key == "labels":
            pad_value = self.IGNORE_INDEX
        elif key == "attention_mask":
            pad_value = False
        elif key == "position_ids":
            pad_value = 1
        elif key == "modality_mask":
            pad_value = False
        else:
            pad_value = 0
            
        # Find max length and pad both tensors
        max_length = max(tensor1.shape[1], tensor2.shape[1])

        # Pad tensor1 with left padding (like in collator)
        if tensor1.shape[1] < max_length:
            pad_size = max_length - tensor1.shape[1]
            # tensor1 = self.padding(tensor1, pad_size, pad_value, padding_side="left")
            tensor1 = torch.stack(
                [
                    self.padding(tensor1[index], pad_size, pad_value, padding_side="left")
                    for index in range(tensor1.shape[0])
                ]
            )

        # Pad tensor2 with left padding
        if tensor2.shape[1] < max_length:
            pad_size = max_length - tensor2.shape[1]
            # tensor2 = self.padding(tensor2, pad_size, pad_value, padding_side="left")
            tensor2 = torch.stack(
                [
                    self.padding(tensor2[index], pad_size, pad_value, padding_side="left")
                    for index in range(tensor2.shape[0])
                ]
            )

        # Concatenate along batch dimension
        return torch.cat([tensor1, tensor2], dim=0)

    def _merge_right_padded_tensors(self, tensor1, tensor2):
        """Merge tensors with right padding (audio, audio_mel, audio_mask, etc.)"""
        # Find max length in the time/sequence dimension (dim 1)
        max_length = max(tensor1.shape[1], tensor2.shape[1])
        # Create new tensors to hold padded results
        tensor1_padded = torch.zeros(tensor1.shape[0], max_length, dtype=tensor1.dtype)
        tensor2_padded = torch.zeros(tensor2.shape[0], max_length, dtype=tensor2.dtype)
        # Pad both tensors to max length using right padding with zeros
        for index in range(tensor1.shape[0]):
            tensor1_padded[index] = self.pad(tensor1[index], max_length, 0)
        for index in range(tensor2.shape[0]):
            tensor2_padded[index] = self.pad(tensor2[index], max_length, 0)
        # tensor1_padded = self.pad(tensor1, max_length, 0)
        # tensor2_padded = self.pad(tensor2, max_length, 0)

        # Concatenate along batch dimension
        # return torch.cat([tensor1_padded, tensor2_padded], dim=0)
        return torch.cat([tensor1_padded, tensor2_padded], dim=0)


class SpeechAuxDatasetJsonl(SpeechDatasetJsonl):
    def __init__(self, dataset_config, tokenizer=None, model=None, processor=None, split="train", llm_name="vicuna-7b-v1.5", noise_fn=noise_text):

        logger.info("Initializing SpeechAuxDatasetJsonl with text-only adaptation config.")

        train_path = dataset_config.train_data_path
        dataset_config.train_data_path = dataset_config.train_aux_data_path
        super().__init__(dataset_config, tokenizer, processor, split, llm_name, model=model, noise_fn=noise_fn)
        dataset_config.train_data_path = train_path  # restore original path, just in case

        self.model = model
        self.noise_fn = noise_fn
        self.sigma_a = dataset_config.get("text_only_adaptation_config", {}).get("sigma_a", None)
        self.sigma_ta = dataset_config.get("text_only_adaptation_config", {}).get("sigma_ta", None)
        self.sigma_t = dataset_config.get("text_only_adaptation_config", {}).get("sigma_t", None)
        tau_t = dataset_config.get("text_only_adaptation_config", {}).get("tau_t", 0.5)
        tau_a = dataset_config.get("text_only_adaptation_config", {}).get("tau_a", 0)
        tau = tau_t + tau_a

        # Simple check
        if not (0 <= tau <= 1):
            raise ValueError(f"Sum of taus must be in [0,1], got tau_t={tau_t}, tau_a={tau_a}, sum={tau}")

        sigmas = {
            "sigma_a": self.sigma_a,
            "sigma_ta": self.sigma_ta,
            "sigma_t": self.sigma_t,
        }

        # Sum of provided (non-None) sigmas
        known_sum = sum(v for v in sigmas.values() if v is not None)
        none_keys = [k for k, v in sigmas.items() if v is None]

        if tau + known_sum > 1:
            raise ValueError(f"tau + provided sigmas exceed 1: tau={tau}, known_sigma_sum={known_sum}, total={tau + known_sum}")

        remaining = 1 - tau - known_sum

        if none_keys:
            # Distribute remaining mass uniformly among missing sigmas
            for k in none_keys:
                sigmas[k] = remaining / len(none_keys)

        # Assign resolved sigmas back
        self.sigma_a = sigmas["sigma_a"]
        self.sigma_ta = sigmas["sigma_ta"]
        self.sigma_t = sigmas["sigma_t"]

        # Log final values
        logger.info(
            f"Text adaptation overall batch proportions: tau_t={tau_t:.3f}, tau_a={tau_a:.3f}, "
            f"sigma_a={self.sigma_a:.3f}, sigma_ta={self.sigma_ta:.3f}, sigma_t={self.sigma_t:.3f}"
        )

        # Scale sigmas so they sum to 1 independently of tau (relative proportions)
        sigma_sum_no_tau = self.sigma_a + self.sigma_ta + self.sigma_t
        if sigma_sum_no_tau == 0:
            self.sigma_a = 0
            self.sigma_ta = 0
            self.sigma_t = 0
        else:
            self.sigma_a /= sigma_sum_no_tau
            self.sigma_ta /= sigma_sum_no_tau
            self.sigma_t /= sigma_sum_no_tau

    def __getitem__(self, index):
        # Monte Carlo sampling among item types according to sigma_a, sigma_ta, sigma_t
        # sigma_a -> "audio", sigma_ta -> "noise_audio", sigma_t -> "noise"
        r = np.random.rand()
        if r < self.sigma_a:
            item_type = "audio"
        elif r < self.sigma_a + self.sigma_ta:
            item_type = "noise_audio"
        else:
            item_type = "noise"
        return super().__getitem__(index, item_type=item_type)


class SpeechAndTextDatasetJsonl(SpeechDatasetJsonl):
    def __init__(self, dataset_config, tokenizer=None, model=None, processor=None, split="train", llm_name="vicuna-7b-v1.5", noise_fn=noise_text):

        logger.info("Initializing SpeechAndTextDatasetJsonl with text-only adaptation config.")
        super().__init__(dataset_config, tokenizer, processor, split, llm_name, model=model, noise_fn=noise_fn)

        self.model = model
        self.noise_fn = noise_fn
        self.tau_t = dataset_config.get("text_only_adaptation_config", {}).get("tau_t", 0.5)
        self.tau_a = dataset_config.get("text_only_adaptation_config", {}).get("tau_a", 0)

        logger.info(f"Target domain in-batch proportions: tau_t={self.tau_t}, tau_a={self.tau_a}")

        # Simple check
        tau_sum = self.tau_t + self.tau_a
        if not (0 <= tau_sum <= 1):
            raise ValueError(f"Sum of taus must be in [0,1], got tau_t={self.tau_t}, tau_a={self.tau_a}, sum={tau_sum}")

        # Scale tau so their in relative proportions to their sum
        self.tau_a /= tau_sum
        self.tau_t /= tau_sum

    def __getitem__(self, index):
        # Monte Carlo sampling among item types according to tau_t, tau_a
        # tau_t -> "noise" (text), tau_a -> "audio"
        r = np.random.rand()
        if r < self.tau_a:
            item_type = "audio"
        else:
            item_type = "noise"
        return super().__getitem__(index, item_type=item_type)


def get_speech_dataset(dataset_config, tokenizer, processor, split, llm_name="vicuna-7b-v1.5", model=None, auxiliary=False):
    if dataset_config.use_text_only_adaptation and split == dataset_config.train_split:
        noise_type = dataset_config.get("text_only_adaptation_config", {}).get("noise_type", "naive")
        # "naive", "echo", "empty"
        if noise_type.lower() == "naive":
            noise_fn = noise_text
        elif noise_type.lower() == "echo":
            noise_fn = lambda x: x  # Echo the target as pseudo audio tokens
        elif noise_type.lower() == "empty":
            noise_fn = lambda _: ""  # Empty string as pseudo audio tokens
        elif noise_type.lower() == "random":
            noise_fn = lambda x: "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "), size=len(x)))  # random sequence of characters of same length as target
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Expected one of ['naive', 'random', 'echo', 'empty']")

        logger.info(f"Using text-only adaptation with noise_type='{noise_type}' for training split.")

        if auxiliary:
            return SpeechAuxDatasetJsonl(
                dataset_config, tokenizer, model, processor, split, llm_name=llm_name, noise_fn=noise_fn
            )
        return SpeechAndTextDatasetJsonl(
            dataset_config, tokenizer, model, processor, split, llm_name=llm_name, noise_fn=noise_fn
        )

    return SpeechDatasetJsonl(dataset_config, tokenizer, processor, split, llm_name)
