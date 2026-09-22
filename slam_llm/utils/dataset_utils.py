# SPDX-FileCopyrightText: Copyright (c) 2024 SLAM-LLM contributors
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Modifications by Idiap Research Institute (© 2025):
#   - Added text-only adaptation support (auxiliary dataset and model passthrough)

import importlib
from functools import partial
from pathlib import Path

import torch

import logging
logger = logging.getLogger(__name__)


def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, processor, split: str, llm_name: str, model=None, auxiliary: bool = False):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, processor, split, llm_name, model=model, auxiliary=auxiliary)
    except AttributeError as e:
        logger.info(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e


def get_preprocessed_dataset(
    tokenizer, processor, dataset_config, split: str = "train", llm_name: str = "vicuna-7b-v1.5", model=None, auxiliary: bool = False
) -> torch.utils.data.Dataset:

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    return get_custom_dataset(
        dataset_config,
        tokenizer,
        processor,
        get_split(),
        llm_name,
        model=model,
        auxiliary=auxiliary,
    )
