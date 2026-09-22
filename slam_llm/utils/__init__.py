# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import re

from slam_llm.utils.memory_utils import MemoryTrace
from slam_llm.utils.dataset_utils import *
from slam_llm.utils.fsdp_utils import fsdp_auto_wrap_policy
from slam_llm.utils.train_utils import *

def expand_tokens(text, prompt_token):
    if not text:
        return text
    return re.sub(f"{prompt_token[:-1]}:\d+{prompt_token[-1]}", lambda m: prompt_token * int(m.group(0).split(":")[1][:-1]), text)
