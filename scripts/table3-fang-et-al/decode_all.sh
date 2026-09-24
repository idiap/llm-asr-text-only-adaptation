#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute
#
# SPDX-License-Identifier: MIT
#
# Decode every early checkpoint saved for table3-fang-et-al (used to pick the pre-forgetting model).
export TABLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TABLE_DIR/../common/decode_all.sh" "$@"
