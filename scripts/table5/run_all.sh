#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Entry point for table5. Runs the full pipeline for this table.
export TABLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TABLE_DIR/../common/run_all.sh" "$@"
