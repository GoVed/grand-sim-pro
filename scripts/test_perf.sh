#!/bin/bash
# Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
# Part of an independent research project into emergent biological complexity.
#
# Copyright (C) 2026 Ved Hirenkumar Suthar
# Licensed under the GNU General Public License v3.0 or later.
# * This software is provided "as is", without warranty of any kind.
# See the LICENSE file in the project root for full license details.

# Run performance tests in release mode and maintain a log of the last 5 runs.

set -e

PERF_LOG="PERFORMANCE_LOG.md"

# Ensure log exists with header
if [ ! -f "$PERF_LOG" ]; then
    echo "| Timestamp | 10k Repro Time | 100 Sim Steps (10k agents) |" > "$PERF_LOG"
    echo "|-----------|----------------|----------------------------|" >> "$PERF_LOG"
fi

# Run tests and capture output
echo "Running performance tests in release mode..."
OUTPUT=$(cargo test --release --test performance -- --nocapture 2>&1)

# Extract values (using simple grep/awk)
REPRO_TIME=$(echo "$OUTPUT" | grep "10000 reproductions in" | awk '{print $4}')
SIM_TIME=$(echo "$OUTPUT" | grep "Processed 500 steps with 10000 agents in" | awk '{print $8}')
WORLD_TIME=$(echo "$OUTPUT" | grep "World generation (800x600) took" | awk '{print $5}')
SORT_TIME=$(echo "$OUTPUT" | grep "High Density: 100 Optimized Spatial Sorts" | awk '{print $11}')

# Log current run
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
NEW_ENTRY="| $TIMESTAMP | $REPRO_TIME | $SIM_TIME | $WORLD_TIME | $SORT_TIME |"

# Read current entries (excluding header)
# Use tail to skip header (2 lines)
EXISTING_ENTRIES=$(tail -n +3 "$PERF_LOG")

# Prepare new file content with up to 4 old entries + 1 new
echo "| Timestamp | 10k Repro | 500 Sim Steps | World Gen | 100 Sorts (20k) |" > "${PERF_LOG}.tmp"
echo "|-----------|-----------|---------------|-----------|-----------------|" >> "${PERF_LOG}.tmp"
# Append existing entries, limited to 4
echo "$EXISTING_ENTRIES" | tail -n 4 >> "${PERF_LOG}.tmp"
# Append new entry
echo "$NEW_ENTRY" >> "${PERF_LOG}.tmp"

# Cleanup any empty lines if EXISTING_ENTRIES was empty
sed -i '/^$/d' "${PERF_LOG}.tmp"

mv "${PERF_LOG}.tmp" "$PERF_LOG"

echo "Performance logged in $PERF_LOG"
echo "Latest: $NEW_ENTRY"
