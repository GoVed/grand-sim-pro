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

# Run tests and capture output
echo "Running Stress Performance Tests in release mode..."
OUTPUT=$(cargo test --release --test performance -- --nocapture 2>&1)

# Extract values from new stress test
COMPUTE_TIME=$(echo "$OUTPUT" | grep "Compute 100 ticks" | awk '{print $5}')
TPS=$(echo "$OUTPUT" | grep "Theoretical TPS:" | awk '{print $3}')
FETCH_TIME=$(echo "$OUTPUT" | grep "Agent State Fetch" | awk '{print $5}')
CELL_TIME=$(echo "$OUTPUT" | grep "Full Map Cell Fetch" | awk '{print $7}')
SORT_TIME=$(echo "$OUTPUT" | grep "Parallel Spatial Sort" | awk '{print $6}')

# Log current run
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
NEW_ENTRY="| $TIMESTAMP | $TPS | $COMPUTE_TIME | $FETCH_TIME | $CELL_TIME | $SORT_TIME |"

# Read current entries (excluding header)
if [ -f "$PERF_LOG" ]; then
    EXISTING_ENTRIES=$(tail -n +3 "$PERF_LOG")
else
    EXISTING_ENTRIES=""
fi

# Prepare new file content
echo "| Timestamp | TPS (Stress) | 100 Ticks | 20k Fetch | 10M Cell Fetch | 20k Sort |" > "${PERF_LOG}.tmp"
echo "|-----------|--------------|-----------|-----------|----------------|----------|" >> "${PERF_LOG}.tmp"
echo "$EXISTING_ENTRIES" | tail -n 4 >> "${PERF_LOG}.tmp"
echo "$NEW_ENTRY" >> "${PERF_LOG}.tmp"

sed -i '/^$/d' "${PERF_LOG}.tmp"
mv "${PERF_LOG}.tmp" "$PERF_LOG"

echo "Performance logged in $PERF_LOG"
echo "Latest: $NEW_ENTRY"
