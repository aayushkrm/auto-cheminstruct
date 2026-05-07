#!/bin/bash
set -e

# Accumulate pairs across multiple sessions to reach 100+
TARGET=100
TOTAL=0
SESSION=0
MAX_SESSIONS=10

# Clear old data
rm -rf .checkpoints/*.db datasets/autochem-* benchmarks/chemcot_final.json
> logs/scale_accumulate.log

echo "=== Auto-ChemInstruct Scale Run ==="
echo "Target: $TARGET pairs"
echo "Start: $(date)"

while [ $TOTAL -lt $TARGET ] && [ $SESSION -lt $MAX_SESSIONS ]; do
  SESSION=$((SESSION + 1))
  echo ""
  echo "--- Session $SESSION (accumulated: $TOTAL/$TARGET pairs) ---"
  
  # Run pipeline with small batch for reliability
  uv run python -m src.cli.main pipeline -n 20 -b 5 -B 2 >> logs/scale_accumulate.log 2>&1 || true
  
  # Count total pairs across all datasets
  TOTAL=0
  for f in datasets/autochem-*/train.jsonl datasets/autochem-*/test.jsonl datasets/autochem-*/val.jsonl; do
    if [ -f "$f" ]; then
      COUNT=$(wc -l < "$f" 2>/dev/null || echo 0)
      TOTAL=$((TOTAL + COUNT))
    fi
  done
  
  echo "Session $SESSION complete: $TOTAL pairs accumulated"
  sleep 2
done

echo ""
echo "=== Scale Run Complete ==="
echo "Total pairs: $TOTAL"
echo "Sessions: $SESSION"
echo "End: $(date)"

# Run final quality analysis
LATEST_DATASET=$(ls -d datasets/autochem-* 2>/dev/null | tail -1)
if [ -n "$LATEST_DATASET" ]; then
  uv run python -m src.cli.main chemcot -d "$LATEST_DATASET" -o benchmarks/chemcot_final.json
fi
