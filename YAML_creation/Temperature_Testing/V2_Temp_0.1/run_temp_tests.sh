#!/bin/bash

set -u

MODEL="GPT-5.3-Instant"
TEMP="0.1"
STEPS="10"
TIMEOUT_SEC="600"

PROMPT="Read the YAML file in the current directory. Create files and execute necessary tasks as described in the prompt."

RESULTS_CSV="../temp_testing.csv"

# Create the CSV header if it doesn't exist
if [ ! -f "$RESULTS_CSV" ]; then
    echo "run_id,model,temp,steps,timeout_sec,elapsed_sec,timed_out,total_tokens,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,xml_exists,xml_file,xml_compile_pass" > "$RESULTS_CSV"
fi

for TEST_DIR in test*/; do
    TEST_DIR="${TEST_DIR%/}"
    echo "Running $TEST_DIR"

    cd "$TEST_DIR" || exit 1

    # Clean previous outputs
    rm -f opencode_events.jsonl opencode_timing.log opencode_stderr.log

    /usr/bin/time \
        -f "elapsed_sec=%e" \
        -o opencode_timing.log \
        timeout "$TIMEOUT_SEC" \
        opencode run --format json "$PROMPT" \
        > opencode_events.jsonl \
        2> opencode_stderr.log

    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 124 ]; then
        TIMED_OUT="TRUE"
    else
        TIMED_OUT="FALSE"
    fi

    ELAPSED=$(grep "elapsed_sec=" opencode_timing.log | cut -d= -f2)

    TOTAL_TOKENS=$(jq -s '[.[] | select(.type=="step_finish") | .part.tokens.total] | add // 0' opencode_events.jsonl)
    INPUT_TOKENS=$(jq -s '[.[] | select(.type=="step_finish") | .part.tokens.input] | add // 0' opencode_events.jsonl)
    OUTPUT_TOKENS=$(jq -s '[.[] | select(.type=="step_finish") | .part.tokens.output] | add // 0' opencode_events.jsonl)
    CACHE_READ=$(jq -s '[.[] | select(.type=="step_finish") | .part.tokens.cache.read] | add // 0' opencode_events.jsonl)
    CACHE_WRITE=$(jq -s '[.[] | select(.type=="step_finish") | .part.tokens.cache.write] | add // 0' opencode_events.jsonl)

    XML_FILE=$(find . -maxdepth 1 -name "*.xml" | head -n 1)

    if [ -n "$XML_FILE" ]; then
        XML_EXISTS="TRUE"
    else
        XML_EXISTS="FALSE"
        XML_FILE=""
    fi

    # XML compile check
    if [ "$XML_EXISTS" = "TRUE" ]; then
        XML_COMPILE_PASS=$(python3 - "$XML_FILE" <<'PYEOF'
import mujoco
import sys

xml_file = sys.argv[1]

try:
    mujoco.MjModel.from_xml_path(xml_file)
    print("TRUE")
except Exception:
    print("FALSE")
PYEOF
)
    else
        XML_COMPILE_PASS="FALSE"
    fi

    echo "$TEST_DIR,$MODEL,$TEMP,$STEPS,$TIMEOUT_SEC,$ELAPSED,$TIMED_OUT,$TOTAL_TOKENS,$INPUT_TOKENS,$OUTPUT_TOKENS,$CACHE_READ,$CACHE_WRITE,$XML_EXISTS,$XML_FILE,$XML_COMPILE_PASS" >> "$RESULTS_CSV"

    cd ..
done