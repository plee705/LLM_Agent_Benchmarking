#!/bin/bash

set -u

MODEL="unknown"
STEPS="10"
TIMEOUT_SEC="150"
RESULTS_CSV="temp_testing.csv"

PROMPT="Read the YAML file in the current directory. Create files and execute necessary tasks as described in the prompt."

if [ ! -f "$RESULTS_CSV" ]; then
    echo "run_id,model,temp,steps,timeout_sec,elapsed_sec,timed_out,total_tokens,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,xml_exists,xml_file,xml_compile_pass" > "$RESULTS_CSV"
fi

for CONFIG_DIR in V2_Temp_0.1/ V3_Temp_0.4/ V4_Temp_0.7/ V5_Temp_1.0/ V6_Temp_0.0/ V7_Temp_0.2/ V8_Temp_0.3/ V9_Temp_0.5/; do
    CONFIG_DIR="${CONFIG_DIR%/}"

    if [ ! -d "$CONFIG_DIR" ]; then
        continue
    fi

    TEMP=$(echo "$CONFIG_DIR" | sed -n 's/.*Temp_\([0-9.]*\).*/\1/p')
    if [ -z "$TEMP" ]; then
        TEMP="default"
    fi

    for TEST_NUM in $(seq -w 21 30); do
        TEST_DIR="$CONFIG_DIR/test$TEST_NUM"

        if [ ! -d "$TEST_DIR" ]; then
            mkdir -p "$TEST_DIR"
        fi

        # Seed new test folders with the standard YAML prompt if it is missing.
        if ! compgen -G "$TEST_DIR/*_Pend.YAML" > /dev/null; then
            SOURCE_YAML=$(find "$CONFIG_DIR/test01" -maxdepth 1 -name "*_Pend.YAML" | head -n 1)
            if [ -n "$SOURCE_YAML" ]; then
                cp "$SOURCE_YAML" "$TEST_DIR/"
            fi
        fi

        # Ensure each new test folder has the temperature-specific opencode config.
        if [ ! -f "$TEST_DIR/opencode.json" ]; then
            cp "$CONFIG_DIR/opencode.json" "$TEST_DIR/opencode.json"
        fi

        RUN_ID="${TEST_DIR%/}"
        echo "Running $RUN_ID"

        cd "$TEST_DIR" || exit 1

        rm -f opencode.json
        cp ../opencode.json ./opencode.json

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

        if [ "$XML_EXISTS" = "TRUE" ]; then
            XML_COMPILE_PASS=$(python - "$XML_FILE" <<'PYEOF'
import mujoco
import sys

try:
    mujoco.MjModel.from_xml_path(sys.argv[1])
    print("TRUE")
except Exception:
    print("FALSE")
PYEOF
)
        else
            XML_COMPILE_PASS="FALSE"
        fi

        cd ../..

        echo "$RUN_ID,$MODEL,$TEMP,$STEPS,$TIMEOUT_SEC,$ELAPSED,$TIMED_OUT,$TOTAL_TOKENS,$INPUT_TOKENS,$OUTPUT_TOKENS,$CACHE_READ,$CACHE_WRITE,$XML_EXISTS,$XML_FILE,$XML_COMPILE_PASS" >> "$RESULTS_CSV"
    done
done