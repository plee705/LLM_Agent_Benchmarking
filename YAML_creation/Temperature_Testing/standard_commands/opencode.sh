#!/bin/bash

PROMPT="Read the YAML file in the current directory. Create files and execute necessary tasks as described in the prompt."

/usr/bin/time \
  -f "elapsed_sec=%e" \
  -o opencode_timing.log \
  timeout 600 \
  opencode run --format json "$PROMPT" \
  > opencode_events.jsonl \
  2> opencode_stderr.log