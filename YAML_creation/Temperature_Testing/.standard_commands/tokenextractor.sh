#!/bin/bash

jq -s '
{
  total_tokens: ([.[] | select(.type=="step_finish") | .part.tokens.total] | add),
  input_tokens: ([.[] | select(.type=="step_finish") | .part.tokens.input] | add),
  output_tokens: ([.[] | select(.type=="step_finish") | .part.tokens.output] | add),
  cache_read_tokens: ([.[] | select(.type=="step_finish") | .part.tokens.cache.read] | add),
  cache_write_tokens: ([.[] | select(.type=="step_finish") | .part.tokens.cache.write] | add)
}
' opencode_events.jsonl