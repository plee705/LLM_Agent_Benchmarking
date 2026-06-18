/usr/bin/time \
  -f "elapsed_sec=%e" \
  -o opencode_timing.log \
  timeout 600 \
  opencode run \
  "Read the YAML file in the current directory. Create files and execute necessary tasks as described in the prompt." \
  2> >(tee opencode_stderr.log >&2) \
  | tee opencode_stdout.log