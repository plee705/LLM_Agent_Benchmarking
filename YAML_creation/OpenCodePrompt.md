## Base Prompt

Read the YAML file in the current directory. Create files and execute necessary tasks as described in the prompt.

## Terminal Prompt

/usr/bin/time -f "elapsed_sec=%e" -o opencode_timing.log \
opencode run "Read the YAML file in the current directory. Create files and execute necessary tasks as described in the prompt." \
2> >(tee opencode_stderr.log >&2) | tee opencode_stdout.log

### What this does:

- Establishes a timing and recording log for OpenCode
- Provides OpenCode with the standardized prompt
- OpenCode executes the task per the prompt and accessible files
- Bash script records exe

## Method:

Sometimes OpenCode doesn't exit out of the program after finishing the command. Use the terminal prompt to run OpenCode. 