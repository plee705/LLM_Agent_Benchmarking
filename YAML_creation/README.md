# Comparing LLM Prompting Outputs

## Why would we compare LLM prompt outputs?

While working to create environments and set up OpenCode for MuJoCo model creation, I quickly learned that various LLMs were producing different outputs for the same prompts. When creating context files, such as AGENTS.md files, I observed significant differences in results between models, such as ChatGPT, Copilot LLMs, and Gemma-4. Providing OpenCode with these prompts led to varying results, in which OpenCode showed difficulties in executing tasks described in context files from certain models.

This led me to wonder about how different these LLMs may produce outputs for the same prompts, and whether there are optimal LLMs for creating file types, such as context files for agents.

## The YAML Prompt Test:

LLM-based tools (agents, GPTs, etc.) can see performance improvements in task execution when concise, human-language prompts are provided. Rather than providing a long, detailed prompt, such as:

```text
I want to simulate an ideal pendulum with the MuJoCo physics engine, using the Python API. It should use a hinge joint to rotate about the y-axis, which is located on the end of a massless rod with length 0.2485 m. The other end of the rod should be welded to a 0.1 m diameter sphere with mass 1 kg. The model should be exposed to gravity, with an initial position of 90 degrees from vertical.
```

You can instead utilize a YAML file, which addresses the same concepts but in a concise format:

```code
model:
    name: simple_ideal_pendulum
    description: >
        Simple ideal pendulum in MuJoCo with a massless rod and spherical bob. 
    API: python, XML

environment:
    gravity:
        x: 0.0
        y: 0.0
        z: -9.81
        units: m/s^2

pendulum:
    rod:
        length: 0.2485
        .
        .
        .
```

And so forth. Benefits of such a prompt include:
- Lower Token Use
- Prevents Misunderstanding of Wording
- Consistent direction
- Ability for controlled revisions

## Testing 

Models will be tested by comparing their responses to identical prompts and/or tasks. Special configuration of LLM host settings (ChatGPT, Copilot) will be needed to ensure models don't reference tests of sibling models within their hosts. 

### Testing Set Up (ChatGPT):

- Turn off memory
- Upload prompts to new chats
- Do not use project folders
- Disable custom instructions

### Testing Process:

1. Create a standardized YAML task prompt
2. Upload the task to the LLM
3. Save the LLM output YAML file
4. Upload YAML file to OpenCode
5. Have OpenCode execute the task
6. evaluate OpenCode's outputs
