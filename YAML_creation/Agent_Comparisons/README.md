# Agent Comparisons


## Overview of OpenCode Temperature Testing

Through June 18-19, 2026, I tested the impact of temperature settings on agentic performance in OpenCode with the Gemma-4 26B LLM model. OpenCode was provided with a prompt inside of an isolated folder which included a YAML instructions file, providing directions to create a pendulum model file in MuJoCo. Eight temperature values were evaluated, with each being tested 30 times, for a total of 240 test runs. Indicators such as token usage, computational time, and agent looping rate were tracked.

LLM temperature is a tunable parameter that changes output randomness, and is on a scale of 0 to 1. A temperature value of 0 is most likely to repeat output, whereas the LLM is the most randomized at a temperature of 1.

During the course of testing, we observed an average overall failure rate of 40.1% for all temperatures. A failure was indicated when the agent exceeded the threshold time of 150 seconds to complete the task. Often, this is due to looping. Testing showed that the OpenCode/Gemma-4 failure rate was independent of temperature, as all temperatures were within 5% of the average, except for `T = 0.1`, with an average failure rate of 32.1%.

![Bar chart showing timeout failure rates across eight temperature values from 0.1 to 1.0 for OpenCode with Gemma-4 26B LLM model. The y-axis displays failure rate percentage ranging from 0 to 50 percent, and the x-axis shows temperature settings. Most temperatures cluster around 40 percent failure rate, while temperature 0.1 shows a notably lower failure rate of approximately 32 percent. The chart title indicates n equals 28, with a timestamp of 20260622_110052_330101.](../Temperature_Testing/analysis_plots/timeout_rate_vs_temp_single_bar_n28_20260622_110052_330101.png)