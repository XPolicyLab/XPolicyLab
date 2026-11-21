# Astra-6.2-Pilot

idata-robot's evaluation-only RoboDojo simulation pilot for ARX X5, using joint actions and one environment at a time. The module name is `Astra_6_2_Pilot`. The server sends the official instruction, three RGB views when available, measured joint positions and a short note from the previous decision to GPT-6.2 Astra via the OpenAI Responses API. Each reply supplies absolute joint targets. The adapter interpolates eight steps and limits the arm displacement to 0.25 radians per decision. An episode reset clears the note and observation. The policy uses the shared X5 embodiment description and has no task-specific prompt.

## Installation

Place this checkout at `RoboDojo/XPolicyLab`. Install RoboDojo's simulator, robot assets and environment configurations in the parent checkout. From the XPolicyLab root:

```bash
conda create -y -n astra-pilot python=3.12 pip
conda run -n astra-pilot bash policy/Astra_6_2_Pilot/install.sh
```

Configure `OPENAI_API_KEY` in the policy server environment with access to `gpt-6.2-astra`. API inference sends observations to the configured OpenAI endpoint and incurs usage charges. This adapter requests high reasoning effort and disables response storage. The client request timeout is 210 seconds.

## Data processing and training

Inference uses a hosted model. There is no local checkpoint, dataset conversion or training step.

## Evaluation

From the XPolicyLab root, replace `robodojo` with the simulator's conda environment name:

```bash
EVAL_ENV_TYPE=sim bash policy/Astra_6_2_Pilot/eval.sh \
  RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 astra-pilot robodojo
```

For transport debugging, select `EVAL_ENV_TYPE=debug`. The debug client supplies synthetic observations and still calls the configured model endpoint. Set `DEBUG_OBS_ENCODED=1` to exercise encoded camera inputs. Initial validation used a local Responses API stub and the stock debug client, one 20-step episode each for raw and encoded observations. Simulator success rate and live model latency are pending.
