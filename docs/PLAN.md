# Frostbite-30M: Architecture Specification & Implementation Plan

## 1. Executive Summary

Frostbite-30M is a 29.8 million parameter hybrid neural network designed for edge-deployed continuous control and autonomous compute routing. By synthesizing Self-Attention, Closed-form Continuous-time (CfC) liquid dynamics, and internal Reinforcement Learning (RL) decision cells, the architecture shifts from static computation to dynamic, agentic routing. The model is constrained to train entirely within a 12 GB VRAM environment while maintaining inference latency suitable for edge telemetry and embedded control systems.

## 2. System Architecture

The network consists of three primary macro-structures: the Temporal Embedding Engine, the Reinforced Liquid Blocks (the core of the hybrid approach), and the Global Laya-Style Cortex.

### 2.1 Parameter Budget allocation

The $d_{model}$ is fixed at 256 across all layers to ensure expressivity while strictly adhering to the memory constraints.

| Module | Sub-Component | Configuration | Approx. Params |
| --- | --- | --- | --- |
| **1. Embedding Layer** | Time-Series / Sensor Input | Linear projection, $d=256$ | 8.2M |
| **2. Reinforced Block (x6)** | Multi-Head Self-Attention | 4 heads, $d=256$ | 1.6M |
|  | Internal Auto-RL Router | 2-layer MLP (Output: Skip/Route/Halt) | 0.5M |
|  | CfC Continuous Substrate | 1-layer CfC replacing standard FFN | 12.5M |
| **3. Global Cortex** | Choice Head (Routing) | MLP $\rightarrow$ Softmax | 3.5M |
|  | Score Head (Evaluation) | MLP $\rightarrow$ Ordinal Scalar | 1.7M |
|  | Noul Head (Confidence) | MLP $\rightarrow$ Sigmoid (Binary) | 1.8M |
| **Total Budget** |  |  | **~29.8M** |

### 2.2 The Reinforced Liquid Block Anatomy

Unlike standard transformers where every token passes through every feed-forward network, each of the 6 Frostbite blocks contains an internal agent.

1. **Spatial Context (Attention):** The block receives input $X$. Multi-head attention evaluates the global spatial context across the sequence or sensor array.
2. **The Auto-RL Cell (Micro-Policy):** The attention output $Z$ is fed into the internal RL cell. The cell outputs a discrete probability distribution $\pi_{internal}(a\vert{}Z)$ over three actions:
* $A_0$ (`ROUTE`): Pass $Z$ into the CfC Substrate to evaluate liquid time-series dynamics.
* $A_1$ (`SKIP`): Bypass the CfC Substrate via a residual connection.
* $A_2$ (`HALT`): Immediately exit the layer stack and pass the current representation to the Global Cortex.


3. **Temporal Context (CfC):** If $A_0$ is chosen, the representation is passed through the liquid CfC layer to track continuous environment changes over time $t$.

## 3. Training Protocol & Optimization Strategy

Training is split into two phases to bridge the non-differentiable gap created by the internal RL cells, optimized for a single RTX 3060 12GB GPU.

### Phase 1: Substrate Pre-training (Differentiable Relaxation)

The goal is to teach the Attention and CfC layers useful representations before the internal agents learn to route.

* **Mechanism:** Use the **Gumbel-Softmax trick** (a continuous relaxation of discrete sampling). This allows gradients to flow backward through the "hard" choices of the Auto-RL cell.
* **Objective:** Next-state prediction (MSE) for time-series, or standard supervised cross-entropy for offline trajectories.
* **Hardware Profile:** Batch size of 128 using PyTorch Mixed Precision (`torch.amp`). The static model consumes < 1 GB VRAM, with activations peaking around 3.5 GB, leaving 8 GB for gradient history and optimizer states.

### Phase 2: Autonomous Reinforcement (PPO/REINFORCE)

Once representations are stable, we freeze the embedding and attention layers and train the Auto-RL cells and Global Cortex as actual agents.

* **Mechanism:** Transition from Gumbel-Softmax to categorical sampling. Gradients are calculated via the Log-Derivative trick (REINFORCE algorithm).
* **Reward Function Formulation:** The internal cells are trained on a composite reward $R_{total}$:

$$R_{total} = \alpha R_{task} + \beta R_{compute}$$


* $R_{task}$: Extrinsic reward from the environment (+1 for correct global choice, -1 for failure).
* $R_{compute}$: Intrinsic efficiency reward (+0.1 for choosing `SKIP`, +0.2 for `HALT`, -0.1 for `ROUTE`).


* **Outcome:** The model learns to use the heavy CfC compute only when the continuous dynamics are complex, and skips/halts when the environment is static.

## 4. Implementation Roadmap

### Sprint 1: Scaffolding & Continuous Substrate (Weeks 1-2)

* Initialize PyTorch environment with RTX 3060 CUDA optimization.
* Implement the base CfC layer utilizing the `ncps` (Neural Circuit Policies) PyTorch library.
* Construct the standard Attention mechanism.
* **Milestone:** A forward pass of a static (non-RL) Attention + CfC block matching the 30M parameter budget.

### Sprint 2: The Auto-RL Cell & Gumbel-Softmax (Weeks 3-4)

* Build the `MicroRouter` PyTorch module (2-layer MLP).
* Implement the Gumbel-Softmax straight-through estimator for Phase 1 differentiability.
* Wire the dynamic control flow (Route, Skip, Halt) using PyTorch conditional execution (`if/else` paths during inference, masked tensors during training).
* **Milestone:** Phase 1 supervised training loop running without gradient breaks.

### Sprint 3: Global Cortex & RL Optimization Loop (Weeks 5-6)

* Implement the Global Laya-style cortex (Choice, Score, Noul heads).
* Write the custom PPO/REINFORCE training loop for Phase 2.
* Design the reward function logic, ensuring intrinsic compute rewards are properly credited to the specific layers that made the routing choices.
* **Milestone:** Successful Phase 2 policy gradient updates on a toy environment (e.g., OpenAI Gym CartPole or simulated telemetry).

### Sprint 4: Edge Deployment Translation (Weeks 7-8)

* Export the PyTorch model to ONNX.
* Develop a lightweight inference wrapper (e.g., in Go or C++) for deployment on edge microcontrollers or single-board compute hardware.
* Test inference latency limits of the non-autoregressive routing heads.

## 5. Proposed Repository Structure

```text
frostbite-30m/
├── configs/
│   ├── arch_30m.yaml          # Hyperparameters (d_model, heads, budget)
│   └── train_phase1.yaml      # BPTT learning rates, batch sizes
├── frostbite/
│   ├── modules/
│   │   ├── attention.py       # Multi-head spatial context
│   │   ├── cfc_substrate.py   # Liquid dynamics (ncps wrapper)
│   │   └── auto_rl_cell.py    # Micro-router (Gumbel & Categorical)
│   ├── blocks/
│   │   └── reinforced_layer.py # The assembled hybrid layer
│   ├── heads/
│   │   └── laya_cortex.py     # Global Choice/Score/Noul heads
│   └── model.py               # Full Frostbite-30M assembly
├── training/
│   ├── phase1_supervised.py   # BPTT loop with Gumbel-Softmax
│   ├── phase2_reinforce.py    # Policy Gradient loop & Reward logic
│   └── rewards.py             # Intrinsic & Extrinsic reward functions
├── deployment/
│   ├── export_onnx.py         # Graph freezing for inference
│   └── go-inference/          # Edge deployment wrapper
│       ├── main.go
│       └── model_runner.go
├── tests/
│   ├── test_gradients.py      # Assert gradients flow through router
│   └── test_param_count.py    # Assert strict 30M limit
├── requirements.txt
└── README.md

```