# MiniGPT

> **A GPT-style Large Language Model built from scratch with PyTorch — from BPE tokenization and Transformer architecture to pretraining, fine-tuning, optimized inference, evaluation, and production serving.**

MiniGPT is an educational and engineering-focused implementation of a GPT-style language model built **from first principles**.

The goal is not simply to use an existing LLM library, but to understand and implement the major components behind modern autoregressive language models:

```text
Raw Text
   │
   ▼
BPE Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Token Embeddings
   │
   ▼
Transformer Blocks
   │
   ├── Layer Normalization
   ├── Causal Self-Attention
   ├── MLP / Feed Forward
   └── Residual Connections
   │
   ▼
Language Model Head
   │
   ▼
Next Token Prediction
   │
   ▼
Text Generation
```

The project is designed to evolve from a **small educational GPT model** into a complete **LLM training and serving system**.

---

## Project Goals

The primary goal is to understand the complete lifecycle of an LLM.

### Core objectives

* Build a BPE tokenizer from scratch
* Implement GPT architecture using PyTorch
* Implement causal self-attention
* Implement Transformer blocks
* Build a complete training loop
* Train a language model from raw text
* Implement checkpointing and recovery
* Support mixed-precision training
* Implement distributed training
* Implement instruction fine-tuning
* Implement efficient inference
* Implement KV caching
* Build an OpenAI-compatible API
* Evaluate model quality
* Containerize the system
* Deploy the model as a production service

---

# Architecture

```text
                         ┌──────────────────┐
                         │    Raw Dataset   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Data Processing  │
                         │ Cleaning/Dedup   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  BPE Tokenizer   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Tokenized Dataset│
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │         MiniGPT            │
                    │                            │
                    │ Token Embeddings           │
                    │       ↓                    │
                    │ Transformer Block          │
                    │       ↓                    │
                    │ Causal Self-Attention      │
                    │       ↓                    │
                    │ MLP / FFN                  │
                    │       ↓                    │
                    │ Residual Connections       │
                    │       ↓                    │
                    │ LayerNorm                  │
                    │       ↓                    │
                    │ Transformer × N            │
                    │       ↓                    │
                    │ Language Model Head        │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
              Pretraining                Evaluation
                    │
                    ▼
              Fine-Tuning
                    │
                    ▼
             Optimized Inference
                    │
                    ▼
              FastAPI Server
                    │
                    ▼
          Docker / Kubernetes / Cloud
```

---

# Model Architecture

MiniGPT follows the decoder-only Transformer architecture used by GPT-style language models.

```text
Input Tokens
     │
     ▼
Token Embeddings
     │
     ▼
┌─────────────────────────────┐
│     Transformer Block       │
│                             │
│  LayerNorm                  │
│      ↓                      │
│  Causal Self-Attention      │
│      ↓                      │
│  Residual Connection        │
│      ↓                      │
│  LayerNorm                  │
│      ↓                      │
│  Feed Forward / MLP         │
│      ↓                      │
│  Residual Connection        │
└──────────────┬──────────────┘
               │
               ▼
        Repeat N layers
               │
               ▼
          Final LayerNorm
               │
               ▼
          LM Head
               │
               ▼
         Token Probabilities
```

---

# Core Components

## 1. BPE Tokenizer

The tokenizer converts raw text into subword tokens.

```text
"Hello world"

        ↓

["Hello", " world"]

        ↓

[15496, 995]
```

The tokenizer implementation covers:

* Vocabulary creation
* Pair-frequency calculation
* BPE merge rules
* Encoding
* Decoding
* Special tokens
* Vocabulary persistence

---

# 2. Causal Self-Attention

The core attention mechanism is:

```text
Q = XWq
K = XWk
V = XWv

Attention(Q,K,V)
=
softmax(QKᵀ / √d)V
```

MiniGPT applies a causal mask so that a token cannot attend to future tokens.

```text
        Token 1  Token 2  Token 3  Token 4

Token 1    ✓
Token 2    ✓        ✓
Token 3    ✓        ✓        ✓
Token 4    ✓        ✓        ✓        ✓
```

This ensures autoregressive generation.

---

# 3. Transformer Block

Each Transformer block contains:

```text
Input
  │
  ▼
LayerNorm
  │
  ▼
Causal Self-Attention
  │
  ▼
Residual Connection
  │
  ▼
LayerNorm
  │
  ▼
MLP
  │
  ▼
Residual Connection
```

---

# 4. Language Modeling Objective

MiniGPT is trained using next-token prediction.

Given:

```text
The cat is
```

the model learns to predict:

```text
The cat is → sleeping
```

Training data becomes:

```text
Input:

[The, cat, is]

Target:

[cat, is, sleeping]
```

The model minimizes cross-entropy loss:

```text
L = -Σ log P(xₜ | x₁,...,xₜ₋₁)
```

---

# Project Structure

```text
minigpt/
│
├── tokenizer/
│   ├── bpe.py
│   ├── trainer.py
│   └── tokenizer.py
│
├── model/
│   ├── embeddings.py
│   ├── attention.py
│   ├── mlp.py
│   ├── transformer.py
│   └── gpt.py
│
├── data/
│   ├── preprocessing.py
│   ├── cleaning.py
│   ├── deduplication.py
│   └── dataset.py
│
├── training/
│   ├── trainer.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── checkpoint.py
│   └── distributed.py
│
├── finetuning/
│   ├── sft.py
│   └── preference.py
│
├── inference/
│   ├── generate.py
│   ├── sampler.py
│   ├── kv_cache.py
│   └── batching.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── benchmarks.py
│   └── evaluator.py
│
├── serving/
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
│
├── distributed/
│   ├── ddp.py
│   └── fsdp.py
│
├── configs/
│   ├── tiny.yaml
│   ├── small.yaml
│   └── gpt2.yaml
│
├── tests/
│
├── scripts/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# Model Configurations

MiniGPT supports configurable model sizes.

| Configuration | Parameters | Layers | Hidden Size | Heads | Context |
| ------------- | ---------: | -----: | ----------: | ----: | ------: |
| Tiny          |       ~10M |      6 |         384 |     6 |     512 |
| Small         |       ~50M |      8 |         512 |     8 |     512 |
| GPT-2 Small   |      ~117M |     12 |         768 |    12 |    1024 |

> Parameter counts are approximate and depend on vocabulary size and implementation details.

The initial development target is the **Tiny** configuration so that the complete pipeline can be trained and debugged on limited hardware.

---

# Getting Started

## Requirements

* Python 3.10+
* PyTorch
* CUDA GPU recommended for training
* Git

Clone the repository:

```bash
git clone https://github.com/<your-username>/minigpt.git

cd minigpt
```

Create an environment:

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -e .
```

---

# Train the Tokenizer

```bash
python -m tokenizer.trainer \
    --input data/raw.txt \
    --vocab-size 16000 \
    --output tokenizer/
```

Test it:

```python
from tokenizer.tokenizer import BPETokenizer

tokenizer = BPETokenizer.load("tokenizer/")

text = "Hello, world!"

tokens = tokenizer.encode(text)

print(tokens)

print(tokenizer.decode(tokens))
```

---

#  Train MiniGPT

Start with the tiny configuration:

```bash
python -m training.trainer \
    --config configs/tiny.yaml
```

Training output:

```text
Step 100
Train Loss: 5.42
Learning Rate: 0.00030

Step 200
Train Loss: 4.87
Learning Rate: 0.00029

Step 300
Train Loss: 4.31
Learning Rate: 0.00028
```

---

#  Checkpointing

Training checkpoints contain:

```text
checkpoint/
├── model.pt
├── optimizer.pt
├── scheduler.pt
├── rng_state.pt
└── metadata.json
```

Resume training:

```bash
python -m training.trainer \
    --config configs/tiny.yaml \
    --resume checkpoints/latest/
```

The objective is to make training recoverable after interruptions or hardware failures.

---

# Text Generation

After training:

```bash
python -m inference.generate \
    --checkpoint checkpoints/latest \
    --prompt "Artificial intelligence is"
```

Example:

```text
Artificial intelligence is transforming how
software systems understand information and
interact with users...
```

Generation parameters:

```text
temperature
top_k
top_p
max_tokens
repetition_penalty
```

Example:

```bash
python -m inference.generate \
    --checkpoint checkpoints/latest \
    --prompt "The future of AI" \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-tokens 200
```

---

#  KV Cache

MiniGPT implements KV caching for autoregressive generation.

Without KV caching:

```text
Token 1 → recompute
Token 2 → recompute token 1 + 2
Token 3 → recompute token 1 + 2 + 3
...
```

With KV caching:

```text
Previous K,V
      +
New Token
      ↓
Attention
      ↓
New K,V
```

This significantly reduces redundant computation during generation.

---

#  Instruction Fine-Tuning

After pretraining, MiniGPT can be fine-tuned on instruction datasets.

```text
Base MiniGPT
      │
      ▼
Instruction Dataset
      │
      ▼
Supervised Fine-Tuning
      │
      ▼
Instruction-Tuned MiniGPT
```

Example:

```json
{
  "instruction": "Explain binary search.",
  "response": "Binary search is an algorithm..."
}
```

Run:

```bash
python -m finetuning.sft \
    --model checkpoints/pretrained \
    --dataset data/instructions.jsonl
```

---

#  API Server

MiniGPT provides an OpenAI-compatible API.

Start the server:

```bash
uvicorn serving.main:app \
    --host 0.0.0.0 \
    --port 8000
```

API:

```text
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

Example:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minigpt",
    "messages": [
      {
        "role": "user",
        "content": "Explain transformers."
      }
    ],
    "temperature": 0.7
  }'
```

---

#  Docker

Build:

```bash
docker build -t minigpt .
```

Run:

```bash
docker run \
    --gpus all \
    -p 8000:8000 \
    minigpt
```

---

#  Evaluation

MiniGPT tracks:

### Training metrics

* Training loss
* Validation loss
* Perplexity
* Learning rate
* Gradient norm
* Tokens/second
* GPU utilization

### Generation metrics

* Generation latency
* Tokens/second
* Time to first token
* Memory usage
* KV-cache memory

### Model evaluation

Planned benchmarks include:

* Language modeling perplexity
* Knowledge evaluation
* Basic reasoning
* Reading comprehension
* Instruction following

---

#  Experiment Tracking

Each experiment records:

```text
Experiment
├── Model configuration
├── Dataset
├── Tokenizer
├── Batch size
├── Learning rate
├── Optimizer
├── Training steps
├── Validation loss
├── Perplexity
├── Throughput
├── GPU memory
└── Checkpoint
```

Example:

```text
Experiment: tiny-v1

Parameters:      10.4M
Dataset tokens:  50M
Batch size:      32
Context:         512
Optimizer:       AdamW
Learning rate:   3e-4

Final train loss: 2.31
Final val loss:   2.48
```

Actual benchmark values will be added as experiments are completed.

---

#  Distributed Training

The project is designed to support multi-GPU training.

```text
                 Training Job
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        GPU 0       GPU 1       GPU 2
          │           │           │
          └───────────┼───────────┘
                      │
                Gradient Sync
```

Planned support:

* PyTorch DDP
* FSDP
* Mixed precision
* BF16
* Gradient accumulation
* Gradient checkpointing
* Sharded checkpoints

---

# 🔬 Training Pipeline

The complete pipeline:

```text
Raw Data
   ↓
Cleaning
   ↓
Deduplication
   ↓
Quality Filtering
   ↓
Tokenization
   ↓
Dataset Sharding
   ↓
Pretraining
   ↓
Validation
   ↓
Checkpointing
   ↓
Instruction Fine-Tuning
   ↓
Evaluation
   ↓
Inference Optimization
   ↓
API Serving
```

---

# 🧪 Testing

Run unit tests:

```bash
pytest tests/unit
```

Integration tests:

```bash
pytest tests/integration
```

All tests:

```bash
pytest
```

Tests cover:

* Tokenizer correctness
* Attention masking
* Model shapes
* Forward pass
* Loss computation
* Checkpoint restoration
* Generation
* API endpoints

---

# Design Principles

MiniGPT follows several engineering principles:

### 1. Understand before optimizing

The initial implementation favors clarity over maximum performance.

### 2. Reproducibility

Experiments should be reproducible through:

* Fixed seeds
* Configuration files
* Dataset versions
* Model checkpoints
* Experiment metadata

### 3. Modular architecture

Tokenizer, model, training, inference, evaluation, and serving are independently testable.

### 4. Production-oriented engineering

The project progressively introduces:

```text
Testing
+
Observability
+
Distributed training
+
Fault tolerance
+
Containerization
+
Cloud deployment
```

---

#  Roadmap

## Phase 1 — Foundation

* [x] Project structure
* [ ] BPE tokenizer
* [ ] Dataset pipeline
* [ ] GPT architecture
* [ ] Causal attention
* [ ] Training loop

## Phase 2 — Training

* [ ] AdamW
* [ ] Learning-rate scheduling
* [ ] Mixed precision
* [ ] Gradient accumulation
* [ ] Checkpointing
* [ ] Resume training
* [ ] Validation pipeline

## Phase 3 — Scaling

* [ ] Multi-GPU DDP
* [ ] FSDP
* [ ] Gradient checkpointing
* [ ] Dataset sharding
* [ ] Distributed checkpoints

## Phase 4 — Fine-Tuning

* [ ] Instruction tuning
* [ ] SFT
* [ ] Preference optimization
* [ ] Evaluation datasets

## Phase 5 — Inference

* [ ] KV cache
* [ ] Sampling strategies
* [ ] Streaming generation
* [ ] Continuous batching
* [ ] Quantization

## Phase 6 — Production

* [ ] FastAPI server
* [ ] OpenAI-compatible API
* [ ] Docker
* [ ] Kubernetes
* [ ] Prometheus
* [ ] Grafana
* [ ] OpenTelemetry
* [ ] Load testing
* [ ] Autoscaling

---

#  What I Am Learning

This project is intended to develop practical understanding of:

```text
Deep Learning
     │
     ├── Neural Networks
     ├── Optimization
     ├── Backpropagation
     └── Mixed Precision
     
Transformers
     │
     ├── Attention
     ├── Causal Masking
     ├── Positional Encoding
     └── Transformer Blocks

LLMs
     │
     ├── Tokenization
     ├── Pretraining
     ├── Fine-Tuning
     ├── Evaluation
     └── Inference

Systems
     │
     ├── Distributed Training
     ├── GPU Optimization
     ├── Model Serving
     ├── APIs
     └── Cloud Deployment
```

---

# Project Scope

MiniGPT is primarily an **engineering and learning project**.

It is not intended to compete with modern frontier models.

The purpose is to understand the architecture, algorithms, training process, optimization techniques, and infrastructure required to build and serve GPT-style language models.

---

# Contributing

Contributions are welcome.

Potential contribution areas:

* Tokenization
* Model architecture
* Training optimization
* Distributed training
* Evaluation
* Inference optimization
* Documentation
* Testing

```bash
git checkout -b feature/my-feature
```

Make your changes, add tests, and submit a pull request.

---


# Project Philosophy

> **Don't just call an LLM API. Understand how the model works.**

MiniGPT is an attempt to go from:

```text
"How do I use an LLM?"
```

to:

```text
"How do I build, train, optimize, evaluate,
serve, and scale an LLM?"
```

---

## Long-Term Vision

The long-term goal is to evolve MiniGPT into a complete open-source LLM engineering stack:

```text
                 MiniGPT
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Training     Evaluation   Inference
        │           │           │
        ▼           ▼           ▼
   Distributed   Benchmarks   KV Cache
        │                       │
        └──────────┬────────────┘
                   ▼
              Model Server
                   │
                   ▼
                FastAPI
                   │
                   ▼
             Kubernetes
                   │
                   ▼
                 Cloud
```

**From tokenizer to Transformer.
From training to inference.
From research code to production infrastructure.**
