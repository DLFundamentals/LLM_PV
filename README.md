# LLM-ERM: 

Our work addresses the following question : 
> Can we design learning algorithms that combine the sample efficiency of finite-class program search with the computational efficiency of modern optimization methods?

## Abstract

We seek algorithms for program learning that are both sample-efficient and computationally feasible. Classical results show that targets admitting short program descriptions (e.g., with short python code can be learned with a small number of examples (scaling with the size of the code) via length-first program enumeration, but the search is exponential in description length. Consequently, Gradient-based training avoids this cost yet can require exponentially many samples on certain short-program families.

To address this gap, we introduce LLM-ERM, a propose-and-verify framework that replaces exhaustive enumeration with an LLM-guided search over candidate programs while retaining ERM-style selection on held-out data. Specifically, we draw $k$ candidates with a pretrained reasoning-augmented LLM, compile and check each on the data, and return the best verified hypothesis, with no feedback, adaptivity, or gradients. Theoretically, we show that coordinate-wise online mini-batch SGD requires many samples to learn certain short programs. {\em Empirically, LLM-ERM solves tasks such as parity variants, pattern matching, and primality testing with as few as 200 samples, while SGD-trained transformers overfit even with 100,000 samples}. These results indicate that language-guided program synthesis recovers much of the statistical efficiency of finite-class ERM while remaining computationally tractable, offering a practical route to learning succinct hypotheses beyond the reach of gradient-based training.

## LLM-ERM • In-Context • Fine-Tuning

The repository contains three types of experiments.

- **llm-erm** — Synthesize a Python function `f(x)` that matches the input-output relationship.
- **in_context_learning** — Run local models via **vLLM**; few-shot prompts with hundreds of examples; batched inference and accuracy summaries.
- **finetuning** — Train from scratch or finetune LMs (e.g., Qwen/Llama/DeepSeek) as **binary classifiers** on the same tasks; YAML sweeps included.

## Quickstart

```bash
# 1) Create environment (Conda)
conda env create -f environment.yaml
conda activate llm_erm

# 2) (Not Required) Already included in environment.yaml
pip install -r requirements.txt
```

### LLM-ERM (OpenAI)
```bash
export OPENAI_API_KEY=sk-...
python program_synthesis/runner.py --functions fn_a --lengths 50 --attempts 3 --enable-code-interpreter
```

### In-context learning (vLLM)
```bash
python in_context_learning/vllm_incontext.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507" --functions fn_a --lengths 50 --train-size 200 --test-size 100
```

### Fine-tuning (local training)
```bash
python finetuning/main.py --results-path ./results_new/0/0 --model qwen1.7B --target_func fn_a --sequence_length 50 --train_set_size 200 --test_set_size 10000 --batch_size 20 --n_epochs 200
```


## Repo layout
```
.
├── program_synthesis/        # LLM-ERM
├── in_context_learning/      # vLLM few-shot ICL runner (batched inference)
├── finetuning/               # Training/finetuning code + YAML sweeps
├── src/                      # Shared data generators & target functions
├── requirements.txt, environment.yaml
└── README.md
```

> Details, CLI flags, artifacts, and extensions live in each submodule’s README:
> - [`program_synthesis/README.md`](program_synthesis/README.md)  
> - [`in_context_learning/README.md`](in_context_learning/README.md)  
> - [`finetuning/README.md`](finetuning/README.md)
