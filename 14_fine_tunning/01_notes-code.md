# LLM Fine-Tuning Crash Course — Notes

**Video:** LLM Fine Tuning Crash Course | LLM Fine Tuning Tutorial
**Channel:** codebasics (Dhaval Patel & Hem Vadivel)
**Published:** October 28, 2025
**Link:** https://www.youtube.com/watch?v=IIvORO248Zs
**Code repo:** github.com/codebasics/llm-fine-tuning-crash-course (notebooks: `1_quantization_basics.ipynb`, `2_unsloth_finetuning.ipynb`)

---

## Video Roadmap (Timestamps)

| Time | Section |
|---|---|
| 00:00 | Intro |
| 00:17 | What is Fine Tuning |
| 06:48 | LoRA |
| 13:10 | Quantization Basics |
| 29:29 | QLoRA |
| 34:37 | Fine-Tuning Llama with Unsloth |

---

## 1. What Is Fine-Tuning?

- **Base/pretrained LLM** = a general-purpose model trained on massive, broad text corpora. It "knows a lot about everything" but isn't specialized.
- **Fine-tuning** = continuing to train that base model on a smaller, task/domain-specific dataset so it adapts its behavior, tone, format, or knowledge to a specific use case.
- Common reasons to fine-tune:
  - Teach a **specific style, tone, or output format** that prompting alone can't reliably enforce.
  - Inject **domain expertise** (legal, medical, technical, internal company data).
  - Specialize for a **narrow task** (classification, extraction, customer support, code generation in a specific framework).
  - Keep data **private** by training locally instead of depending on a third-party API.
- **Fine-tuning vs. RAG (important clarification):**
  - A common myth is "fine-tuning can't teach new knowledge, only RAG can." This is **false**.
  - Fine-tuning *does* change the model's weights, so it can genuinely learn new patterns/knowledge.
  - RAG only augments what the model *sees at inference time* (via retrieved context) — it never updates the underlying weights.
  - Rule of thumb: if better prompting or RAG already solves your problem, use that — it's cheaper and faster. Fine-tune only when those aren't enough.
- **Standard method:** Supervised Fine-Tuning (SFT) — training on labeled input/output (question/answer) pairs.
- Other post-training methods (mentioned for context): preference optimization (DPO, ORPO), distillation, and Reinforcement Learning approaches (GRPO, GSPO) where a model learns via reward feedback rather than static labels.
- **A good rule before jumping to Full Fine-Tuning (FFT):** try LoRA/QLoRA first. If it doesn't work with LoRA, it's unlikely to magically work with full fine-tuning (and FFT is far more compute-hungry).

---

## 2. LoRA (Low-Rank Adaptation)

**The core problem:** Large models have billions of parameters (e.g., Llama 70B ≈ 70 billion weights). Updating *all* of them during fine-tuning is extremely expensive in compute and memory.

**LoRA's idea:**
- Freeze the original pretrained weights entirely.
- For selected weight matrices, add two small ("thin") trainable matrices, **A** and **B**, whose product approximates the *update* to the original weight matrix.
- Only A and B are trained — this can be as little as **~1% of the total parameters**.
- Because A and B are low-rank, they're tiny compared to the full weight matrix, but still expressive enough to steer model behavior.

**Key hyperparameters:**
- **Rank (r):** controls the size/capacity of matrices A and B. Higher rank → more trainable parameters → more capacity, but more memory. Common values: 8 (fast), 16, 32, 64 (balanced), 128 (high quality).
- **Alpha:** scales the strength of the LoRA update relative to rank. A common heuristic is alpha = 2× rank.
- **Dropout:** regularization that randomly zeroes some LoRA activations during training to reduce overfitting.
- **Target modules:** which weight matrices get LoRA adapters — e.g., attention layers (`q_proj, k_proj, v_proj, o_proj`) and/or MLP/FFN layers (`gate_proj, up_proj, down_proj`). Applying LoRA to **both** attention and MLP layers tends to perform best.

**Why it matters:** LoRA can match full fine-tuning performance in many cases while using roughly **4× less VRAM**, because the frozen base weights don't need gradients or optimizer states — only the small adapter matrices do.

---

## 3. Quantization Basics

**What is quantization?** Reducing the numerical precision used to store model weights (and sometimes activations), trading a small amount of accuracy for a large reduction in memory footprint.

- Standard training/inference often uses **16-bit** (FP16/BF16) or **32-bit** (FP32) precision.
- Quantization compresses weights down to lower-bit representations — most notably **4-bit**.
- **4-bit NormalFloat (NF4):** a specialized 4-bit data type (not a plain 4-bit integer) designed to match the way neural network weights are typically distributed (roughly normal/Gaussian). This makes NF4 more information-efficient than naive 4-bit integer quantization, preserving more of the original signal per bit used.
- **Double quantization:** quantizing the quantization *constants* themselves (the scaling factors used during quantization) to squeeze out additional memory savings.
- **Paged optimizers:** a memory-management technique that handles GPU memory spikes during training (borrowed from OS-style memory paging) so training doesn't crash from occasional memory bursts.
- **Net effect:** quantizing a 7B-parameter model's weights to 4-bit can shrink weight storage from ~14–28 GB (16/32-bit) down to roughly **3.5 GB**, a memory reduction of **over 75%**.

---

## 4. QLoRA (Quantized LoRA)

**QLoRA = quantization + LoRA combined.**

- The frozen base model is loaded in **4-bit precision** (instead of LoRA's usual 16-bit), while the small trainable LoRA adapter matrices (A and B) are still kept in higher precision (e.g., 16-bit) for stable training.
- This lets you fine-tune much larger models on much smaller GPUs than LoRA alone would allow.

**LoRA vs. QLoRA comparison:**

| | LoRA | QLoRA |
|---|---|---|
| Base model precision | 16-bit | 4-bit |
| VRAM usage | Higher (~4× more than QLoRA) | ~4× less than LoRA |
| Speed | Slightly faster | Slightly slower |
| Accuracy | Slightly higher | Marginally lower (near-negligible gap) |
| Best for | Max accuracy, when VRAM isn't a constraint | Consumer GPUs / very large models on limited hardware |

- Illustrative rough memory comparison for a 7B model:
  - **Full fine-tuning:** weights + gradients + optimizer states ≈ 80+ GB → needs multiple high-end GPUs (e.g., 2× A100 80GB).
  - **QLoRA:** ≈ 8 GB total → fits comfortably on a single consumer GPU (e.g., RTX 4090, 24GB) or even smaller cards.
- Empirically, the quality gap between full fine-tuning and QLoRA is often only **1–2%**, a small tradeoff for roughly a 10× reduction in hardware requirements.
- QLoRA can even fine-tune a **70B parameter model on under 48GB VRAM**, which was previously unthinkable outside large multi-GPU clusters.
- A further accuracy tip from the QLoRA paper: masking out the input/prompt portion and training loss only on the **completion (assistant's answer)** tokens can improve results by roughly 1%.

---

## 5. Fine-Tuning Llama with Unsloth (Hands-On)

**What is Unsloth?** An optimization library/framework that makes LoRA/QLoRA fine-tuning significantly faster and more memory-efficient than standard Hugging Face training pipelines, via hand-optimized GPU kernels — while staying compatible with the Hugging Face ecosystem (transformers, PEFT, TRL).

- Benefits typically cited: **~2× faster training** and **up to 60–80% less VRAM** versus vanilla implementations.
- Can fine-tune models with as little as **3GB VRAM** for smaller models, and makes 70B-scale fine-tuning feasible on a single consumer-class GPU when combined with QLoRA.

**General Unsloth + Llama fine-tuning workflow:**

1. **Environment setup**
   - Create a clean Python environment.
   - Install PyTorch with CUDA support.
   - Install Unsloth (it pulls in compatible versions of `bitsandbytes`, `transformers`, `peft`, `trl` automatically).

2. **Load the base model**
   - Choose a beginner-friendly model such as **Llama 3.1 (8B)** or **Llama 3.2**, typically the pre-quantized 4-bit version for QLoRA.

3. **Prepare the dataset**
   - Data must be in a tokenizable format — typically a set of conversational examples with `role` (user/assistant) and `content` fields, or a simple question/answer pair structure.
   - **Data quality over quantity**: a few hundred to a few thousand excellent, consistent examples usually beat tens of thousands of noisy ones.
   - Keep formatting consistent and cover diverse scenarios relevant to the target use case.
   - Most examples should fit within the chosen max sequence length (e.g., 2048–4096 tokens).

4. **Configure LoRA/QLoRA parameters**
   - Set rank (r), alpha, dropout, and target modules.
   - Set learning rate (commonly **2e-4** as a starting point for LoRA/QLoRA), batch size, gradient accumulation steps, and number of epochs (often ~3 for small high-quality datasets, fewer for large/noisy ones).

5. **Train**
   - Use an SFT-style trainer (e.g., Hugging Face's `SFTTrainer`) configured with the model, tokenizer, and dataset.
   - Optionally set up an evaluation dataset and periodic evaluation steps to monitor performance during training.

6. **Save / export the model**
   - Save just the LoRA adapter weights (small, portable), or merge the adapters back into the base model to produce a single standalone fine-tuned model (e.g., `save_pretrained_merged(..., save_method="merged_16bit")`).
   - Models can also be converted to formats like GGUF for lightweight local inference.

7. **Benchmark / validate**
   - Run generation on sample prompts before and after fine-tuning to sanity-check behavior changes.
   - Measure inference speed (tokens/sec) and quality (automated metrics + manual/LLM-as-judge review).

---

## Key Takeaways

1. Fine-tuning genuinely updates model weights and **can** teach new knowledge — it's not automatically inferior to RAG; they solve different problems.
2. **LoRA** trains small adapter matrices instead of full weights, cutting trainable parameters to ~1% of the original model.
3. **Quantization** (especially 4-bit NF4) shrinks memory footprint by storing weights at lower precision with minimal accuracy loss.
4. **QLoRA** = 4-bit quantized base model + LoRA adapters — enables fine-tuning very large models on a single consumer GPU.
5. **Unsloth** is a practical toolkit that makes the whole LoRA/QLoRA pipeline faster and more memory-efficient, and is well-suited for hands-on fine-tuning of models like Llama on modest hardware.
6. Practical advice: start small (smaller model, small high-quality dataset), verify the pipeline works, then scale up — and always ask whether fine-tuning is even necessary before investing the effort.

---

## 6. Actual Notebook Code (from `2_unsloth_finetuning.ipynb`)

The course's companion notebook fine-tunes **Llama-3.2-3B-Instruct** on the **ServiceNow-AI/R1-Distill-SFT** dataset to give it DeepSeek-R1-style "thinking" behavior. Below is the actual code, step by step.

### 6.1 Install dependencies

```python
!pip install -q unsloth
# Also get the latest nightly Unsloth!
!pip install -q --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

### 6.2 Load the base model in 4-bit

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # Choose any! Unsloth also supports RoPE scaling internally.
dtype = None            # None for auto detection. Float16 for Tesla T4/V100, Bfloat16 for Ampere+
load_in_4bit = True      # Use 4-bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",  # or "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,  # Will load the 4-bit quantized model
)
```

Parameter notes from the notebook:
- `model_name`: the pretrained model to load.
- `max_seq_length`: max tokens the model can process (here 2048).
- `dtype`: `None` auto-selects; `torch.float16` for older GPUs (T4/V100), `torch.bfloat16` for newer (A100).
- `load_in_4bit`: enables 4-bit quantization for memory efficiency.

### 6.3 Attach LoRA adapters (for QLoRA fine-tuning)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,       # higher alpha = more weight to LoRA activations
    lora_dropout = 0,      # Supports any, but 0 is optimized
    bias = "none",         # Supports any, but "none" is optimized
    use_gradient_checkpointing = "unsloth",  # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

Parameter notes:
- `r`: rank of the LoRA matrices; higher = more capacity, more memory.
- `target_modules`: which layers get adapters (all attention + MLP layers here).
- `lora_alpha`: scaling factor for LoRA updates.
- `lora_dropout`: dropout for LoRA layers (0 = fastest/optimized in Unsloth).
- `bias`: how biases are handled ("none" is optimized for speed).
- `use_gradient_checkpointing`: `"unsloth"` uses Unsloth's optimized checkpointing to cut memory use further.
- `use_rslora`: toggle for Rank-Stabilized LoRA (more stable training variant).
- `loftq_config`: optional LoftQ (quantization-aware init) config; disabled here.

### 6.4 Load and format the dataset

```python
from datasets import load_dataset
dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", 'v0', split="train")
```

Each row has fields like `problem`, `reannotated_assistant_content` (the R1-style reasoning/thinking trace), and `solution`. A custom prompt template wraps these into one training string:

```python
r1_prompt = """You are a reflective assistant engaging in thorough, iterative reasoning, mimicking human stream-of-consciousness thinking. Your approach emphasizes exploration, self-doubt, and continuous refinement before coming up with an answer.
<problem>
{}
</problem>

{}
{}
"""
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    problems = examples["problem"]
    thoughts = examples["reannotated_assistant_content"]
    solutions = examples["solution"]
    texts = []

    for problem, thought, solution in zip(problems, thoughts, solutions):
        text = r1_prompt.format(problem, thought, solution) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True,)
```

This teaches the model to output a `<think>...</think>`-style reasoning block before the final answer, matching the DeepSeek-R1 pattern.

### 6.5 Set up the trainer

```python
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,     # parallel processes for tokenizing dataset
    packing = False,          # can make training 5x faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,           # total training steps (demo-scale, not a full epoch)
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",     # memory-efficient 8-bit optimizer
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",       # set to "wandb"/"tensorboard" etc. for tracking
    ),
)
```

### 6.6 Train

```python
trainer_stats = trainer.train()
```

In the notebook run, training loss dropped from about **0.91** at step 1 to roughly **0.4–0.6** by step 60 (fluctuating, since this is a short 60-step demo run on a T4 GPU, not a full training pass over the ~172k-example dataset).

### 6.7 Run inference on the fine-tuned model

```python
from unsloth.chat_templates import get_chat_template

sys_prompt = """You are a reflective assistant engaging in thorough, iterative reasoning, mimicking human stream-of-consciousness thinking. Your approach emphasizes exploration, self-doubt, and continuous refinement before coming up with an answer.
<problem>
{}
</problem>
"""
message = sys_prompt.format("How many 'r's are present in 'strawberry'?")

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"role": "user", "content": message},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 1024,
    use_cache = True,
    temperature = 1.5,
    min_p = 0.1,
)
response = tokenizer.batch_decode(outputs)
print(response[0])
```

In the notebook, the fine-tuned model responded with a visible "thinking" section before giving a final answer (correctly counting 3 'r's in "strawberry" during its reasoning, though the final formatted answer in that particular run miscounted — a reminder that a 60-step demo fine-tune won't be fully reliable).

### 6.8 Save the model locally

```python
model.save_pretrained("chintan-001-3B")       # Local saving
tokenizer.save_pretrained("chintan-001-3B")
```

### 6.9 Merge LoRA into the base model and export to GGUF (for Ollama / llama.cpp)

```python
model.save_pretrained_gguf("chintan-001-3B-GGUF", tokenizer,)
```

This merges the 4-bit + LoRA weights back into 16-bit full weights, then converts to **GGUF** format (quantized to **Q8_0** in this run) — the format used by `llama.cpp`-based tools like Ollama.

### 6.10 Run the model locally via Ollama (optional)

```python
!curl -fsSL https://ollama.com/install.sh | sh
```

```python
import subprocess
subprocess.Popen(["ollama", "serve"])
import time
time.sleep(3)
```

```python
print(tokenizer._ollama_modelfile)
```

This prints an auto-generated Ollama `Modelfile`, including the Llama-3.1 chat template and default sampling parameters (`temperature 1.5`, `min_p 0.1`).

```python
!ollama create unsloth_model -f ./chintan-001-3B-GGUF/Modelfile
```

This registers the fine-tuned GGUF model with Ollama under the name `unsloth_model`, so it can be run locally like any other Ollama model (e.g., `ollama run unsloth_model`).

---

## Key Takeaways

1. Fine-tuning genuinely updates model weights and **can** teach new knowledge — it's not automatically inferior to RAG; they solve different problems.
2. **LoRA** trains small adapter matrices instead of full weights, cutting trainable parameters to ~1% of the original model.
3. **Quantization** (especially 4-bit NF4) shrinks memory footprint by storing weights at lower precision with minimal accuracy loss.
4. **QLoRA** = 4-bit quantized base model + LoRA adapters — enables fine-tuning very large models on a single consumer GPU.
5. **Unsloth** is a practical toolkit that makes the whole LoRA/QLoRA pipeline faster and more memory-efficient, and is well-suited for hands-on fine-tuning of models like Llama on modest hardware.
6. The notebook fine-tunes **Llama-3.2-3B-Instruct** on a DeepSeek-R1-style reasoning dataset (`ServiceNow-AI/R1-Distill-SFT`) using a custom `<problem>...<think>...</think>...` prompt format, then exports the result to GGUF for local use via Ollama.
7. Practical advice: start small (smaller model, small high-quality dataset, few training steps), verify the pipeline works end-to-end, then scale up.

---

*Note: The conceptual sections above are compiled from the video's official title, description, and timestamp outline, cross-referenced with authoritative technical documentation on LoRA, quantization, QLoRA, and Unsloth (since a full transcript of the video itself wasn't accessible). Section 6 is transcribed directly from the actual companion Jupyter notebook you provided.*
