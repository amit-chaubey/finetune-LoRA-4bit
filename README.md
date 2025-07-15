
base_model: microsoft/Phi-3-mini-4k-instruct
library_name: peft
---

# Model Card for LoRA Fine-Tuned Phi-3-mini-4k

This model is a LoRA (Low-Rank Adaptation) fine-tuned version of the `microsoft/Phi-3-mini-4k-instruct` model, using 4-bit quantization for efficient training and inference. The model is trained and tested for instruction-following tasks and can be easily saved and shared via the Hugging Face Hub.

---

## Model Details

### Model Description

This model adapts the Phi-3-mini-4k-instruct LLM using LoRA, a parameter-efficient fine-tuning technique, and 4-bit quantization for reduced memory usage. It is suitable for a variety of NLP tasks, especially where resource efficiency is important.

- **Developed by:** Amit Chaubey
- **Funded by [optional]:** N/A
- **Shared by [optional]:** Amit Chaubey
- **Model type:** Causal Language Model (LLM), LoRA fine-tuned, 4-bit quantized , nf4
- **Language(s) (NLP):** English
- **License:** MIT (or the license of the base model, if different)
- **Finetuned from model [optional]:** microsoft/Phi-3-mini-4k-instruct

### Model Sources

- **Repository:** https://github.com/amit-chaubey/finetune-LoRA-4bit
- **Paper [optional]:** https://arxiv.org/abs/2106.09685 (LoRA)
- **Demo [optional]:** [More Information Needed]

---

## Uses

### Direct Use

- Supervised learning
- Instruction following
- Educational and research purposes

### Downstream Use

- Can be further fine-tuned for domain-specific tasks (e.g., summarization, Q&A)
- Integration into chatbots or virtual assistants

### Out-of-Scope Use

- Not suitable for real-time safety-critical applications
- Not intended for generating harmful, biased, or misleading content

---

## Bias, Risks, and Limitations

- The model may reflect biases present in the training data.
- Not suitable for sensitive or high-stakes decision-making.
- Outputs should be reviewed by humans before use in production.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Always validate outputs for your use case.

---

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_adapter)
tokenizer = AutoTokenizer.from_pretrained(base_model)

prompt = "The weather is good today."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Training Details

### Training Data

- Used Own dataset

### Training Procedure

- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Target modules: ["query_key_value", "o_proj", "qkv_proj", "gate_up_proj", "down_proj"]
- 4-bit quantization (nf4)

#### Speeds, Sizes, Times [optional]

- Model size: ~3.8B parameters (0.33% trainable)
- Training time: Varies by hardware (e.g., 1-2 hours on A100 GPU for small datasets)

---

## Evaluation

### Testing Data, Factors & Metrics

- Used a held-out portion of the training dataset for evaluation.
- Metrics: Perplexity, qualitative review of generated outputs.

### Results

- The model demonstrates strong instruction-following and text generation capabilities after LoRA fine-tuning.

---

## Environmental Impact

- **Hardware Type:** NVIDIA A100 GPU
- **Hours used:** ~2
- **Cloud Provider:** [Your Cloud Provider]
- **Compute Region:** [Your Region]
- **Carbon Emitted:** Estimate using [ML CO2 Impact calculator](https://mlco2.github.io/impact#compute)

---

## Technical Specifications

### Model Architecture and Objective

- Base: Phi-3-mini-4k-instruct (Causal LM)
- LoRA fine-tuning with PEFT
- 4-bit quantization for efficiency


