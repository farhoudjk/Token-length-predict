# Research Questions and Background Summary

## RQ1 — Effect of Model Configuration on Output Token Length

### **Research Question**
**RQ1:** *How do different LLM configurations (full-precision instruct, base, and quantized models) influence output token-length distributions and efficiency under a controlled multi-domain workload?*

### **Background & Motivation**
LLMs exhibit widely different verbosity behaviors depending on training regime and precision. Through a controlled workload of **2,500 prompts** across multiple domains, we observed substantial variation in output token lengths:

- **Full-precision Instruct models (Llama 3.1 8B Instruct, Llama 2 8B Instruct)** generated extremely long responses, with **~90–95%** of outputs filling the full 2000–4096 token range.
- **Base models** produced inconsistent behavior, including many extremely short outputs (<9 tokens) due to sensitivity to long input prompts.
- **Quantized models** exhibited **high natural diversity**, with **~80–88%** of outputs falling in the useful 10–2000 token range, avoiding both pathological short outputs and excessive verbosity.
- **Mistral 7B** and **DeepSeek 7B** showed intermediate behaviors, but still tended toward over-generation or under-generation compared to quantized models.

### **Interpretation**
The quantized model’s balanced output distribution arises from the effects of quantization:
- Reduced logit precision dampens long-tail continuation probabilities.
- Increased uncertainty leads to earlier EOS emission.
- Quantization implicitly "regularizes verbosity", preventing runaway generation.

Thus, **quantized models are the most suitable for downstream analysis** of token-length prediction and feature correlations, due to their stable and diverse output-length behavior.

---

## RQ2 — Feature Selection for Predicting Output Token Length

### **Research Question**
**RQ2:** *Which prompt-level features most strongly predict output token length for quantized LLMs, and how does feature importance vary across prompt categories and difficulty levels?*

### **Motivation**
Given the stable and diverse output-length signals produced by quantized models, we aim to build predictive models that estimate expected output length based on prompt characteristics. Such predictions have value for:
- **Latent cost estimation** in serving systems.
- **Token budgeting and scheduling** for parallel inference.
- **Understanding prompt–response relationships** in model behavior.

### **Feature Families Considered**
Features extracted from each prompt include:

#### **1. Structural Features**
- Input token length
- Number of sentences / words
- Presence of list structures
- Code blocks, math indicators

#### **2. Syntactic Features**
- Presence of question marks
- Punctuation density
- Wh-word indicators (why/how vs who/when)

#### **3. Semantic / Metadata Features**
- Domain (factual, writing, coding, reasoning, professional)
- Difficulty (easy/medium/hard)
- Prompt length type (short/medium/long)

#### **4. Lexical Features**
- Unique word count
- Type-token ratio
- TF-IDF signal

### **Modeling Approach**
To identify feature importance and predict token length, models considered include:
- Linear regression / L1-regularized regression (interpretable coefficients)
- Tree-based models such as Random Forest or XGBoost (non-linear importances)
- SHAP-based interpretability for global and local attributions

### **Outcome Expectation**
Quantized models provide a predictable, non-pathological token-length distribution suitable for learning. This RQ aims to:
- Identify the strongest predictors of verbosity
- Compare feature importance across domains
- Build token-length prediction models that may later be used for scheduling or adaptive generation

---

## Conclusion
RQ1 establishes that quantized models provide the most informative variation in output token lengths for analysis. RQ2 builds on this by investigating which prompt-level features influence these output lengths, enabling predictive modeling and deeper understanding of prompt–response dynamics in LLMs.