# Faster Scientific Discovery using a reduced-size LM-JEPA for Symbolic Regression.

This repository contains the prototype and screening test implementation for my ML4SCI GSoC proposal.

### Tokenization Rationale 
To preprocess `FeynmanEquations.csv`, I mapped the formulas into strictly **binary prefix notation trees** using SymPy's AST parser (e.g., forcing `x + y + z` into `add(x, add(y, z))`).

**Rationale:**
1. **No Vocabulary Explosion:** By breaking down complex polynomial expansions into fundamental binary operators (`add`, `mul`, `pow`), the model's output vocabulary size is permanently constrained to a highly compact footprint (~40 tokens). This prevents dynamic variables and constants from exploding the generation space.
2. **Grammar Enforcement:** Enforcing a strict prefix structure ensures the Autoregressive Decoder learns the exact grammatical arity of mathematical operators, substantially reducing physically invalid "hallucinated" structures during evaluation.

### Current Evaluation Status 
The JEPA latent representation (in pretraining) efficiently converges to a loss of `~0.015`. 

Currently, the causal finetuning loss plateaus around `3.2` when training on the 100 benchmark Feynman equations. This is expected, as training a mathematics Transformer from scratch requires a massive volume of sequence tokens to generalize.
