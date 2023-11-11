# FLLM_2023
Open sourced repository for experiments and implementations outlined in "Semantic Compression With Large Language Models"


## Installation
```bash
gh repo clone henrygilbert22/FLLM_2023
cd FLLM_2023
pip install -r requirements.txt
```

Create a .env file in the root directory and add your OpenAI API key as follows:
```bash
OPENAI_API_KEY=<your_api_key>
```

## Overview
This repository contains the code for the experiments and implementations outlined in "Semantic Compression With Large Language Models". The code is organized into the following directories:
- 'text_data' contains the source text data used for the experiments
- 'utils' contains helper functions for common functions and a OpenAI API wrapper
- 'experiment_data' contains the data generated from the experiments
- 'experiments notebooks' contains the self sufficient code for the experiments

## Experiments
1. base_compression_analysis.ipynb
    - This notebook provides the base compression performance of GPT4 without specified meta prompting. 
    - Compression performance is evaluated across edit distance, cosine similarity and compression ratio

2. prompted_compression_analysis.ipynb
    - This notebook provides the prompted compression performance of GPT4 and GPT3.05 with specified meta prompting for both lossless and semantic compression.
    - Compression performance is evaluated across edit distance, cosine similarity and compression ratio
    - Novel evaluation metrics, Semantic Reconstruction Effectiveness and Exact Reconstruction Effectiveness, are introduced and evaluated
