<h1 align="center">Multilingual Blending: LLM Safety Alignment Evaluation with Language Mixture</h1>

 
This is the anonymous repository for our paper: *Multilingual Blending: LLM Safety Alignment Evaluation with Language Mixture*.

<div align="center">
  <img src="plots/sample_illustration.png" alt="workflow" width="650">
  <em>Multilingual-Blending Sample Illustration</em>
</div>

<br>

<div align="center">
  <img src="plots/workflow_illustration.png" alt="workflow" width="650">
    <em>Multilingual-Blending Workflow</em>

</div>

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## 👉 Installation

### Setting up Python Environment
1. Ensure you have Python 3.8+ installed.
2. Download this repository
3. Navigate to the project directory and set up a virtual environment:
   ```bash
   cd multilingual-blending
   conda create -n env_name python=3.8
4. Activate the virtual environment:
   ```bash
   conda activate env_name
5. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
## 🛠️ Usage
The `multilingual-blending` folder contains all the essential scripts for translation, LLM inference, and evaluation.

The `translation_utils.py` contains all the functions and classes used for multilingual translation and mixed-language transformation. 

The `llm_utils.py` contains all the functions and classes used for call LLMs from OpenAI API and Llama API.
Note: you need your own API key to infer the LLMs.

The `translatio_wrapper.py` and `inference_wrapper.py` provide a direct implementation for language translation and LLM inference. 
See the example below.

### 🚀 **Example**

### **1. Generating new malicious questions with designated languages**
To create malicious questions with your preferred language, you first need to identify the `code` of your language. You can search the language code according to [ISO 369-2](https://www.loc.gov/standards/iso639-2/) and confirm if the language is supported by the [Google Translation API](https://py-googletrans.readthedocs.io/en/latest/).

```python
python3 translatio_wrapper.py \
   --embedding_model all-MiniLM-L6-v2 \
   --lang_combinations List-of-your-language-code \
   --path_to_question dataset/mailcious_questions.csv \
   --sim_thres 0.9 \
   --iter_thres 30 
```
#### Argument Specification

`--embedding_model`: The name of the embedding model for semantic similarity calculation.

`--lang_combinations`: the list of language codes. For example, to translate the queries into a combination of [English, German, French], the input list should be `[en, de, fr]`.   The script also supports nested lists with various language combinations; namely, with an input of `[[ja, zh-cn, kr], [en, de, fr]],` the script will produce two transformed datasets w.r.t to the two combinations, respectively.

`--path_to_question`: Path to read the malicous questions. The default path leads to the 120 collectd samples used in our study. You are free to use your own data as long as you align the data format with default one.

`--sim_thres`: The threshold for semantic similarity checkup. Any transtions below this threshold will be rejeced and regenerated. 

`--iter_thres`: The maximum iterations allowed for translating one sample.

### **2. LLM inference and evaluation**
To pass the queries to the LLM and evaluate the generated content.

```python
python3 inference_wrapper.py \
   --embedding_model all-MiniLM-L6-v2 \
   --path_to_data path-tp-translated-data \
   --llm_model pname-of-the-model \
   --setting mix_to_mix \
   --open_ai_key Your OpenAI API \
   --PERSPECTIVE_API_KEY Your Perspective API KEY \
   --num_iter 1 
```
#### Argument Specification

`--embedding_model`: The name of the embedding model for semantic similarity calculation.

`--path_to_data`:Path to read the malicous questions. The default path leads to the 120 collectd samples used in our study. You are free to use your own data as long as you align the data format with default one.

`--llm_model`: name of the LLM. Our scripts support [OpenAI models](https://platform.openai.com/docs/models) as well as the models provided by [LLAMA API](https://www.llama-api.com/), such as llama3, Mistral and Qwen.

`--setting`: The type of mixed-language scheme.  `mix_to_en` stands for mixed-language queries with Engish responses, `en_to_mix` stands for English queries with mixed-language responses, `single` is for individual source languages and `mix_to_mix` is for mixed-language queries and mixed-language responses (Multilingual Blending).

`--open_ai_key`: You need your own OpenAI API key to access the models.

`--PERSPECTIVE_API_KEY`: You need your own Perspective API key to initiate the evaluator (it's free).

`--num_iter`: Number of inferecne for each query. THe default value is 1, but you change it for multi-time inferences.



## 📃 Results

All experiment results are available in the `dataset` directory,  categorized with different models and factors. 

