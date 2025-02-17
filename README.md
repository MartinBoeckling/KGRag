# KGRag
This repository contains the relevant code for the paper **Walk&Retrieve: Leveraging Knowledge Graph Walks for Zero-Shot Retrieval-Augmented Generation**. 

If you encounter any issues or have questions about the implementation, feel free to open an issue in this repository.

To get started, install the required dependencies using the provided [requirements.txt file](requirements.txt).

```bash
pip install -r requirements.txt
```

## Data
The datasets used in this research can be obtained from:
- [CRAG repository](https://github.com/facebookresearch/CRAG) 
- [MetaQA repository](https://github.com/yuyuz/MetaQA)
For MetaQA, all hops (1-hop, 2-hop, and 3-hop) are relevant for evaluating our method.
## Setup
To run the experiments successfully, please follow the folder structure outlined below. Ensuring the correct structure helps maintain relative path consistency in the scripts. Please run the following [bash script](start_ollama_docker.sh) to start the Ollama container necessary for the Llama 3.1 LLM as well as the Mistral LLM.
## Folder Structure
In the following the folder structure of our project is visible. Please make sure to recreate the folder structure to make sure relative paths work within the script.
root
├── data
│   ├── CRAG
│   │   ├── dataset
│   │   ├── model_answers
|   |   |── rdf_corpus
│   │   └── triple_corpus
│   └── MetaQA
│       ├── 1-hop
│       ├── 2-hop
│       ├── 3-hop
│       ├── model_answers
│       ├── rdf_corpus
│       └── triple_corpus
├── data_preparation
├── models
│   ├── ollama_model_state
│   ├── vanilla_llm
│   ├── vanilla_rag
│   └── walk_retrieve
└── prompt
## Data Preparation
To prepare the datasets for evaluation, we provide scripts to transform the data into the required format.
* [Crag Triple Generation](data_preparation/metaqa_kg_prep.py): This script generates a knowledge graph (KG) based on the CRAG repository’s mock API, enabling walk retrieval over the graph.
* [MetaQA KG Preparation](data_preparation/metaqa_kg_prep.py): This script transforms the MetaQA KG into the format required by our code.
## Models
This repository includes the implementations of both our Walk&Retrieve method and baseline approaches. For external baselines, follow their respective GitHub repositories:
- [SubgraphRAG](https://github.com/Graph-COM/SubgraphRAG)
- [Rewrite-Retrieve-Answer](https://github.com/wuyike2000/Retrieve-Rewrite-Answer)
### Vanilla LLM
In [this folder](models/vanilla_llm) we provide the implementation of the Vanilla LLM solution allowing to simply ask questions agains a Vanilla LLM instance.
### Vanilla RAG
In [this folder](models/vanilla_rag) we provide the necessary scripts to run the Vanilla RAG approach.

### Walk and Retrieve
In [this folder](models/vanilla_rag) we provide the necessary scripts to run the Walk and retrieve approach.
## Prompt Collection
In the [following script](prompt/prompt_collection.py) we provide the collection of our prompts used throughout different stages of our code.