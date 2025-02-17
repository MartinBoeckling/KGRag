import pickle
import pandas as pd
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    AzureMLChatOnlineEndpoint,
	CustomOpenAIChatContentFormatter
)
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from prompt.prompt_collection import corpus_construction_prompt
from constants import BatchCallback


rate_limitation = InMemoryRateLimiter(
    requests_per_second=10,  
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

chat_model = AzureMLChatOnlineEndpoint(
    endpoint_url="https://Meta-Llama-3-1-70B-Instruct-fzwu.swedencentral.models.ai.azure.com/chat/completions",
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key="xty3mdKWCanNj08QW5XzW2Da8VLd8xsC",
    content_formatter=CustomOpenAIChatContentFormatter(),
    rate_limiter = rate_limitation,
    max_retries=5
)



with open("data/MetaQA/rdf_corpus/corpus_bfs_4_100.pkl", "rb") as file:
    kg_corpus = pickle.load(file)

pd_kg_corpus = pd.DataFrame(kg_corpus, columns=["node_id", "extracted_triple"])
triple_set = pd_kg_corpus["extracted_triple"].to_list()

tasks = [corpus_construction_prompt.format_prompt(triples=walk) for walk in triple_set]

cb = BatchCallback(len(tasks))
response_list = []
for index, response in chat_model.batch_as_completed(tasks, config = {"callbacks": [cb]}):
    response_list.append((index, response.content))

pd_response = pd.DataFrame(response_list, columns=["index", "verbalized_triple"])
pd_response = pd_response.set_index(keys="index")
pd_kg_corpus = pd.merge(pd_response, pd_kg_corpus, how="inner", left_index=True, right_index=True)

pd_kg_corpus.to_parquet("data/MetaQA/triple_corpus/corpus_bfs_4_100.parquet")

# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
# from langchain.prompts import ChatPromptTemplate
# from langchain_huggingface import HuggingFacePipeline
# from prompt.prompt_collection import corpus_construction_prompt
# # import torch
# import pickle
# from constants import BatchCallback, llama_api_key, groq_api_key
# import pandas as pd
# # from langchain.llms import OpenAI
# # from langchain_community.chat_models import ChatOpenAI
# import mlflow
# from pathlib import Path
# import os
# # from transformers.utils import logging
# import getpass
# from langchain_openai import AzureChatOpenAI
# if not os.environ.get("AZURE_OPENAI_API_KEY"):
#   os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")
# model = AzureChatOpenAI(
#     azure_endpoint="https://Meta-Llama-3-1-70B-Instruct-fzwu.swedencentral.models.ai.azure.com/chat/completions",
#     azure_deployment="Meta-Llama-3-1-70B-Instruct-fzwu",
#     openai_api_version="2024-02-15-preview",
# )
# response = model.invoke("Hello, world!")
# print(response)
# logging.set_verbosity_error() 
# os.environ["GROQ_API_KEY"] = groq_api_key
# # model_id = "meta-llama/Llama-3.1-70B"
# model_id = "llama3-70b-8192"

# with open("data/MetaQA/rdf_corpus/corpus_bfs_1_100.pkl", "rb") as file:
#     kg_corpus = pickle.load(file)

# pd_kg_corpus = pd.DataFrame(kg_corpus, columns=["node_id", "extracted_triple"])
# triple_set = pd_kg_corpus["extracted_triple"].to_list()


# print(triple_set[0])
# print(responses[0])
# tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="models/huggingface/")
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map = "auto",
#     torch_dtype=torch.float16, 
#     cache_dir="models/huggingface/"
# )
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=256,
#     device_map="auto"
# )
# hf_pipeline = HuggingFacePipeline(pipeline=pipe)
# chat_model = ChatHuggingFace(llm=hf_pipeline)

# tasks = [corpus_construction_prompt.format_prompt(triples=walk) for walk in triple_set]
# cb = BatchCallback(len(tasks))
# responses = chat_model.batch(tasks, config = {"callbacks": [cb]})
# responses = [response.content for response in responses]
# import mlflow
# mlflow.set_tracking_uri("127.0.0.1:5000")
# mlflow.set_experiment("Triple construction")
# # corpus_data = Path("data/MetaQA/rdf_corpus").glob("corpus*")
# corpus_path = Path("data/MetaQA/rdf_corpus/corpus_bfs_4_100.pkl")
# parameters = corpus_path.stem.split("_")
# mlflow.langchain.autolog(silent=True, log_traces=True,
#     log_models=False)
# with mlflow.start_run() as run:
#     mlflow.log_params({"walk_method": parameters[1], "walk_distance": parameters[2], "walk_number": parameters[3]})
#     with open(corpus_path, "rb") as file:
#         kg_corpus = pickle.load(file)

#     pd_kg_corpus = pd.DataFrame(kg_corpus, columns=["node_id", "extracted_triple"])
#     triple_set = pd_kg_corpus["extracted_triple"].to_list()
#     tasks = [corpus_construction_prompt.format_prompt(triples=walk) for walk in triple_set]

#     chat_model = ChatOpenAI(
#         api_key = llama_api_key,
#         base_url = "https://api.llama-api.com",
#         model_name = "llama3.1-70b"
#     )


#     cb = BatchCallback(len(tasks))
#     responses = chat_model.batch(tasks, config = {"callbacks": [cb]})
#     responses = [response.content for response in responses]
# pd_kg_corpus["verbalized_triple"] = responses
# pd_kg_corpus.to_parquet("data/MetaQA/triple_corpus/corpus_bfs_4_100.parquet")

