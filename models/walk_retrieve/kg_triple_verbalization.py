from pathlib import Path
from langchain_community.embeddings.ollama import OllamaEmbeddings
import pandas as pd
import pickle
from constants import MODEL_NAME, BatchCallback
from langchain_community.chat_models.ollama import ChatOllama
from langchain_openai import ChatOpenAI
from prompt.prompt_collection import corpus_construction_prompt


def triple_verbalization(walk_corpus_path: str, output_path_folder: str) -> None:
    """_summary_

    Args:
        walk_corpus_path (str): _description_
    """
    data_representation_path = Path(walk_corpus_path)
    parameters = "_".join(data_representation_path.stem.split("_")[1:])
    with open(walk_corpus_path, "rb") as file:
        kg_corpus = pickle.load(file)
    pd_kg_corpus = pd.DataFrame(kg_corpus, columns=["node_id", "extracted_triple"])
    triple_set = pd_kg_corpus["extracted_triple"].to_list()
    tasks = [corpus_construction_prompt.format_prompt(triples=walk) for walk in triple_set]
    if MODEL_NAME == "gpt-4o":
        chat_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    else:
        chat_model = ChatOllama(
            model=MODEL_NAME,
            temperature=0
        )
    cb = BatchCallback(len(tasks))
    response_list = []
    checkpoint_iteration = 5000
    iteration = 0
    for index, response in chat_model.batch_as_completed(tasks, config = {"callbacks": [cb]}):
        response_list.append((index, response.content))
        if checkpoint_iteration % iteration == 0:
            with open(f"checkpoint_response_list.pkl", "wb") as file:
                print(iteration)
                pickle.dump(response_list, file)
        iteration += 1

    pd_response = pd.DataFrame(response_list, columns=["index", "verbalized_triple"])
    pd_response = pd_response.set_index(keys="index")
    pd_kg_corpus = pd.merge(pd_response, pd_kg_corpus, how="inner", left_index=True, right_index=True)
    pd_kg_corpus.to_parquet(f"{output_path_folder}/verbalized_triple_{parameters}.parquet")

