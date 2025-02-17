from pathlib import Path
from langchain_community.embeddings.ollama import OllamaEmbeddings
import pandas as pd
from constants import MODEL_NAME

def embedding_verbalized_triple(data_path: str, output_path: str) -> None:
    """Generates embeddings for verbalized triples and saves them in parquet format.

    This function reads a dataset of verbalized triples from a parquet file, computes
    embeddings using the `OllamaEmbeddings` model, and saves the results in parquet format.
    The embeddings are computed both at the node level (grouped by `node_id`) and at the 
    individual triple level.

    Args:
        data_path (str): Path to verbalized triple data in parquet format
        output_path (str): Path to save the output of the embedding of the verbalized triple data

    Processing Steps:
        1. Read the input parquet file containing verbalized triples.
        2. Group verbalized triples by `node_id` and concatenate them into a single string.
        3. Compute embeddings for the grouped node-level representations.
        4. Compute embeddings for individual verbalized triples.
        5. Save both embeddings as separate parquet files.

    The output filenames are determined based on the input file name structure, using 
    the extracted parameters from the file name.
    """

    data_representation_path = Path(data_path)
    parameters = "_".join(data_representation_path.stem.split("_")[1:])
    ollama_emb_model = OllamaEmbeddings(
        model=MODEL_NAME,
        show_progress=True
    )
    kg_corpus = pd.read_parquet(data_path)
    grouped_kg_corpus = kg_corpus.groupby("node_id")["verbalized_triple"].apply(list).reset_index()
    grouped_kg_corpus["verbalized_triple"] = grouped_kg_corpus["verbalized_triple"].apply(lambda x: " ".join(x))
    node_document_representation = grouped_kg_corpus["verbalized_triple"].to_list()
    node_vector_representation = ollama_emb_model.embed_documents(node_document_representation)
    column_names = [f"embedding_{i}" for i in range(node_vector_representation.shape[1])]
    node_vector_representation = pd.DataFrame(node_vector_representation, columns=column_names)
    node_vector_representation["node_id"] = grouped_kg_corpus["node_id"]
    node_vector_representation.to_parquet(f"{output_path}/node_embedding_{parameters}.parquet")
    # Individual triple representation
    individual_triple_representation = kg_corpus["verbalized_triple"].to_list()
    triple_vector_representation = ollama_emb_model.embed_documents(individual_triple_representation)
    column_names = [f"embedding_{i}" for i in range(triple_vector_representation.shape[1])]
    triple_vector_representation = pd.DataFrame(triple_vector_representation, columns=column_names)
    triple_vector_representation.to_parquet(f"{output_path}/triple_embedding_{parameters}.parquet")
