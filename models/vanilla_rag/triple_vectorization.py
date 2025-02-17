from pathlib import Path
from langchain_community.embeddings.ollama import OllamaEmbeddings
import pandas as pd
from constants import MODEL_NAME

def vanilla_rag_embedding(data_path: str, output_path: str) -> None:
    """Generates embeddings for triples using a vanilla RAG (Retrieval-Augmented Generation) approach.

    This function reads a dataset containing knowledge graph triples from a parquet file,
    constructs textual representations of each triple by concatenating the subject, predicate, 
    and object, and then computes embeddings using the `OllamaEmbeddings` model. The resulting
    embeddings are saved in a parquet file.

    Args:
        data_path (str): Path to the input parquet file containing knowledge graph triples.
        output_path (str): Directory path where the computed embeddings should be saved.

    Outputs:
        - A parquet file containing embeddings for each triple, saved as 
          `vanillaRAG_embedding_<parameters>.parquet`.

    Processing Steps:
        1. Read the input parquet file containing knowledge graph triples.
        2. Construct a textual representation for each triple by concatenating the `subject`, 
           `predicate`, and `object` fields.
        3. Compute embeddings for the generated text representations using `OllamaEmbeddings`.
        4. Save the resulting embeddings as a parquet file.

    The output filename is determined based on the input file name structure, extracting
    parameters from the file name.
    """
    data_representation_path = Path(data_path)
    parameters = "_".join(data_representation_path.stem.split("_")[1:])
    ollama_emb_model = OllamaEmbeddings(
        model=MODEL_NAME,
        show_progress=True
    )
    kg_corpus = pd.read_parquet(data_path)
    kg_corpus["corpus"] = kg_corpus["subject"] + " " + kg_corpus["predicate"]  + " " + kg_corpus["object"]
    triple_corpus_list = kg_corpus["corpus"].to_list()
    triple_vector_representation = ollama_emb_model.embed_documents(triple_corpus_list)
    column_names = [f"embedding_{i}" for i in range(triple_vector_representation.shape[1])]
    triple_vector_representation = pd.DataFrame(triple_vector_representation, columns=column_names)
    triple_vector_representation = pd.concat([kg_corpus, triple_vector_representation], axis=1)
    triple_vector_representation.to_parquet(f"{output_path}/vanillaRAG_embedding_{parameters}.parquet")