from prompt.prompt_collection import question_prompt
from constants import MODEL_NAME, document_embedding_path, PATH_RETRIEVAL, NODE_RETRIEVAL
import pandas as pd
import numpy as np
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity


class WalkRetrieveRAG:
    """
    A Retrieval-Augmented Generation (RAG) system that enhances query responses by retrieving relevant 
    knowledge graph walks and document embeddings before generating answers.

    This class retrieves relevant nodes based on semantic similarity between the query embedding and 
    stored node embeddings. It then retrieves associated triples (documents) from the knowledge graph
    and uses a large language model (LLM) to generate responses.
    """
    def __init__(self):
        pass

    def get_batch_size(self) -> int:
        """
        Returns the batch size for processing.

        Defines the maximum number of queries that can be processed in a batch.

        Returns:
            int: The predefined batch size (200).
        """
        return 200

    def batch_generate_answer(self, batch_elements: list) -> list:
        """
        Generates answers for a batch of input queries using a hierarchical Retrieval-Augmented Generation (RAG) approach.

        This method first retrieves relevant nodes based on semantic similarity between the query embedding 
        and stored node embeddings. Then, it retrieves relevant triples (documents) associated with these 
        nodes before generating responses using a large language model (LLM).

        Args:
            batch_elements (list): A list of queries, where each query is under the key `"query"`.

        Returns:
            list: A list of generated responses corresponding to the input queries.
        """
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
        
        ollama_emb_model = OllamaEmbeddings(
            model=MODEL_NAME,
            show_progress=True
        )
        triple_embeddings_document = pd.read_parquet(document_embedding_path)
        
        node_embedding_document = pd.read_parquet(document_embedding_path)
        node_embeddings = node_embedding_document.filter(like="embedding_", axis=1)
        responses = []
        for query in batch_elements["query"]:
            query_embedding = ollama_emb_model.embed_query(query)
            sim_nodes = cosine_similarity([query_embedding], node_embeddings)
            sim_nodes = sim_nodes[0]
            relevant_nodes = sim_nodes[np.argsort(sim_nodes)[-NODE_RETRIEVAL:]].tolist()
            triple_embeddings_document = triple_embeddings_document[triple_embeddings_document["node_id"].isin(relevant_nodes)]
            triple_embeddings = triple_embeddings_document.filter(like="embedding_", axis=1)
            sim_documents = cosine_similarity([query_embedding], triple_embeddings)
            sim_documents = sim_documents[0]
            relevant_index = np.argsort(sim_documents)[-PATH_RETRIEVAL:].tolist()
            retrieved_documents = triple_embeddings_document.iloc[relevant_index]
            rag_document = retrieved_documents["corpus"].str.cat(sep="; ")
            rag_query = question_prompt.format_prompt(question=query, context=rag_document)
            response = chat_model.invoke(rag_query)
            responses.append(response.content)
        return responses
