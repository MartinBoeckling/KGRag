from prompt.prompt_collection import question_prompt
from constants import MODEL_NAME, document_embedding_path, PATH_RETRIEVAL
import pandas as pd
import numpy as np
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity

class VanillaRAG:
    """
    A class for performing Retrieval-Augmented Generation (RAG) using a vanilla LLM approach.

    This class retrieves relevant documents based on semantic similarity using embeddings,
    then generates responses by augmenting the query with retrieved context before passing
    it to a large language model (LLM). The model selection depends on the `MODEL_NAME` parameter.
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
        Generates answers for a batch of input queries using Retrieval-Augmented Generation (RAG).

        This method retrieves relevant documents based on cosine similarity between the query
        embedding and stored triple embeddings, then augments the query with the retrieved
        context before generating a response using an LLM.

        Args:
            batch_elements (list): A list containing queries under the key `"query"`.

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
        triple_embeddings = triple_embeddings_document.filter(like="embedding_", axis=1)
        responses = []
        for query in batch_elements["query"]:
            query_embedding = ollama_emb_model.embed_query(query)
            sim_documents = cosine_similarity([query_embedding], triple_embeddings)
            sim_documents = sim_documents[0]
            relevant_index = np.argsort(sim_documents)[-PATH_RETRIEVAL:].tolist()
            retrieved_documents = triple_embeddings_document.iloc[relevant_index]
            rag_document = retrieved_documents["corpus"].str.cat(sep="; ")
            rag_query = question_prompt.format_prompt(question=query, context=rag_document)
            response = chat_model.invoke(rag_query)
            responses.append(response.content)
        return responses