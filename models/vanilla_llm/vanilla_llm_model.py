from prompt.prompt_collection import vanilla_question_prompt
from constants import BatchCallback, MODEL_NAME
from langchain_community.chat_models.ollama import ChatOllama
from langchain_openai import ChatOpenAI

class VanillaLLM:
    """
    A class for generating answers using a vanilla LLM (Large Language Model) approach.

    This class provides methods to determine the batch size for processing and to generate 
    answers in batches using either the OpenAI `gpt-4o` model or an `Ollama` model. The model 
    selection depends on the value of `MODEL_NAME`.
    """
    def __init__(self):
        pass

    def get_batch_size(self) -> int:
        """
        Returns the batch size for processing.

        This method defines the maximum number of queries that can be processed in a batch.

        Returns:
            int: The predefined batch size (200).
        """
        return 200

    def batch_generate_answer(self, batch_elements: list) -> list:
        """
        Generates answers for a batch of input queries using the specified LLM.

        This method selects the appropriate model (`gpt-4o` or `Ollama`) based on `MODEL_NAME`,
        formats the queries using a predefined prompt template, and generates responses 
        asynchronously in a batch.

        Args:
            batch_elements (list): A list containing queries under the key `"query"`.

        Returns:
            list: A list of generated responses corresponding to the input queries.
        """
        # initialize model
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
        tasks = [vanilla_question_prompt.format_prompt(question=walk) for walk in batch_elements["query"]]
        cb = BatchCallback(len(tasks))
        responses = chat_model.batch(tasks, config = {"callbacks": [cb]})
        responses = [response.content for response in responses]
        return responses
