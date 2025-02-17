from dotenv import load_dotenv
import os
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict
from uuid import UUID
from tqdm.auto import tqdm

class BatchCallback(BaseCallbackHandler):
	def __init__(self, total: int):
		super().__init__()
		self.count = 0
		self.progress_bar = tqdm(total=total) # define a progress bar

	# Override on_llm_end method. This is called after every response from LLM
	def on_llm_end(self, response, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
		self.count += 1
		self.progress_bar.update(1)

load_dotenv(".secrets")

openai_key = os.getenv("OPENAI_KEY")

# Prameters of Walk and retrieve model runs
NODE_RETRIEVAL = 3
PATH_RETRIEVAL = 3
MODEL_NAME = "llama3.1:70b"
document_embedding_path = ""
node_embedding_path = ""