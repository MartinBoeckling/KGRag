from models.vanilla_llm.vanilla_llm_model import VanillaLLM
from models.vanilla_rag.vanilla_rag_model import VanillaRag
from models.walk_retrieve.rag_model import WalkRetrieveRAG

def UserModel():
    return VanillaLLM()
    # return VanillaRag()
    # return WalkRetrieveRAG()