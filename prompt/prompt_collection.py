from langchain.prompts import ChatPromptTemplate
corpus_construction_prompt = ChatPromptTemplate([
    ("system", "Please provide me from an extracted triple set of a Knowledge Graph a sentence. The triple set consists of one extracted random walk. Therefore, a logical order of the shown triples is present. Please consider this fact when constructing the sentence. Prevent introduction words:"),
    ("human", "Please return only the constructed sentence from the following set of node and edge labels extracted from the Knowledge Graph: {triples}")
])

question_prompt = ChatPromptTemplate([
    ("system", "You are provided with context information from a RAG retrieval, which gives you the top n context information. Please use the provided context information to answer the question. If your are not able to answer the question based on the context information, please return the following sentence: 'I do not know the answer'"),
    ("human", "Please answer the following question: {question}. Use the following context information to answer the question: {context}")
])
