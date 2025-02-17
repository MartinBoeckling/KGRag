from langchain.prompts import ChatPromptTemplate

corpus_construction_prompt = ChatPromptTemplate([
    ("system", "Please provide me from an extracted triple set of a Knowledge Graph a sentence. The triple set consists of one extracted random walk. Therefore, a logical order of the shown triples is present. Please consider this fact when constructing the sentence. Prevent introduction words:"),
    ("human", "Please return only the constructed sentence from the following set of node and edge labels extracted from the Knowledge Graph: {triples}")
])

question_prompt = ChatPromptTemplate([
    ("system", "You are provided with context information from a RAG retrieval, which gives you the top n context information. Please use the provided context information to answer the question. If your are not able to answer the question based on the context information, please return the following sentence: 'I do not know the answer'"),
    ("human", "Please answer the following question: {question}. Use the following context information to answer the question: {context}")
])

vanilla_question_prompt = ChatPromptTemplate([
    ("system", "You are a useful assistant. If you are not sure, please return 'I do not know the answer'"),
    ("human", "Please answer the following question: {question}.")
])

INSTRUCTIONS = """
    # Task: 
    You are given a Question, a model Prediction, and a list of Ground Truth answers, judge whether the model Prediction matches any answer from the list of Ground Truth answers. Follow the instructions step by step to make a judgement. 
    1. If the model prediction matches any provided answers from the Ground Truth Answer list, "Accuracy" should be "True"; otherwise, "Accuracy" should be "False".
    2. If the model prediction says that it couldn't answer the question or it doesn't have enough information, "Accuracy" should always be "False".
    3. If the Ground Truth is "invalid question", "Accuracy" is "True" only if the model prediction is exactly "invalid question".
    4. Determine the Hits@1 accuracy of the model by comparing the model prediction with the first element relevant to answer the question.
    # Output: 
    Respond with only JSON string with an "Accuracy" field which is "True" or "False" and a second key with a "Hits1" field.
"""

IN_CONTEXT_EXAMPLES = """
    # Examples:
    Question: how many seconds is 3 minutes 15 seconds?
    Ground truth: ["195 seconds"]
    Prediction: 3 minutes 15 seconds is 195 seconds.
    Accuracy: True

    Question: Who authored The Taming of the Shrew (published in 2002)?
    Ground truth: ["William Shakespeare", "Roma Gill"]
    Prediction: The author to The Taming of the Shrew is Roma Shakespeare.
    Accuracy: False

    Question: Who played Sheldon in Big Bang Theory?
    Ground truth: ["Jim Parsons", "Iain Armitage"]
    Prediction: I am sorry I don't know.
    Accuracy: False
"""
