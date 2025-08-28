import os
from dotenv import load_dotenv

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv()

# ============ Langfuse setup ============
langfuse = Langfuse(
    public_key=os.environ.get('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.environ.get('LANGFUSE_SECRET_KEY'),
    host=os.environ.get('LANGFUSE_HOST')
)
langfuse_handler = CallbackHandler()

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: OverallState):
    return {"answer": "bye", "notes": "... his name is Lance"}

def answer_node(state: OverallState):
    return {"answer": "bye Lance"}

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node("thinking_node", thinking_node)
builder.add_node("answer_node", answer_node)
builder.add_edge(START, "thinking_node")
builder.add_edge("thinking_node", "answer_node")
builder.add_edge("answer_node", END)

graph = builder.compile()
config = {"callbacks": [langfuse_handler]}
print(graph.invoke({"question": "hi"}, config=config))