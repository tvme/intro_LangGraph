from dotenv import load_dotenv
import os
import random
from typing import Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

# загружаем переменные окружения
load_dotenv()
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
TAVILY_API_KEY=os.environ.get('TAVILY_API_KEY')

gpt4o_chat = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0
)


class State(TypedDict):
    graph_state: str

def node_1(state: State) -> State:
    print("----Node 1----")
    return {"graph_state": state["graph_state"] + " I am"}

def node_2(state: State) -> State:
    print("----Node 2----")
    return {"graph_state": state["graph_state"] + " very happy!"}

def node_3(state: State) -> State:
    print("----Node 3----")
    return {"graph_state": state["graph_state"] + " so sad!"}

def decide_mood(state: State) -> Literal["node_2", "node_3"]:
    user_input = state["graph_state"]

    if random.random() < 0.5:
        return "node_2"
    else:
        return "node_3"

builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()
result = graph.invoke({"graph_state": "Hi, this is Alex."})
print(f"Graph result: {result['graph_state']}")