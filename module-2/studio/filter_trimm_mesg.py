import os
from dotenv import load_dotenv

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from typing_extensions import TypedDict
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import RemoveMessage, trim_messages
from langchain_openai import ChatOpenAI


load_dotenv()

# ============ Langfuse setup ============
langfuse = Langfuse(
    public_key=os.environ.get('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.environ.get('LANGFUSE_SECRET_KEY'),
    host=os.environ.get('LANGFUSE_HOST')
)
langfuse_handler = CallbackHandler()


llm = ChatOpenAI(model="gpt-4o-mini")

def filter_messages(state: MessagesState):
    # delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}   

def chat_model_node(state: MessagesState):
    trimmed_messages = trim_messages(state["messages"],
                             max_tokens=100,
                             strategy="last",
                             token_counter=ChatOpenAI(model="gpt-4o-mini"),
                             allow_partial=False)
    return {"messages": llm.invoke(trimmed_messages)}

builder = StateGraph(MessagesState)
# builder.add_node("filter_messages", filter_messages)
builder.add_node("chat_model_node", chat_model_node)
builder.add_edge(START, "chat_model_node")
# builder.add_edge("filter_messages", "chat_model_node")
builder.add_edge("chat_model_node", END)

graph = builder.compile()

messages = [AIMessage(f"So you said you were researching ocean mammals?", name="Bot", id="1")]
messages.append(HumanMessage(f"Yes, I know about whales. But what others should I learn about?", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))


config = {"callbacks": [langfuse_handler]}
output = graph.invoke({"messages": messages}, config=config)
messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me where Orcas live!", name="Lance"))
print(trim_messages(messages,
                    max_tokens=50,
                    strategy="last",
                    token_counter=ChatOpenAI(model="gpt-4o-mini"),
                    allow_partial=False))

output = graph.invoke({"messages": messages}, config=config)

for m in output["messages"]:
    m.pretty_print()
