from pprint import pprint
import os, getpass
from typing_extensions import TypedDict
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

load_dotenv()

messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.",name="Lance"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

# for m in messages:
#     m.pretty_print()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0
)
# result = llm.invoke(messages)
# print('==========llm response==========')
# print(result.content)
# print('==========response_metadata==========')
# print(result.response_metadata)

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])
tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 11 multiplied by 7", name="Lance")])
print('==============tool call================')
print(tool_call.tool_calls)

# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

# class MessagesState(MessagesState):
#     # Add any keys needed beyond messages, which is pre-built 
#     pass

# # Initial state
# initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
#                     HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
#                    ]

# # New message to add
# new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# # Test
# print(add_messages(initial_messages , new_message))
# print('===============')
# print(type(add_messages(initial_messages , new_message)))

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)   
graph = builder.compile()

messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    m.pretty_print()

messages = graph.invoke({"messages": HumanMessage(content="Multiply 11 and 7")})
for m in messages['messages']:
    m.pretty_print()