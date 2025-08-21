from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtracts b from a.

    Args:
        a: first int
        b: second int
    """
    return a - b

def divide(a: int, b: int) -> float:
    """Divides a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [multiply, add, subtract, divide]
# LLM with bound tool
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools=tools)

# System message
sys_message = SystemMessage(content="You are a helpful assistant that can perform arithmetic operations using tools. You can multiply, add, subtract, and divide numbers. Use the tools when necessary.")

# Node
def arithmetic_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("arithmetic_llm", arithmetic_llm)
builder.add_node("tools", ToolNode(tools=tools))
builder.add_edge(START, "arithmetic_llm")
builder.add_conditional_edges(
    "arithmetic_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "arithmetic_llm")

# Compile graph
graph = builder.compile()
messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Subtract 4 and divide the output by 5")]
messages = graph.invoke({"messages": messages}) 

for mes in messages["messages"]:
    mes.pretty_print()