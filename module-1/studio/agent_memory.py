from dotenv import load_dotenv
import os
import inspect
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver



load_dotenv()
DB_USER=os.environ.get('DB_USER')
DB_PASSWORD=os.environ.get('DB_PASSWORD')
DB_HOST=os.environ.get('DB_HOST')
DB_PORT=os.environ.get('DB_PORT')
DB_NAME=os.environ.get('DB_NAME')

# ============ Langfuse setup ============
langfuse = Langfuse(
    public_key=os.environ.get('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.environ.get('LANGFUSE_SECRET_KEY'),
    host=os.environ.get('LANGFUSE_HOST')
)
langfuse_handler = CallbackHandler()


def ensure_database():
    # подключаемся к системной базе postgres
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    exists = cur.fetchone()

    if not exists:
        print(f"Создаём базу {DB_NAME}...")
        cur.execute(f'CREATE DATABASE "{DB_NAME}"')
    else:
        print(f"База {DB_NAME} уже существует.")

    cur.close()
    conn.close()

def ensure_schema():
    with psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    ) as conn:
        with conn.cursor() as cur:
            # Проверяем, есть ли нужные таблицы
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            tables = {row[0] for row in cur.fetchall()}

            if not {"checkpoints"} <= tables:
                PG_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=disable"
                with PostgresSaver.from_conn_string(PG_URI) as memory:
                    memory.setup()
            else:
                print("Схема LangGraph уже готова.")

def is_running_in_studio():
    """Определяет, запущен ли код в LangGraph Studio через стек вызовов"""
    try:
        # Получаем стек вызовов
        stack = inspect.stack()
        
        # Проверяем, есть ли в стеке вызовов модули LangGraph Studio
        studio_modules = [
            'langgraph_api',
            'langgraph_runtime',
            'langgraph_server',
            'starlette'  # из вашей ошибки видно, что используется starlette
        ]
        
        for frame_info in stack:
            filename = frame_info.filename.lower()
            if any(module in filename for module in studio_modules):
                return True
    except Exception as e:
        print(f"Error detecting studio environment: {e}")
        # В случае ошибки предполагаем standalone режим
        return False
# ============ Tools ============
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
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools=tools)

# System message
sys_message = SystemMessage(content="You are a helpful assistant that can perform arithmetic operations using tools. You can multiply, add, subtract, and divide numbers. Use the tools when necessary.")

# Node
def arithmetic_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}

# ============ Build graph ============

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

# ============ Compile graph ============
ensure_database()
ensure_schema()
PG_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=disable"
with PostgresSaver.from_conn_string(PG_URI) as memory:
    if is_running_in_studio():
        graph = builder.compile() # checkpointer=memory
    else:
        graph = builder.compile(checkpointer=memory)
# Invoke graph with specified thread
    config = {"configurable": {"thread_id": "1"}, "callbacks": [langfuse_handler]}

    messages = [HumanMessage(content="Add 11 and 4.")]
    messages = graph.invoke({"messages": messages}, config) 

    messages = [HumanMessage(content="Multiply that by 2.")]
    messages = graph.invoke({"messages": messages}, config) 


for mes in messages["messages"]:
    mes.pretty_print()