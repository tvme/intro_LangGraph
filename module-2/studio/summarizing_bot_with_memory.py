from dotenv import load_dotenv
import os
from typing_extensions import Literal
import inspect
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, RemoveMessage
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

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
USER_THREAD = "a1"

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
    return False

class State(MessagesState):
    summary: str

# Define the logic to call the model
def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_msg = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_msg)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
    # Get existing summary
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but 2 most recent messages
    if len(messages) > 2:
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {"summary": response.content, "messages": delete_messages}

def should_continue_conversation(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""

    messages = state.get("messages", [])
    if len(messages) > 6:
        # Check if the conversation should continue based on the state
        return "summarize_conversation"
    return END

workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue_conversation)
workflow.add_edge("summarize_conversation", END)

ensure_database()
ensure_schema()
PG_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=disable"
with PostgresSaver.from_conn_string(PG_URI) as memory:
    if is_running_in_studio():
        graph = workflow.compile() # checkpointer in Studio

    else:
        graph = workflow.compile(checkpointer=memory)
        graph_png = graph.get_graph(xray=True).draw_mermaid_png()
        with open("summarizing_bot_with_memory_schema.png", "wb") as f:
            f.write(graph_png)
        config = {"configurable": {"thread_id": USER_THREAD}, 
                  "callbacks": [langfuse_handler],
                  "metadata": {"langfuse_session_id": USER_THREAD,}}
        while True:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                break

            output = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            print("Bot:", output["messages"][-1].content) 