from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
# from langchain_community.tools.tavily_search import TavilySearchResults

# загружаем переменные окружения
load_dotenv()
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
TAVILY_API_KEY=os.environ.get('TAVILY_API_KEY')

gpt4o_chat = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0
)

# Create a message
msg = HumanMessage(content="Hello world", name="Alex")

# Message list
messages = [msg]

# Invoke the model with a list of messages 
print(gpt4o_chat.invoke(messages).content)

user_request = "What is LangGraph?"
tavily_search = TavilySearch(max_results=3)
search_doc = tavily_search.invoke(user_request)
print(f"Search results for '{user_request}':")
for i, doc in enumerate(search_doc['results']):
    print(f"Result {i + 1}: {doc}")