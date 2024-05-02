from config import *
from customOpenAIEmbeddings import *


llm_chat = ChatOpenAI(
    openai_api_base=OPENAI_URL,
    temperature=0.6,
    openai_api_key="lm-studio"
)

llm_data = ChatOpenAI(
    openai_api_base=OPENAI_URL,
    temperature=0.3,
    openai_api_key="lm-studio"
)

llm_checker = ChatOpenAI(
    openai_api_base=OPENAI_URL,
    temperature=0,
    api_key="lm-studio"
)

embeddings = CustomOpenAIEmbeddings(
    openai_api_base=OPENAI_URL,
    openai_api_key="lm-studio"
)