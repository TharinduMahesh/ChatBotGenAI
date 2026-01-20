from langchain_core.prompts import ChatPromptTemplate


system_prompt = (
    "You are a helpful medical assistant. Use the context below to answer "
    "the user's question. If you don't know the answer, just say that you don't know. "
    "Keep the answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])