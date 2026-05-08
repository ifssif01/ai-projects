# Import the required packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

# Load the HTML document
loader = UnstructuredHTMLLoader(file_path="data/mg-zs-warning-messages.html")
car_docs = loader.load()

# Load the models
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])

# 1. Split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(car_docs)

# 2. Store embeddings in vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 3. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Prompt template
template = """You are a helpful car assistant providing guidance to drivers based on the car manual.
Use only the context below to answer the question. Be clear and concise — your response may be read aloud.

Context: {context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

# 5. Define RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 6. Invoke RAG chain
query = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"
response = rag_chain.invoke(query)
answer = response.content

print(answer)