from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate ,ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from ragas.integrations.langchain import EvaluatorChain
# from ragas.metrics import faithfulness
# from ragas.metrics import context_precision
import os
import glob
from dotenv import load_dotenv

load_dotenv()

# txt_files = glob.glob("./data/Sports/*.txt")

# with open("sports.txt","w",encoding="utf-8") as allfile:
#     for file in txt_files :
#         with open(file,'r',encoding="utf-8") as eachfile:

#             allfile.write(eachfile.read())



file = open("./sports.txt",'r')
file_content = file.read()

def split_text(text):
    text_spliter = CharacterTextSplitter(
        separator= "   ",
        chunk_size=80,
        chunk_overlap=15)

    chunks = text_spliter.create_documents([text])
    return chunks

class Embeding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self,docs):
        embedings = self.model.encode(docs)
        return embedings.tolist()
    
    def embed_query(self,query):
        return self.model.encode(query).tolist()


def VectorStore(chunks):

    embedding_model = Embeding()

    vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model)
    return vector_store


def build_chain(vector_store):

    gemini_key = os.getenv("Gemini_key")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  #
        google_api_key=gemini_key,
        temperature=0)

    retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3})

    prompt_text = os.getenv("prompt")

    prompt_template = PromptTemplate.from_template(prompt_text)

    chain = (
        {"context": retriever, "question":RunnablePassthrough()}
        |prompt_template
        |llm
        |StrOutputParser())
    
    return chain

# def evaluation()
#     faithfulness_chain = EvaluatorChain(
#     metric=faithfulness,
#     llm=llm,
#     embeddings=Embeding())
#     eval_result= faithfulness_chain()


if __name__ == "__main__":

    chunks = split_text(file_content)
    vector_store = VectorStore(chunks)
    chain = build_chain(vector_store)

    answer = chain.invoke("ماذا تعرف عن المدافعون الذين يحاول برشلونه ضمهم ؟")
    print(answer)



# prompt = """
# You are an AI bot , your role is to answer the user 
# questions from the knowledge you get from the document,
# If you get the answer :
# - Give the user his answer 
# - Thank him for his question 
# - Ask him if he needs anything else

# If you don't get the answer : 
# - Appolgy to him 
# - Tell him that you can't get the answer from the document
# - Ask the user if he has any question else
# - Don't tell him anything doesn't belong the answer 

# Knowledge you know:
# {context}

# Question:
# {question}

# answer:
# """