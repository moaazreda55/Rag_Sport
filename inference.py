import gradio as gr
from main import split_text, Embeding, VectorStore, build_chain


with open("./sports.txt", "r", encoding="utf-8") as file:
    file_content = file.read()

chunks = split_text(file_content)
vector_store = VectorStore(chunks)
chain = build_chain(vector_store)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def getting_answer(question):
   
   answer = chain.invoke(question)
   
   similar_responce = retriever.invoke(question)
   similar_text = similar_responce[0].page_content
   
   return  answer , similar_text

demo = gr.Interface(
   fn=getting_answer,
   inputs=["text"],
   outputs=["text","text"],
)
demo.launch()