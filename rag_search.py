from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA

def run_retrieval_qa(supabase):
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    vectorstore = SupabaseVectorStore.from_texts(
        texts=[d["content"] for d in supabase.table("travel_vectors").select("content").execute().data],
        embedding=embeddings,
        client=supabase,
        table_name="travel_vectors"
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    print("\nWelcome to the Travel Assistant Q&A! Type 'exit' to return to main menu.")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ['exit', 'quit']:
            print("Returning to main menu...\n")
            break
        answer = qa.invoke(query)
        if isinstance(answer, dict) and "result" in answer:
            print("\nAnswer:", answer["result"])
        else:
            print("\nAnswer:", answer)
