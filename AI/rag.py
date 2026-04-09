from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()
import json



llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0, 
    max_tokens=800
    )


vector_store = None
def load_pdf(pdf):
    global vector_store
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    #split teh docs into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    #create vector and store into vector store/database
    embeddings = HuggingFaceEndpointEmbeddings(model='sentence-transformers/paraphrase-MiniLM-L3-v2')
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    
def generate_questions(job_desc):
    global vector_store
#use retrievers to get relevent chunks from vector store/database based on query
    retrievers = vector_store.as_retriever(type='mmr',kwargs={'k' : 2 ,'fetch_k' : 4, 'lambda_mult' : 0.5})

    def formated_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    prompt1 = PromptTemplate(
        template="""
    Analyze resume and job description.

    You must return ONLY valid JSON.
    Do not repeat sections.
    Do not add explanation.
    Do not add text before or after JSON.

    Return format:
        {{
            "match_percentage": "",
            "matching_skills": [],
            "missing_skills": [],
            "technical_questions": [],
            "hr_questions": []
        }}

        Resume:
            {document}
            Job:
                {job_desc}
                """,
        input_variables=['document', 'job_desc']
    )

    parser = StrOutputParser()
    resume_query = "skills experience projects technologies achievements responsibilities tools"
    parallelChain = RunnableParallel({
        'document' : RunnableLambda(lambda x : resume_query ) | retrievers | RunnableLambda(formated_docs),
        'job_desc' : RunnablePassthrough()
        }
    )

    chain = parallelChain | prompt1 | llm | parser 

    res = chain.invoke(job_desc)
    return json.loads(res)

    
# print("\nMatch Percentage:\n", result["match_percentage"])

# print("\nMatching Skills:")
# for s in result["matching_skills"]:
#     print("-", s)

# print("\nMissing Skills:")
# for s in result["missing_skills"]:
#     print("-", s)

# print("\nTechnical Questions:")
# for q in result["technical_questions"]:
#     print("-", q)

# print("\nHR Questions:")
# for q in result["hr_questions"]:
#     print("-", q)