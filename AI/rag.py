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


def generate_questions(pdf,job_desc):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=800)
    #load pdf
        # pdf = r"D:\FullStack Projects\GenAI Projects\AI-Placement-Preperation\docs\MVResume.pdf"
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

    #use retrievers to get relevent chunks from vector store/database based on query
    retrievers = vector_store.as_retriever(type='mmr',kwargs={'k' : 2 ,'fetch_k' : 4, 'lambda_mult' : 0.5})

    def formated_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    # resume_query = "skills experience projects technologies achievements responsibilities tools"
    # context_docs = retrievers.invoke(resume_query)

    # resume_context = "\n\n".join([doc.page_content for doc in context_docs])


    # job_description = """
    # **Job Title:** Generative AI Engineer / LLM Engineer Intern

    # **Job Description:**
    # We are looking for a Generative AI Engineer Intern to design and build AI-powered applications using large language models (LLMs). You will work on real-world AI features such as Retrieval-Augmented Generation (RAG), prompt engineering, embeddings, and AI agents.

    # **Responsibilities:**

    # * Build applications using LLMs (OpenAI, Groq, Claude, etc.)
    # * Implement RAG pipelines using vector databases (FAISS, Pinecone, Chroma)
    # * Design prompts and evaluation pipelines
    # * Develop REST APIs for AI features
    # * Integrate AI models with frontend applications
    # * Work with embeddings and semantic search
    # * Optimize latency and token usage
    # * Deploy AI services using AWS / Docker

    # **Required Skills:**

    # * Python
    # * LangChain / LlamaIndex
    # * REST API development
    # * Vector databases (FAISS / Chroma / Pinecone)
    # * Prompt Engineering
    # * JSON structured outputs
    # * Git

    # **Preferred Skills:**

    # * FastAPI
    # * React / MERN stack
    # * RAG architecture
    # * Agents & tool calling
    # * AWS (EC2, S3)
    # * Docker
    # * Streaming responses

    # **Nice to Have:**

    # * Built AI chatbot or RAG app
    # * Experience with embeddings
    # * Resume parser / document QA
    # * LLM evaluation techniques

    # **Example Projects:**

    # * AI chatbot with RAG
    # * Resume analyzer using LLM
    # * AI coding assistant
    # * Document question-answering system

    # **Experience:** 0 to 1 years (Intern / Fresher)
    # **Location:** Remote / Hybrid
    # """


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