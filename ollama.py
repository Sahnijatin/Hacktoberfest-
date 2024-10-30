import os
import logging
from flask import Flask, request, jsonify
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import Ollama
import openai

os.system("c:\Users\JatinSahni\OneDrive - inmorphis.com\Desktop\OpenAI\Pinecone\.venv\Scripts\python.exe -m pip install flask pypdf2 pinecone")

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = ""

# Specify the path to the directory containing PDF files
docs_path = os.getenv("DOCS_PATH", "C:/Users/JatinSahni/OneDrive - inmorphis.com/Desktop/OpenAI/Pinecone/data")

def extract_text_from_pdfs(docs_path):
    text_chunks = []
    for f_name in os.listdir(docs_path):
        doc_path = os.path.join(docs_path, f_name)
        if doc_path.endswith('.pdf'):  # Check if the file is a PDF file
            try:
                with open(doc_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        text_chunks.append({"text": page.extract_text(), "filename": f_name})
            except Exception as e:
                logger.error(f"Error reading {f_name}: {e}")
    return text_chunks

def preprocess_text_chunks(text_chunks):
    return [{"text": string["text"].strip().strip('\n'), "filename": string["filename"]} for string in text_chunks if len(string["text"].split()) >= 10]

def generate_embeddings(text_chunks):
    from openai.embeddings_utils import get_embedding  # Importing inside the function to reduce unnecessary dependencies if not called
    return [get_embedding(chunk["text"], engine='text-embedding-ada-002') for chunk in text_chunks]

def format_embeddings(text_chunks, text_embeddings):
    return [{"id": str(i), "values": embedding, "metadata": {"filename": chunk["filename"], "text": chunk["text"]}} for i, (chunk, embedding) in enumerate(zip(text_chunks, text_embeddings))]

def index_embeddings(formatted_embeddings):
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "servicenow"
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='gcp', region='us-central1')
        )

    index = pc.Index(index_name, host="https://servicenow-1e5aki9.svc.gcp-starter.pinecone.io")
    index.upsert(formatted_embeddings)
    return index

def search_docs(query, index):
    from openai.embeddings_utils import get_embedding  # Importing inside the function to reduce unnecessary dependencies if not called
    xq = get_embedding(query, engine="text-embedding-ada-002")
    res = index.query(vector=[xq], top_k=5, include_metadata=True)
    return [match['metadata'] for match in res['matches']]

def construct_prompt(query, conversation_history, matches):
    chosen_text = "\n".join(match['text'] for match in matches)
    context_with_history = f"{chosen_text}\n\nPrevious Conversation:\n{conversation_history}"
    
    prompt = (
        "Answer the question as truthfully as possible using the context below, and if the answer is not within the context, say 'I don't know.'\n\n"
        f"Context: {context_with_history}\n\n"
        f"Question: {query}\n"
        "Answer: "
    )
    return prompt

def answer_question(query, conversation_history):
    matches = search_docs(query, index)
    prompt = construct_prompt(query, conversation_history, matches)
    
    try:
        cached_llm = Ollama(model="phi3")
        response = cached_llm.invoke(prompt)
        answer = response.strip()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        answer = None
    
    return answer

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_question = data['question']
    conversation_history = data.get('conversation_history', [])
    
    answer = answer_question(user_question, conversation_history)
    
    conversation_history.append({"user_question": user_question, "bot_answer": answer})
    if len(conversation_history) > 3:
        conversation_history = conversation_history[-3:]
    
    response = {
        "answer": answer,
        "conversation_history": conversation_history
    }
    return jsonify(response)

if __name__ == '__main__':
    text_chunks = extract_text_from_pdfs(docs_path)
    text_chunks = preprocess_text_chunks(text_chunks)
    text_embeddings = generate_embeddings(text_chunks)
    formatted_embeddings = format_embeddings(text_chunks, text_embeddings)
    index = index_embeddings(formatted_embeddings)
    
    app.run(debug=True)
