from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Initialize OpenAI Embeddings instance
embeddings = OpenAIEmbeddings()

# Global variable to store context
context = {}

def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        if pages:
            return pages
        else:
            return None
    except Exception as e:
        print(f"Error loading PDF document: {e}")
        return None

def retrieve_response(question, faiss_index, chain):
    similar_response = faiss_index.similarity_search(question, k=3)
    best_practice = [doc.page_content for doc in similar_response]
    
    # Pass context to the chain
    response = chain.run(message=question, best_practice=best_practice, context=context.get('last_response'))
    
    # Update context with current response
    context['last_response'] = response
    
    return response

@app.route('/api/qna', methods=['POST'])
def qna():
    data = request.get_json()
    question = data.get('question')
    pdf_path = data.get('pdf_path')

    if not question or not pdf_path:
        return jsonify({'error': 'Invalid input data. Both question and pdf_path are required.'}), 400

    if not os.path.exists(pdf_path):
        return jsonify({'error': f'PDF file not found at {pdf_path}.'}), 404

    pages = load_pdf(pdf_path)

    if pages:
        faiss_index = FAISS.from_documents(pages, embeddings)

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

        template = """
        You are an Expert trained in ITSM Framework, Your task is to assist users with queries around ITSM Framework with valid data
        Below is a message I received from Users:
        {message}

        Here is a list of best practices of how we normally respond to prospects in similar scenarios:
        {best_practice}

        Please write the best response that I should send to this prospect:
        """

        prompt = PromptTemplate(
            input_variables=["message", "best_practice"],
            template=template
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        try:
            result = retrieve_response(question, faiss_index, chain)
            return jsonify({'answer': result}), 200
        except Exception as e:
            print(f"Error processing Q&A: {e}")
            return jsonify({'error': 'An error occurred while processing the Q&A.'}), 500
    else:
        return jsonify({'error': 'Failed to load PDF or extract text content.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
