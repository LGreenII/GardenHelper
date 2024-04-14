from openai import OpenAI
import os
from dotenv import load_dotenv
from secret_key import openai_key
from secret_key import google_palm_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
import geopy
from geopy import Nominatim
import ssl
import certifi

# Loading Environment Variables
load_dotenv()
google_api_key = os.environ["GOOGLE_PALM_KEY"]
openai.api_key = os.environ["openai_key"]

# Create LLM Model
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)

# Create Embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings()

vector_db_file_path ="faiss_index"

def create_vector_db():
    # Loading .csv file to create vecto db
    loader = CSVLoader(file_path="GardeningPromptResponse.csv", csv_args={"delimiter": ",", "quotechar": '"'})
    data = loader.load()
    vector_db = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vector_db.save_local(vector_db_file_path)

def get_qa_chain():
    vector_db = FAISS.load_local(folder_path=vector_db_file_path, embeddings=instructor_embeddings,
                                 allow_dangerous_deserialization=True)

    retriever = vector_db.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document without 
making answers up. If the answer is not found in the context, please state "I do not know." 
Do not try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_llm(
                        llm=llm,
                        retriever=retriever,
                        input_key="query",
                        return_source_documents=True)

    return chain

def usda_hardiness_zones(zip_code):
    prompt = f'''What is the USDA Hardiness Zone for {zip_code} in 2023, 
             what fruit can be grown in this zone,
             and how long will it take seeds of this fruit to germinate?
             
             Always return your answer as a table with 3 columns.
             Column 1 is the USDA Hardiness Zone.
             Column 2 is the plant name. 
             Column 3 is the Germination Time (days) for seeds of the plant from column 2.'''

    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": f'''What is the USDA Hardiness Zone for {zip_code} in 2023, 
             what fruit can be grown in this zone,
             and how long will it take seeds of this fruit to germinate?
             
             Always return your answer as a table with 3 columns.
             Column 1 is the USDA Hardiness Zone.
             Column 2 is the plant name. 
             Column 3 is the Germination Time (days) for seeds of the plant from column 2. The answer should be:'''}
    ])
    print(response.choices[0].message.content)

    return response.choices[0].message.content

def get_map_data_frame(zip_code):

    ctx = ssl.create_default_context(cafile=certifi.where())
    geopy.geocoders.options.default_ssl_context = ctx

    geolocator = Nominatim(scheme='http', user_agent="GardenHelper")
    location = geolocator.geocode(zip_code)
    print(location)



###if __name__ == '__main__':



