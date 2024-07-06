
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
import torch
import bitsandbytes as bnb
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from typing import Any
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from langchain import LLMChain
from langchain.chains import ConversationChain
from transformers import StoppingCriteriaList,StoppingCriteria
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import re
from langchain_huggingface.llms import HuggingFacePipeline
import torch
from typing import List
import wikipediaapi
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from langchain.docstore.document import Document
import sys
import requests
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

# Load the tokenizer and model
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='cuda:0')
print("Model quantized successfully.")

# Load the embeddings model
embeddings=HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct', model_kwargs={'device': 'cuda:0'} )

# Function to clean text
def clean_text(text):
    # Lowercasing
    text = text.lower()
    # Removing punctuation and non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Removing extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def retrieval(query):
  wiki_wiki = wikipediaapi.Wikipedia('en')
  docs = {}
  try:
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch={query}"
    search_response = requests.get(search_url)
    search_data = search_response.json()
    wiki_title = search_data['query']['search'][0]['title']
    wiki_text = wiki_wiki.page(wiki_title).text
    docs[wiki_title] = wiki_text
    # documents = [Document(page_content=docs[title]) for title in docs]
    documents = [Document(page_content=docs[title]) for title in docs]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    return texts

  except Exception as e:
      return "Sorry, No Data found"


#Setting the Template and Prompt
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know; don't try to make up an answer."

def generate_prompt(context: str, question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{SYSTEM_PROMPT}

{context}

Question : {question} [/INST] <>""".strip()

template = generate_prompt(
    context="{context}",
    question="{question}",
    system_prompt=SYSTEM_PROMPT
)

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

#Defining LLM
from transformers import pipeline as transformers_pipeline

class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False

stop_tokens = [["Question", ":"]]
stopping_criteria = StoppingCriteriaList(
    [StopGenerationCriteria(stop_tokens, tokenizer, model.device)]
)

# Use transformers_pipeline to initialize the pipeline
pipeline = transformers_pipeline(
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    return_full_text=True,
    task="text-generation",
    trust_remote_code=True,
    device_map="cuda:0",
    max_length=10000,
    do_sample=True,
    top_k=4,
    num_return_sequences=1,
    temperature=0.1,
    truncation=True,
    repetition_penalty=1.7,
    stopping_criteria=stopping_criteria,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={"device": "cuda:0"})

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# class InputData(BaseModel):
#     input: str

class TextInput(BaseModel):
    inputs: str
    parameters: dict[str, Any] | None

@app.get("/")
def status_gpu_check() -> dict:
    gpu_msg = "Available" if torch.cuda.is_available() else "Unavailable"
    return {
        "status": "I am ALIVE!",
        "gpu": gpu_msg
    }

@app.post("/generate/")
async def generate_text(data: TextInput) -> dict[str, str]:
    try:
        # query = data.inputs
        # input_text = prompt.format(input=query)
        # response = llm(input_text)
        # return {"generated_text": response.split("AI:")[1].strip()}

        user_input = data.inputs
        new_words = word_tokenize(user_input)

        new_filtered_words = [word for word in new_words if word.lower() not in stopwords.words('english')]

        user_input1 = ' '.join(new_filtered_words)

        text_chunks = retrieval(user_input1)
        print(text_chunks)

        if text_chunks=="Sorry, No Data found":
          print("No data found")
          text_chunks = retrieval("Todays Date")
          vector_store=FAISS.from_documents(text_chunks, embeddings)
        else:
          vector_store=FAISS.from_documents(text_chunks, embeddings)

        chain = RetrievalQA.from_chain_type(
                                              llm=llm,
                                              chain_type="stuff",
                                              retriever=vector_store.as_retriever(search_kwargs={'k':2}),
                                              return_source_documents=True,
                                              chain_type_kwargs={"prompt": prompt}
                                            )

        result = chain({"query": user_input})
        k = result['result'].split('[/INST] <>')[1].strip()
        return {"generated_text": k}


    except Exception as e:
        print(type(data))
        print(data)
        raise HTTPException(status_code=500, detail=len(str(e)))
