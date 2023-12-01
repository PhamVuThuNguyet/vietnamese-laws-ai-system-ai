from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from schema import TrainingRequest, ChatRequest, SearchRequest, BatchTrainingRequest
import data_training
from chat import ChatAgent

app = FastAPI()

# Set up CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()


@app.get("/health")
async def pong():
    return {"ping": "pong!"}


@app.post("/train-data")
async def train_data(request: TrainingRequest):
    charter = request.charter
    data_training.ingest_plain_text(charter)

@app.post("/train-data-batch")
async def train_data(request: BatchTrainingRequest):
    list_of_charter = request.charter_list
    for charter in list_of_charter:
        new_request = TrainingRequest(charter=charter)
        await train_data(new_request)

@app.post("/chat")
async def chat(request: ChatRequest):
    messages = request.messages
    question = messages[-1]
    chat_agent = ChatAgent()
    relevant_docs =  chat_agent.query_relevant_docs(question)
    response = chat_agent.chat(question, relevant_docs)
    return response

@app.post("/search")
async def search(request: SearchRequest):
    pass