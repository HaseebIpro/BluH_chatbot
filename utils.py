import pinecone
import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


PINECONE_API_KEY = 'pcsk_BmD5J_915paWYqUQiNC2UtdqJ8QnREd8HmTTaWvg6K259NhUseSqJg8ggoaYfLtoHjAht'
PINECONE_ENV = 'us-east-1'
PINECONE_HOST = 'https://domain-knowledge-xq431jt.svc.aped-4627-b74a.pinecone.io'

# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

# Create a new index in Pinecone (run this once)
index_name = "domain-knowledge"
index = pinecone.Index(api_key=PINECONE_API_KEY, host=PINECONE_HOST)

# Load the SentenceTransformer model
model = SentenceTransformer('thenlper/gte-large')



def search_pinecone(query, top_k=3):
    # Generate embedding for the user query
    query_embedding = model.encode(query).tolist()
    
    # Use keyword arguments for the query method
    query_results = index.query(
        vector=query_embedding,  # Use `vector` instead of a positional argument
        top_k=top_k,  # Specify the number of top results
        include_metadata=True  # Include metadata for the results
    )
    
    # Extract the closest completions and their metadata
    completions = [result['metadata']['completion'] for result in query_results['matches']]
    return completions

from groq import Groq

# Initialize the Groq client
client = Groq(api_key="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")

def generate_answer(query, retrieved_completions, chat_history):
    # Combine the user query and retrieved completions into a prompt
    system_message = {
        "role": "system",
        "content": (
        "You are a knowledgeable and helpful assistant for shrimp hatcheries. "
        "Your role is to provide accurate and concise information related to shrimp farming, "
        "hatchery operations, disease management, water quality monitoring, feed optimization, and other relevant topics. "
        "Always act as you are answering from your own knowledge and the provided context is also you own knowledge."
        "If there is no information about a query just apologize and tell that you don't know"
        )
    }
    
    context_messages = [
        {"role": msg["role"], "content": msg["content"]} for msg in chat_history[-5:]
    ]

    user_message = {
        "role": "user",
        "content": f"User query: {query}\n\nRelevant information:\n" + "\n".join(retrieved_completions)
    }
    
    all_messages = [system_message] + context_messages + [user_message]

    # Generate response with Groq API
    chat_completion = client.chat.completions.create(
        messages=all_messages,
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,  # Stream partial results
    )
    
    # Aggregate and return the full response
    response_text = chat_completion.choices[0].message.content
    
    return response_text


def handle_user_query(query, chat_history):
    # Step 1: Retrieve closest completions from Pinecone
    closest_completions = search_pinecone(query)
    
    # Step 2: Pass query and retrieved completions to the LLM for final answer
    final_answer = generate_answer(query, closest_completions, chat_history)
    
    return final_answer