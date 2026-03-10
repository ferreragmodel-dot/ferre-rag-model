import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import base64
import io
from PIL import Image
from pathlib import Path
import traceback
import chromadb

# Vertex AI
from google import genai
from google.genai import types
from google.genai.types import Content, Part, GenerationConfig, ToolConfig
from google.genai import errors
from google.genai.chats import Chat

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-2.0-flash-001"
CHROMADB_HOST = os.environ["CHROMADB_HOST"]
CHROMADB_PORT = os.environ["CHROMADB_PORT"]

#############################################################################
#                       Initialize the LLM Client                           #
llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#############################################################################

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in Gianfranco Ferré and fashion archive research. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.

When answering a query:
1. Carefully read all the text chunks provided.
2. Identify the most relevant information from these chunks to address the user's question.
3. Formulate your response using only the information found in the given chunks.
4. If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
5. Always maintain a professional and knowledgeable tone, befitting a Ferré archive expert.
6. If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.

Remember:
- You are an expert in Ferré and fashion, but your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside of the given text chunks.
- If asked about topics unrelated to Ferré or fashion, politely redirect the conversation back to archive-related subjects.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.

Your goal is to provide accurate, helpful information about Ferré and fashion based solely on the content of the text chunks you receive with each query.
"""

# Initialize chat sessions
chat_sessions: Dict[str, Chat] = {}

# Connect to chroma DB
client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
method = "recursive-split"
collection_name = f"{method}-collection"
# Get the collection
#collection = client.get_collection(name=collection_name)

def generate_query_embedding(query):
    kwargs = {
        "output_dimensionality": EMBEDDING_DIMENSION
    }
    response = llm_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(**kwargs)
    )
    return response.embeddings[0].values

def create_chat_session(past_history=None) -> Chat:
    """Create a new chat session with the model"""
    # Create a new chat session
    return llm_client.chats.create(model=GENERATIVE_MODEL, history=past_history)

def generate_chat_response(chat_session: Chat, message: Dict) -> str:
    """
    Generate a response using the chat session to maintain history.
    Handles both text and image inputs.
    
    Args:
        chat_session: The Vertex AI chat session
        message: Dict containing 'content' (text) and optionally 'image' (base64 string)
    
    Returns:
        str: The model's response
    """
    try:
        collection = client.get_collection(name=collection_name) 
        # Initialize parts list for the message
        message_parts = []
        
        
        # Process image if present
        if message.get("image"):
            try:
                # Extract the actual base64 data and mime type
                base64_string = message.get("image")
                if ',' in base64_string:
                    header, base64_data = base64_string.split(',', 1)
                    mime_type = header.split(':')[1].split(';')[0]
                else:
                    base64_data = base64_string
                    mime_type = 'image/jpeg'  # default to JPEG if no header
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(base64_data)
                
                # Create an image Part using FileData
                image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)
                message_parts.append(image_part)

                # Add text content if present
                if message.get("content"):
                    message_parts.append(message["content"])
                else:
                    message_parts.append("Describe what you see in this image in the context of Gianfranco Ferré fashion archive research")
                
            except ValueError as e:
                print(f"Error processing image: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Image processing failed: {str(e)}"
                )
        elif message.get("image_path"):
            # Read the image file
            image_path = os.path.join("chat-history","llm-rag",message.get("image_path"))
            with Path(image_path).open('rb') as f:
                image_bytes = f.read()

            # Determine MIME type based on file extension
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif'
            }.get(Path(image_path).suffix.lower(), 'image/jpeg')

            # Create an image Part using FileData
            image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)
            message_parts.append(image_part)

            # Add text content if present
            if message.get("content"):
                message_parts.append(message["content"])
            else:
                message_parts.append("Describe what you see in this image in the context of Gianfranco Ferré fashion archive research")
        else:
            # Add text content if present
            if message.get("content"):
                # Create embeddings for the message content
                query_embedding = generate_query_embedding(message["content"])
                # Retrieve chunks based on embedding value 
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5
                )
                INPUT_PROMPT = f"""
                {message["content"]}
                {"\n".join(results["documents"][0])}
                """
                message_parts.append(INPUT_PROMPT)
                    
        
        if not message_parts:
            raise ValueError("Message must contain either text content or image")

        # Send message with all parts to the model
        response = chat_session.send_message(message_parts)
        
        return response.text
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )

def rebuild_chat_session(chat_history: List[Dict], history_dir: str = None) -> Chat:
    """Rebuild a chat session with complete context, including images"""

    formatted_history = []
    for message in chat_history:
        if message["role"] == "user":
            parts = []
            has_image = False
            if message.get("image_path") and history_dir:
                image_full_path = os.path.join(history_dir, message["image_path"])
                try:
                    with Path(image_full_path).open("rb") as f:
                        image_bytes = f.read()
                    mime_type = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".gif": "image/gif",
                    }.get(Path(image_full_path).suffix.lower(), "image/jpeg")
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                    has_image = True
                except Exception as e:
                    print(f"Error loading image for history rebuild: {str(e)}")
            if message.get("content"):
                parts.append(types.Part.from_text(text=message["content"]))
            elif has_image:
                parts.append(types.Part.from_text(
                    text="Describe what you see in this image in the context of Gianfranco Ferré fashion archive research"
                ))
            if parts:
                formatted_history.append(types.UserContent(parts=parts))
        elif message["role"] == "assistant":
            if message.get("content"):
                formatted_history.append(
                    types.ModelContent(
                        parts=[types.Part.from_text(text=message["content"])]
                    )
                )

    new_session = create_chat_session(formatted_history)
    return new_session