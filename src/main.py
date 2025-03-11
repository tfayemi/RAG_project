# -*- coding: utf-8 -*-
"""
rag_improved.py

This script demonstrates an improved version of a RAG (Retrieval-Augmented Generation) pipeline.
It leverages a vector store to index documents and uses a fine-tuned LLM for query-based responses.
The pipeline is inspired by Shaw Talebi's work (article: https://towardsdatascience.com/how-to-improve-llms-with-rag-abdc132f76ac).

Authors:
    Original code by Shaw Talebi.
    Improved version by [Your Name].

Requirements:
    - llama-index
    - llama-index-embeddings-huggingface
    - peft
    - auto-gptq
    - optimum
    - bitsandbytes

To install dependencies, run:
    pip install llama-index llama-index-embeddings-huggingface peft auto-gptq optimum bitsandbytes
"""

# Install necessary packages (if not already installed)
!pip install llama-index
!pip install llama-index-embeddings-huggingface
!pip install peft
!pip install auto-gptq
!pip install bitsandbytes

# %% Imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Import for LLM functionality
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# %% Settings and Configuration
# Initialize embedding model from Hugging Face Hub
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Alternative embedding model (uncomment if desired)
# Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large")

# Set other pipeline parameters
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25


# %% Document Loading and Preprocessing
def load_and_filter_documents(directory: str) -> list:
    """
    Load documents from the specified directory and filter out unwanted content.

    Args:
        directory (str): Path to the directory containing articles.

    Returns:
        List of filtered document objects.
    """
    documents = SimpleDirectoryReader(directory).load_data()
    # Filter out documents containing unwanted phrases
    unwanted_phrases = ["Member-only story", "The Data Entrepreneurs", " min read"]
    filtered_docs = [
        doc for doc in documents if not any(phrase in doc.text for phrase in unwanted_phrases)
    ]
    return filtered_docs


# Load and filter documents from the "articles" directory
documents = load_and_filter_documents("articles")
print(f"Number of documents after filtering: {len(documents)}")

# %% Build Vector Store Index
# Create a vector store index from the filtered documents
index = VectorStoreIndex.from_documents(documents)

# %% Setup Retrieval Engine
# Define number of documents to retrieve per query
top_k = 3

# Configure retriever using the vector index
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

# Assemble the query engine with a similarity postprocessor (cutoff=0.5)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)


# %% Retrieve and Format Context from Relevant Documents
def retrieve_context(query: str, top_k: int) -> str:
    """
    Query the vector index and extract relevant context from the top_k documents.

    Args:
        query (str): Query string.
        top_k (int): Number of top documents to retrieve.

    Returns:
        A formatted string containing the context.
    """
    response = query_engine.query(query)
    context = "Context:\n"
    # Loop through the top_k retrieved documents
    for i in range(top_k):
        context += response.source_nodes[i].text + "\n\n"
    return context


query = "What is fat-tailedness?"  # Set the query string
context = retrieve_context(query, top_k)  # Retrieve the context based on the query
print(f"Query: {query}\nRetrieved Context:\n{context}")


# %% Load and Setup the Fine-Tuned LLM
def load_llm(model_name: str, peft_model_path: str):
    """
    Load a fine-tuned language model using PEFT.

    Args:
        model_name (str): The base model name or path.
        peft_model_path (str): The path or identifier for the fine-tuned model.

    Returns:
        Tuple (model, tokenizer) for the loaded LLM.
    """
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=False,
        revision="main"
    )
    # Load PEFT configuration and apply fine-tuning
    config = PeftConfig.from_pretrained(peft_model_path)
    model = PeftModel.from_pretrained(model, peft_model_path)
    # Load tokenizer for the base model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


# Define model and PEFT paths
base_model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
peft_model_identifier = "shawhin/shawgpt-ft"

# Load the model and tokenizer
model, tokenizer = load_llm(base_model_name, peft_model_identifier)


# %% Define Prompt Templates and Query LLM
def generate_response(comment: str, context: str = None, max_tokens: int = 280) -> str:
    """
    Generate a response using the fine-tuned LLM.

    Args:
        comment (str): The input comment or query.
        context (str, optional): Additional context to include in the prompt.
        max_tokens (int, optional): Maximum number of new tokens to generate.

    Returns:
        The generated response as a string.
    """
    # Define the base instruction for ShawGPT
    base_instructions = (
        "ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, "
        "escalating to technical depth upon request. It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. "
        "ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief "
        "expressions of gratitude or feedback, thus keeping the interaction natural and engaging.\n"
    )

    # Create the prompt with or without additional context
    if context:
        prompt = (
            f"[INST]{base_instructions}\n{context}\n"
            f"Please respond to the following comment. Use the context above if it is helpful.\n\n{comment}\n[/INST]"
        )
    else:
        prompt = (
            f"[INST]{base_instructions}\n"
            f"Please respond to the following comment.\n\n{comment}\n[/INST]"
        )

    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate model outputs
    outputs = model.generate(
        input_ids=inputs["input_ids"].to("cuda"),
        max_new_tokens=max_tokens
    )
    # Decode and return the generated text
    return tokenizer.batch_decode(outputs)[0]


# %% Generate LLM Response without Context
comment = query  # Directly use the previously defined query
print("Prompt without context:")
print(comment)

response_no_context = generate_response(comment)
print("Response without context:")
print(response_no_context)

# %% Generate LLM Response with Context
print("Prompt with context:")
response_with_context = generate_response(comment, context)
print("Response with context:")
print(response_with_context)
