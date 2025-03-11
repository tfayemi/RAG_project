from setuptools import setup, find_packages

setup(
    name="rag_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "llama-index",
        "llama-index-embeddings-huggingface",
        "peft",
        "auto-gptq",
        "optimum",
        "bitsandbytes",
        "transformers",
    ],
    author="Your Name",
    description="A RAG pipeline project using a fine-tuned LLM and vector store.",
)
