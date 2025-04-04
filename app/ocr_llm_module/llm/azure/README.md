# Azure OpenAI Module

This module provides a comprehensive interface for Azure OpenAI services, enabling both language model interactions and embedding capabilities through a unified client. It's designed for document processing, information extraction, and context-aware querying using Retrieval Augmented Generation (RAG) techniques.

## Overview

The `AzureOpenAILanguageModel` class connects to Azure OpenAI services and provides methods for:

- Text generation using language models
- Structured output generation with schema validation
- Text embedding generation 
- Retrieval-based operations (Page RAG and Paragraph RAG)
- Context-sensitive information extraction

## Key Features

### 1. Language Model Capabilities
- Generate plain text responses from prompts
- Create structured outputs with Pydantic validation
- Configure model parameters like temperature and seed for deterministic results
- See the [following example](../../../examples/ex3_ocr_plus_llm/) for more details


### 2. Embedding Operations
- Generate embeddings for text content
- Perform vector similarity searches 
- Support for context-aware retrieval


### 3. RAG (Retrieval Augmented Generation) Techniques

The module implements two distinct RAG approaches:

#### Page RAG
Page RAG (`get_top_k_matching_pages`) retrieves entire pages that best match a given query:

- Treats each page as a complete document unit
- Useful for finding specific pages in lengthy documents
- Returns `PageRAGResult` objects containing:
	- Complete page content
	- Relevance score
	- Page index
- Supports context expansion around the matched page
- Ideal for identifying relevant pages in large documents
- Here an [example](../../../examples/ex6_page_rag/) for more details.

```python
results = llm_client.get_top_k_matching_pages(
		query="Muscle Spindles",
		pages=document_pages,
		k=2,  # Return top 2 matching pages
		context_size=20  # Include 20 characters before/after the match
)
```

#### Paragraph RAG
Paragraph RAG (`get_top_k_matching_paragraphs`) identifies specific paragraphs that match a query:

- Splits all content into paragraph-sized chunks
- Provides more granular and focused results
- Returns `RAGResult` objects with:
	- Paragraph content
	- Relevance score
- Useful for pinpointing specific information within larger documents
- Here an [example](../../../examples/ex7_paragraph_rag/) for more details.

```python
results = llm_client.get_top_k_matching_paragraphs(
		query="Muscle Spindles",
		pages=document_pages,
		k=5,  # Return top 5 matching paragraphs
		context_size=50  # Include 50 characters before/after the match
)
```

### Key Differences: Page RAG vs. Paragraph RAG

| Feature | Page RAG | Paragraph RAG |
|---------|----------|---------------|
| **Retrieval Unit** | Entire pages | Individual paragraphs |
| **Result Granularity** | Page-level (coarser) | Paragraph-level (finer) |
| **Contextual Scope** | Full page context | Focused paragraph context |
| **Use Case** | Finding relevant pages in long documents | Pinpointing specific information |
| **Result Type** | `PageRAGResult` (includes page index) | `RAGResult` (content and score only) |
| **Content Preservation** | Maintains page layout and structure | Focuses on specific information |
| **Context Method** | Expands around the page boundaries | Expands around the paragraph text |


## Environment Setup

The module requires the following environment variables or constructor arguments:

| Environment Variable | Constructor Parameter | Description |
|---------------------|----------------------|-------------|
| `AZURE_OPENAI_ENDPOINT` | `azure_endpoint` | Azure OpenAI service endpoint |
| `AZURE_OPENAI_API_KEY` | `api_key` | API key for authentication |
| `AZURE_OPENAI_API_VERSION` | `api_version` | API version (defaults to "2024-08-01-preview") |
| `AZURE_OPENAI_EMBEDDINGS_ENDPOINT` | `embedings_endpoint` | Endpoint for embeddings service |
| `AZURE_OPENAI_EMBEDDINGS_API_KEY` | `embeddings_api_key` | API key for embeddings |
| `AZURE_OPENAI_EMBEDDINGS_MODEL` | `embedings_model` | Embedding model name (e.g., "text-embedding-ada-002") |

If one or more of these variables are not set in the environment neither passed to the constructor, the module will raise an exception.