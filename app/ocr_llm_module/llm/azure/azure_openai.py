#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) Solver Machine Learning -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverML <info@solverml.co>, Mar 2025
#

import os
from typing import Any, Dict, Optional, Sequence, Type, TypeVar, Union
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from pydantic import BaseModel, SecretStr

_BM = TypeVar("_BM", bound=BaseModel)
DictOrPydanticClass = Union[Dict[str, Any], Type[_BM]]


class RAGResult(BaseModel):
    content: str
    score: float

    def __str__(self) -> str:
        sanitized_content = self.content.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("  ", " ").strip()
        sanitized_content = sanitized_content[:min(len(sanitized_content), 50)]

        return f"With score {self.score}: {sanitized_content}..."

    def __repr__(self) -> str:
        return self.__str__()


class PageRAGResult(RAGResult):
    page_index: int

    def __str__(self) -> str:

        sanitized_content = self.content.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("  ", " ").strip()
        sanitized_content = sanitized_content[:min(len(sanitized_content), 50)]

        return f"Page {self.page_index} with score {self.score}: {sanitized_content}..."

    def __repr__(self) -> str:
        return self.__str__()


class AzureOpenAILanguageModel():
    """
    Represents an Azure-based OpenAI language model and embedding service.
    This class encapsulates the interaction with Azure's OpenAI offerings, providing access to
    both chat-based language models and text embedding models. It ensures that the required API
    key is available for authentication either via an environment variable or as a constructor
    argument, and sets up the necessary client instances for subsequent API interactions.
    Parameters:
        azure_endpoint (str): The Azure endpoint URL for the OpenAI service.

        temperature (float, optional): Controls the randomness of the language model's responses.
        Defaults to 0.0.

        verbose (bool, optional): Enables verbose logging of API interactions if set to True.
        Defaults to True.

        seed (int, optional): A seed value for random number generation to enable reproducible
        outcomes. Defaults to 360.

        api_key (str | None, optional): The API key used for authenticating with the
        Azure OpenAI service. If not provided, the environment variable
        'AZURE_OPENAI_API_KEY' is used.

        api_version (str, optional): The version of the Azure OpenAI API to be used. Defaults
        to "2024-08-01-preview".

        embedings_model (str, optional): The identifier for the embedding model to be used.
        Defaults to "text-embedding-ada-002".

    Raises:
        ValueError: If the API key is not provided either through the environment variable
        'AZURE_OPENAI_API_KEY' or as the api_key parameter.
    Example:
        >>> model = AzureOpenAILanguageModel(
        ...     azure_endpoint="https://your-azure-endpoint.openai.azure.com/",
        ...     api_key="your_api_key_here"
        ... )
        >>> response = model.llm.generate("Hello, how are you?")
        >>> embedding = model.embeddings.embed("Hello, how are you?")
    """

    def __init__(
        self,
        *,
        temperature: float = 0.0,
        verbose: bool = True,
        seed: int = 360,
        azure_endpoint: str | None = os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        api_key: str | None = os.environ.get("AZURE_OPENAI_API_KEY", None),
        api_version: str | None = os.environ.get("AZURE_OPENAI_API_VERSION", None),
        embedings_endpoint: str | None = os.environ.get("AZURE_OPENAI_EMBEDDINGS_ENDPOINT", None),
        embeddings_api_key: str | None = os.environ.get("AZURE_OPENAI_EMBEDDINGS_API_KEY", None),
        embedings_model: str | None = os.environ.get("AZURE_OPENAI_EMBEDDINGS_MODEL", None),
    ):

        if api_key is None:
            raise ValueError(
                "Azure API Key is required to use Azure OpenAI. "
                "Please set the 'AZURE_OPENAI_API_KEY' environment"
                " variable or pass it as an argument in the constructor."
            )

        if azure_endpoint is None:
            raise ValueError(
                "Azure endpoint is required to use Azure OpenAI. "
                "Please set the 'AZURE_OPENAI_ENDPOINT' environment"
                " variable or pass it as an argument in the constructor."
            )

        if api_version is None:
            raise ValueError(
                "Azure API version is required to use Azure OpenAI. "
                "Please set the 'AZURE_OPENAI_API_VERSION' environment"
                " variable or pass it as an argument in the constructor."
            )

        if embedings_model is None:
            raise ValueError(
                "Azure Embeddings model is required to use Azure OpenAI. "
                "Please set the 'AZURE_OPENAI_EMBEDDINGS_MODEL' environment"
                " variable or pass it as an argument in the constructor."
            )

        if embedings_endpoint is None:
            raise ValueError(
                "Azure Embeddings endpoint is required to use Azure OpenAI. "
                "Please set the 'AZURE_OPENAI_EMBEDDINGS_ENDPOINT' environment"
                " variable or pass it as an argument in the constructor."
            )

        if embeddings_api_key is None:
            raise ValueError(
                "Azure Embeddings API Key is required to use Azure OpenAI. "
                "Please set the 'AZURE_OPENAI_EMBEDDINGS_API_KEY' environment"
                " variable or pass it as an argument in the constructor."
            )

        self.llm: AzureChatOpenAI = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=SecretStr(api_key) if api_key else None,
            verbose=verbose,
            temperature=temperature,
            seed=seed,
        )

        self.embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
            azure_endpoint=embedings_endpoint,
            api_version=api_version,
            api_key=SecretStr(embeddings_api_key) if api_key else None,
            model=embedings_model,
        )

    def get_llm_plain_response(self, prompt: str | Sequence[str]) -> str:
        """
        Generate a plain text response from the language model.
        Parameters:
            prompt (str | Sequence[str]): The prompt text to be used for generating the response.
        Returns:
            str: The response text generated by the language model.
        """
        return self.llm.invoke(prompt).text()

    def get_llm_response_with_structured_response(  # type: ignore
        self,
        prompt: str | Sequence[str],
        schema: Optional[DictOrPydanticClass] = None,  # type: ignore
    ) -> DictOrPydanticClass:  # type: ignore
        """
        Generate a structured response from the language model.
        Parameters:
            prompt (str | Sequence[str]): The prompt text to be used for generating the response.
        Returns:
            dict: The structured response generated by the language model.
        """
        return self.llm.with_structured_output(schema).invoke(prompt)  # type: ignore

    def get_embedding(self, text: str) -> Sequence[float]:
        """
        Generate an embedding for the given text.
        Parameters:
            text (str): The text for which an embedding is to be generated.
        Returns:
            Sequence[float]: The embedding vector for the given text.
        """

        output: list[float] = self.embeddings.embed_query(text)

        return output

    def get_top_k_matching_pages(
        self,
        *,
        query: str,
        pages: Sequence[str],
        k: int = 5,
        context_size: int = 0,
    ) -> Sequence[PageRAGResult]:
        """
        Get the top k pages that match the query
        Parameters:
            query (str): The query text to be used for finding matching pages.
            pages (Sequence[str]): The list of pages to be searched for matching the query.
            k (int): The number of top matching pages to return. Defaults to 5.
            context_size (int): The number of lines to consider before and after the
            matching page. Defaults to 0.
        Returns:
            Sequence[int]: The indices of the top k pages that match the query.
        """

        if not pages:
            return []

        if len(pages) == 1:
            return [PageRAGResult(content=pages[0], score=1.0, page_index=0)]

        if len(pages) <= k:
            return [
                PageRAGResult(content=page, score=1.0, page_index=index)
                for index, page in enumerate(pages)
            ]

        vector_store = InMemoryVectorStore(self.embeddings)
        vector_store.add_documents(
            [
                Document(
                    id=str(index),
                    page_content=page
                )
                for index, page in enumerate(pages)
            ]
        )

        results = vector_store.similarity_search_with_score(  # type: ignore
            query=query,  # type: ignore
            k=k,  # type: ignore
        )

        page_scores: list[tuple[str, float, int]] = []

        for doc, score in results:
            if doc.id is None:
                continue

            index = int(doc.id)
            page = pages[index]
            page_scores.append((page, score, index))

        output = [
            PageRAGResult(
                content=page, score=score, page_index=index
            )
            for page, score, index in page_scores
        ]

        if context_size == 0:
            return output

        merged_pages: str = "\n".join(pages)

        for result in output:
            index = merged_pages.find(result.content)
            if index == -1:
                continue

            start_index = max(0, index - context_size)
            end_index = min(len(merged_pages), index + len(result.content) + context_size)

            result.content = merged_pages[start_index:end_index]

        return output

    def get_top_k_matching_paragraphs(
        self,
        query: str,
        pages: Sequence[str],
        context_size: int = 0,
        k: int = 5,
    ) -> Sequence[RAGResult]:
        """
        Get the top k paragraphs that match the query
        Parameters:
            query (str): The query text to be used for finding matching paragraphs.
            pages (Sequence[str]): The list of pages to be searched for matching the query.
            context_size (int): The number of lines to consider before and after the
            matching paragraph. Defaults to 5.
            k (int): The number of top matching paragraphs to return. Defaults to 5.
        Returns:
            Sequence[str]: The top k paragraphs that match the query.
        """

        if context_size < 0:
            raise ValueError("Context size cannot be negative.")

        if len(pages) == 0:
            raise ValueError("No pages found to search for matching paragraphs.")

        if len(query) == 0:
            raise ValueError("Query cannot be empty.")

        if k < 1:
            raise ValueError("k must be greater than 0.")

        vector_store = InMemoryVectorStore(self.embeddings)

        paragraphs = "\n".join(pages).split("\n")
        vector_store.add_texts(paragraphs)  # type: ignore

        search_results = vector_store.similarity_search_with_score(  # type: ignore
            query=query,  # type: ignore
            k=k,  # type: ignore
        )

        results: list[RAGResult] = [
            RAGResult(
                score=score,  # type: ignore
                content=doc.page_content,
            )
            for doc, score in search_results
        ]

        if context_size == 0:
            return results

        merged_pages: str = "\n".join(pages)

        for result in results:
            index = merged_pages.find(result.content)
            if index == -1:
                continue

            start_index = max(0, index - context_size)
            end_index = min(len(merged_pages), index + len(result.content) + context_size)

            result.content = merged_pages[start_index:end_index]

        return results


