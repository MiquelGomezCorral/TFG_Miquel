#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) Solver Machine Learning -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverML <info@solverml.co>, Mar 2025
#

import os
import pytest
from pytest import MonkeyPatch
from ocr_llm_module.llm.azure.azure_openai import AzureOpenAILanguageModel

DUMMY_AZURE_ENDPOINT = "https://dummy.azure.endpoint"
DUMMY_API_KEY = "dummy_key"


@pytest.fixture(autouse=True)  # Function is called before every test function
def clear_api_key(monkeypatch: MonkeyPatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)


def test_raises_without_api_key():
    with pytest.raises(ValueError):
        AzureOpenAILanguageModel(azure_endpoint=DUMMY_AZURE_ENDPOINT)


def test_instantiation_with_provided_api_key():
    model = AzureOpenAILanguageModel(
        azure_endpoint=DUMMY_AZURE_ENDPOINT,
        api_key=DUMMY_API_KEY
    )
    # Ensure the environment variable is set when provided via api_key
    assert os.environ.get("AZURE_OPENAI_API_KEY") == DUMMY_API_KEY
    # Check that the instance has the 'llm' and 'embeddings' attributes initialized.
    assert hasattr(model, "llm")
    assert hasattr(model, "embeddings")


def test_instantiation_with_existing_env_api_key(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", DUMMY_API_KEY)
    model = AzureOpenAILanguageModel(
        azure_endpoint=DUMMY_AZURE_ENDPOINT
    )
    # Confirm that the existing environment variable is used
    assert os.environ.get("AZURE_OPENAI_API_KEY") == DUMMY_API_KEY
    assert hasattr(model, "llm")
    assert hasattr(model, "embeddings")
