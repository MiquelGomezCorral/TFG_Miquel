#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) Solver Machine Learning -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverML <info@solverml.co>, Mar 2025
#


"""
Module: OCR LLM

This package integrates OCR and LLM functionalities for document processing.
It includes submodules for OCR operations (e.g., text extraction from PDFs)
and LLM operations (e.g., querying and structured output generation).
"""
import os

from . import ocr
from . import llm

__version__ = os.getenv("MODULE_VERSION", "0.0.0")

__all__ = [
    "ocr",
    "llm",
]
