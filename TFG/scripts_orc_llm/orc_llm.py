#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) Solver Machine Learning -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverML <info@solverml.co>, Mar 2025
#

"""
In this example, we will use the LLM to extract information from an invoice but
in a structured format. We will use the OCR to read the invoice and then send the
information to the LLM to extract the invoice number and tax information. The
response from the LLM will be converted to a pydantic schema and then saved to a
file in JSON format.
"""
if __name__ == "__main__":
    import sys
    import dotenv
    sys.path.append("/app")
    dotenv.load_dotenv()
    
import argparse
import os
import io
import img2pdf  # type: ignore
from pydantic import BaseModel, Field
from ocr_llm_module.llm.azure.azure_openai import AzureOpenAILanguageModel
from ocr_llm_module.ocr.azure.document_intelligence import AzureDocumentIntelligenceClient


# Define the structure of the response from the LLM
class LLMStructuredResponse(BaseModel):
    """
    Structure for the response from the LLM
    """
    buyer: str = Field(
        title="Buyer",
        description="The name of the buyer on the invoice"
    )
    address: str = Field(
        title="Address",
        description="The address of the buyer"
    )
    date: str = Field(
        title="Date",
        description="The date of the invoice in YYYY-MM-DD format"
    )
    shopping_or_tax: bool | None = Field(
        title="Shopping or Tax",
        description="True for commercial purchases, False for tax-related transactions",
    )
    currency: str = Field(
        title="Currency",
        description="The currency used in the invoice"
    )
    subtotal: float = Field(
        title="Subtotal",
        description="The subtotal amount before discounts and tax"
    )
    discount: float = Field(
        title="Discount",
        description="The discount applied to the invoice"
    )
    tax: float = Field(
        title="Tax",
        description="The total tax applied to the invoice"
    )
    total: float = Field(
        title="Total",
        description="The final total amount after tax and discounts"
    )


def main(args):
    # ================ Initialize clients and aux================
    llm_client: AzureOpenAILanguageModel = AzureOpenAILanguageModel()
    ocr_client: AzureDocumentIntelligenceClient = AzureDocumentIntelligenceClient()

    file_io: io.BytesIO = io.BytesIO()

    os.makedirs(args.save_path, exist_ok=True)
    # ================ Process files ================
    for document in os.listdir(args.dataset_path):
        document_path = os.path.join(args.dataset_path, document)

        # Read invoice from file to bytesIO
        file_io = prepare_document(file_io, document_path)

        # Send document to ORC to extract content
        document_content, pages = document_to_orc(ocr_client, file_io)

        # Define the prompt and send it to send to the LLM
        llm_output = document_to_llm(llm_client, document_content)

        # OUTPUT MANAGEMENT 
        save_output(llm_output, args.save_path, document)


def prepare_document(file_io: io.BytesIO, document_path: str) -> io.BytesIO:
    # If the file is not a pdf we assume it is an image
    if not document_path.endswith(".pdf"):
        a4inpt = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))  # type: ignore
        with open(document_path, "rb") as document:
            file_io.write(
                img2pdf.convert(  # type: ignore
                    [document.read()],
                    layout_fun=img2pdf.get_layout_fun(  # type: ignore
                        pagesize=a4inpt,
                    ),
                )  # type: ignore
            )
    else:
        with open(document_path, "rb") as f:
            file_io.write(f.read())

    file_io.seek(0)
    return file_io


def document_to_orc(ocr_client: AzureDocumentIntelligenceClient, file_io: io.BytesIO):
    pages: list[str] = ocr_client.read_invoice(file_io)

    document_content: str = "\n\n ------- PAGE BREAK ------- \n\n".join(pages)

    return document_content, pages


def document_to_llm(llm_client: AzureOpenAILanguageModel, document_content: str):

    prompt: str = f"""
    Given the following invoice:
    <document>
    {document_content}
    </document>

    Extract the following details:
    - Buyer name
    - Buyer address
    - Invoice date (format: YYYY-MM-DD)
    - Transaction type (Shopping or Tax-related; return True for Shopping, False for Tax)
    - Currency used
    - Subtotal amount
    - Discount applied (Total ammount, not percentage)
    - Tax amount (Total ammount, not percentage)
    - Total amount

    Please provide ONLY the requested data in a structured format.
    """

    # Get the response from the LLM in a structured format
    llm_output: LLMStructuredResponse = (
        llm_client.get_llm_response_with_structured_response(
            prompt=prompt,
            schema=LLMStructuredResponse
        )
    )
    return llm_output


def save_output(llm_output: LLMStructuredResponse, save_path: str, save_name: str):

    # Convert the pydantic model to a JSON string
    json_output: str = llm_output.model_dump_json()

    with open(os.path.join(save_path, f"{save_name}.json"), "w") as f_json:
        f_json.write(json_output)

    # print(json_output)
    return json_output


if __name__ == "__main__":

    sys.path.append("/app")
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./TFG/dataset/FATURA/test")
    parser.add_argument("--save_path", type=str, default="./TFG/outputs/FATURA/orc_llm")
    parser.add_argument("--seed", type=int, default=42)
    args, left_argv = parser.parse_known_args()

    main(args)
