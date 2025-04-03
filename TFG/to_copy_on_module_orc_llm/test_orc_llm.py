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
    
import os
import io
import json
import time
import img2pdf  # type: ignore
import argparse
import itertools
from pydantic import BaseModel, Field
from ocr_llm_module.llm.azure.azure_openai import AzureOpenAILanguageModel
from ocr_llm_module.ocr.azure.document_intelligence import AzureDocumentIntelligenceClient
from utils import parse_seconds_to_minutes, print_separator, print_time, TimeTracker

# Define the structure of the response from the LLM
class LLMStructuredResponse(BaseModel):
    """
    Structure for the response from the LLM
    """
    buyer: str | None = Field(
        title="Buyer",
        description="The name of the buyer on the invoice. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    address: str | None = Field(
        title="Address",
        description="The address of the buyer (Strictly use the same commas and format as seen in the document). IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    date: str | None = Field(
        title="Date",
        description="The date of the invoice in YYYY-MM-DD format. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    shopping_or_tax: bool | None = Field(
        title="Shopping or Tax",
        description="True for commercial purchases, False for tax-related transactions. IF IT DOES NOT APPEAR, 'None' should be assigned to this values.",
    )
    currency: str | None = Field(
        title="Currency",
        description="The currency used in the invoice. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    subtotal: float | None = Field(
        title="Subtotal",
        description="The subtotal amount before discounts and tax. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    discount: float | None = Field(
        title="Discount",
        description="The discount applied to the invoice. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    tax: float | None = Field(
        title="Tax",
        description="The total tax applied to the invoice. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    total: float | None = Field(
        title="Total",
        description="The final total amount after tax and discounts. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )


def main(args):
    # ================ Check directory ================
    os.makedirs(args.save_path, exist_ok=True)

    metadata_path = os.path.join(args.dataset_path, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise KeyError(f"No 'metadata.jsonl' file found at {args.dataset_path}. Add one with the proper format")

    with open(metadata_path, "r", encoding="utf-8") as f:
        n_files = sum(1 for _ in itertools.islice(f, args.max_files))

    # ================ Initialize clients and aux================
    print_separator("Initializing Clients and  variables...", sep_type="LONG")
    llm_client: AzureOpenAILanguageModel = AzureOpenAILanguageModel()
    ocr_client: AzureDocumentIntelligenceClient = AzureDocumentIntelligenceClient()
    TIME_TRACKER = TimeTracker(name="OCR LLM Testing")
    # ================ Process files ================
    # for document in os.listdir(args.dataset_path):
    print_separator(f"Processing {n_files} files...", sep_type="LONG")

    count = 0
    total_time_preraring = .0
    total_time_extracting = .0
    total_time_structured = .0
    
    ground_truths: list[dict] = []
    predictions: list[LLMStructuredResponse] = []

    t_start = time.time()
    TIME_TRACKER.track(tag="Start")
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in itertools.islice(f, args.max_files):
            t_f_start = time.time()
            TIME_TRACKER.track(tag="File start", verbose=True)
            
            document = json.loads(line)
            document_name = document["file_name"]
            ground_truths.append(
                json.loads(document["ground_truth"])["gt_parse"]
            )
            document_path = os.path.join(args.dataset_path, document_name)
            print(f"\nDocument {count+1}/{n_files}: {document_name}...")

            # Read invoice from file to bytesIO
            print(" - Preparing document...", end="\r")
            file_io = prepare_document(io.BytesIO(), document_path)
            TIME_TRACKER.track(tag="- Preparing document.", verbose=True)
            
            print_time(diff, prefix="- Preparing document. ")

            # Send document to ORC to extract content
            print(" - Extracting content...", end="\r")
            document_content, pages = document_to_orc(ocr_client, file_io)
            TIME_TRACKER.track(tag="- Extracting content.", verbose=True)

            # Define the prompt and send it to send to the LLM
            print(" - Creating structured output...", end="\r")
            llm_output = document_to_llm(llm_client, document_content)
            predictions.append(json.loads(llm_output.model_dump_json()))
            
            TIME_TRACKER.track(tag=" - Creating structured output.", verbose=True)

            t_f_end = time.time()
            count += 1
            eta = (n_files - count) * (t_f_end - t_start) / count
            print_time(t_f_end - t_f_start, prefix="Total ", sufix=f". ETA: {parse_seconds_to_minutes(eta)}")

    # OUTPUT MANAGEMENT
    save_output(args.save_path, ground_truths, predictions)

    print_separator("DONE!", sep_type="LONG")

    t_aux_end = time.time()
    with open(os.path.join(args.save_path, "timing_log.txt"), "w") as log_file:
        print(f"Processed {count} files in total\n", file=log_file)
        print_time(t_aux_end - t_start, n_files=count, prefix="Total time: ", out_file=log_file)
        # print_time((t_start - t_aux_end), n_files=count, prefix="Average time File: ", out_file=log_file)
        print_time(total_time_preraring, n_files=count, prefix="Average time Preparing: ", out_file=log_file)
        print_time(total_time_extracting, n_files=count, prefix="Average time Extracting: ", out_file=log_file)
        print_time(total_time_structured, n_files=count, prefix="Average time Structured: ", out_file=log_file)


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


def save_output(save_path, ground_truths, predictions):
    print(" - Saving output...", end="\r")
    json_output = {
        "ground_truths": ground_truths,
        "predictions": predictions
    }
    
    t_aux_4 = time.time()
    with open(os.path.join(save_path, "output.json"), "w", encoding="utf-8") as out_json:
        json.dump(json_output, out_json, ensure_ascii=False, indent=4)
        
    t_aux_5 = time.time()
    print_time(t_aux_5 - t_aux_4, prefix=" - Saving output. ")
    
    # print(json_output)
    return json_output


if __name__ == "__main__":

    sys.path.append("/app")
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./datasets_finetune/outputs/FATURA/test")
    parser.add_argument("--save_path", type=str, default="./outputs/ocr_llm/FATURA_NEXT")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args, left_argv = parser.parse_known_args()

    main(args)
