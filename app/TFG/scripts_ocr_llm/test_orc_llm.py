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
from TFG.scripts_dataset.utils import print_separator
from TFG.scripts_dataset.time_traker import parse_seconds_to_minutes, print_time, TimeTracker

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
    # shopping_or_tax: bool | None = Field(
    #     title="Shopping or Tax",
    #     description="True for commercial purchases, False for tax-related transactions. IF IT DOES NOT APPEAR, 'None' should be assigned to this values.",
    # )
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

    # ================ Initialize clients and aux ================
    print_separator("Initializing Clients and  variables...", sep_type="LONG")
    ocr_client: AzureDocumentIntelligenceClient = AzureDocumentIntelligenceClient()
    llm_client: AzureOpenAILanguageModel = AzureOpenAILanguageModel()
    
    TIME_TRACKER = TimeTracker(name="OCR LLM Testing")
    
    models: list[str] = [
        # f"ocr_finetuned_{i*5}x5_v1" for i in range(1,5+1)
        # "ocr_finetuned_5x5_v1",
        # "ocr_finetuned_4x5_v1",
        "ocr_finetuned_3x5_v1",
        "ocr_finetuned_5x2_v1",
        "ocr_finetuned_5x1_v1",
    ]
    
    # ================ Process files ================
    # for document in os.listdir(args.dataset_path):
    print_separator(f"Processing {n_files} files...", sep_type="LONG")
    
    ground_truths: list[dict] = []
    predictions: list[LLMStructuredResponse] = []

    TIME_TRACKER.start(verbose=False)
    TIME_TRACKER.start_lap(verbose=True)
    for model in models:
        TIME_TRACKER.track(model)
        
        save_path_model = os.path.join(args.save_path, model)
        MODEL_TIME_TRACKER = TimeTracker(name=model, start_track_now=True)
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in itertools.islice(f, args.max_files):
                # MODEL_TIME_TRACKER.track(tag="File start")
                
                document = json.loads(line)
                document_name = document["file_name"]
                ground_truths.append(
                    json.loads(document["ground_truth"])["gt_parse"]
                )
                document_path = os.path.join(args.dataset_path, document_name)
                print(f"\nDocument: {document_name}...")
                MODEL_TIME_TRACKER.start_lap(N=n_files)

                # Read invoice from file to bytesIO
                print(" - Preparing document...", end="\r")
                file_io = prepare_document(io.BytesIO(), document_path)
                MODEL_TIME_TRACKER.track(tag="Preparing document.", space=False)
                
                # Send document to ORC to extract content
                print(" - Extracting content with OCR...", end="\r")
                document_content, pages = document_to_orc(ocr_client, file_io, prebuilt_model=model)
                MODEL_TIME_TRACKER.track(tag="Extracting content.", space=False)
                
                print(f"{document_content = }")
                print(f"{pages = }")

                if args.llm:
                    # Define the prompt and send it to send to the LLM
                    print(" - Creating structured output with LLM...", end="\r")
                    llm_output = document_to_llm(llm_client, document_content)
                    predictions.append(json.loads(llm_output.model_dump_json()))
                    MODEL_TIME_TRACKER.track(tag="Creating structured output.", space=False)
                else:
                    predictions.append(json.loads(pages.model_dump_json()))
                    

                MODEL_TIME_TRACKER.stimate_lap_time(N=n_files)
                MODEL_TIME_TRACKER.finish_lap()
            # END FOR DOCUMENTS
            MODEL_TIME_TRACKER.print_metrics(n_files)
        # END WITH

        # OUTPUT MANAGEMENTj
        save_output(save_path_model, ground_truths, predictions)
        
        TIME_TRACKER.stimate_lap_time(N=len(models))
        TIME_TRACKER.finish_lap()
        os.makedirs(save_path_model, exist_ok=True)
        TIME_TRACKER.save_metric(os.path.join(save_path_model, "timing_log.txt"))
    # END FOR MODELS
    print_separator("DONE!", sep_type="LONG")


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


def document_to_orc(ocr_client: AzureDocumentIntelligenceClient, file_io: io.BytesIO, prebuilt_model: str = None):
    # IMPORTANT TO CHANGE THE TYPE OF DOCUMENT THAT THE OCR HAS TO READ.
    pages: list[str] = ocr_client.read_document(file_io, prebuilt_model=prebuilt_model)

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
    print("\nSaving output...", end="\r")
    json_output = {
        "ground_truths": ground_truths,
        "predictions": predictions
    }
    
    t_aux_4 = time.time()
    os.makedirs(save_path,exist_ok=True)
    with open(os.path.join(save_path, "output.json"), "w", encoding="utf-8") as out_json:
        json.dump(json_output, out_json, ensure_ascii=False, indent=4)
        
    t_aux_5 = time.time()
    print_time(t_aux_5 - t_aux_4, prefix="Saving output. ")
    
    # print(json_output)
    return json_output


if __name__ == "__main__":

    sys.path.append("/app")
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_path", type=str, default="./datasets_finetune/outputs/FATURA/test",
        help="Local path from ./app to the dataset."
    )
    parser.add_argument(
        "s", "--save_path", type=str, default=".TFG/outputs/ocr_llm",
        help="Local path from ./app to the folder where all the outputs will be placed."
    )
    parser.add_argument(
        "f", "--max_files", type=int, default=None,
        help="Max files loaded from the dataset"
    )
    parser.add_argument(
        "l", "--llm", type=bool, default=True, action="store_false"
        help="If to use an LLM to structure the output of the OCR. Default to 'True'"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Randomizer seed. Default to '42'"
    )
    args, left_argv = parser.parse_known_args()

    main(args)
