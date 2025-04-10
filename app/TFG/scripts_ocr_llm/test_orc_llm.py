#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) Solver Machine Learning -All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by SolverML <info@solverml.co>, Mar 2025
#
if __name__ == "__main__":
    import sys
    import dotenv
    sys.path.append("/app")
    dotenv.load_dotenv()
    
import os
import io
import json
import os.path
import time
import img2pdf  # type: ignore
import argparse
import itertools
from pydantic import BaseModel, Field

from ocr_llm_module.llm.azure.azure_openai import AzureOpenAILanguageModel
from ocr_llm_module.ocr.azure.document_intelligence import AzureDocumentIntelligenceClient

from TFG.scripts_dataset.utils import print_separator
from TFG.scripts_dataset.time_traker import parse_seconds_to_minutes, print_time, TimeTracker
from TFG.scripts_ocr_llm.llm import LLMStructuredResponse, document_to_llm
from TFG.scripts_ocr_llm.ocr import document_to_orc, get_fields_from_result

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
        # None,
        "ocr_finetuned_5x5_v1",
        "ocr_finetuned_4x5_v1",
        # "ocr_finetuned_3x5_v1",
        # "ocr_finetuned_5x2_v1",
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
        # BACKUP CHECKING IN CASE OF CRASH
        save_path_model_temp = os.path.join(save_path_model, "temp")
        os.makedirs(save_path_model_temp, exist_ok=True)
        
        MODEL_TIME_TRACKER = TimeTracker(name=model, start_track_now=True)
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in itertools.islice(f, args.max_files):
                document = json.loads(line)
                document_name = document["file_name"]
                
                ground_truths.append(
                    json.loads(document["ground_truth"])["gt_parse"]
                )
                
                # BACKUP CHECKING IN CASE OF CRASH
                temp_file_path = os.path.join(save_path_model_temp, f"{document_name}.json")
                if os.path.exists(temp_file_path):
                    with open(temp_file_path, "r") as temp:
                        predictions.append(json.load(temp))
                    print(f" SKIPPING: {document_name}. File found at {temp_file_path}")
                    MODEL_TIME_TRACKER.increase_lap_number()
                    continue
                
                
                # ACTUAL DOCUMENT PARSING
                document_path = os.path.join(args.dataset_path, document_name)
                print(f"\nDocument: {document_name}...")
                MODEL_TIME_TRACKER.start_lap(N=n_files)

                # Read invoice from file to bytesIO
                print(" - Preparing document...", end="\r")
                file_io = prepare_document(io.BytesIO(), document_path)
                MODEL_TIME_TRACKER.track(tag="Preparing document.", space=False)
                
                # Send document to ORC to extract content
                print(" - Extracting content with OCR...", end="\r")
                document_content, pages, fields_content, json_output  = document_to_orc(ocr_client, file_io, prebuilt_model=model)
                MODEL_TIME_TRACKER.track(tag="Extracting content.", space=False)
                
                if args.llm:
                    # Define the prompt and send it to send to the LLM
                    print(" - Creating structured output with LLM...", end="\r")
                    llm_output = document_to_llm(llm_client, document_content)
                    prediction = json.loads(llm_output.model_dump_json())
                    MODEL_TIME_TRACKER.track(tag="Creating structured output.", space=False)
                else:
                    prediction = json.loads(json_output)
                    
                predictions.append(prediction)
                # BACKUP IN CASE OF CRASH
                with open(temp_file_path, "w") as temp:
                    json.dump(prediction, temp)

                MODEL_TIME_TRACKER.stimate_lap_time(N=n_files)
                MODEL_TIME_TRACKER.finish_lap()
            # END FOR DOCUMENTS
            MODEL_TIME_TRACKER.print_metrics(n_files)
        # END WITH

        # OUTPUT MANAGEMENTj
        save_output(save_path_model, ground_truths, predictions, TIME_TRACKER)
        
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


def save_output(save_path, ground_truths, predictions, TIME_TRACKER):
    print("\nSaving output...", end="\r")
    json_output = {
        "ground_truths": ground_truths,
        "predictions": predictions
    }
    
    os.makedirs(save_path,exist_ok=True)
    with open(os.path.join(save_path, "output.json"), "w", encoding="utf-8") as out_json:
        json.dump(json_output, out_json, ensure_ascii=False, indent=4)
    
    TIME_TRACKER.track("Saving output.")
    # print(json_output)
    return json_output


if __name__ == "__main__":

    sys.path.append("/app")
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_path", type=str, default="./final_dataset_fatura/test",
        help="Local path from ./app to the dataset."
    )
    parser.add_argument(
        "-s", "--save_path", type=str, default="./TFG/outputs/ocr_llm",
        help="Local path from ./app to the folder where all the outputs will be placed."
    )
    parser.add_argument(
        "-f", "--max_files", type=int, default=None,
        help="Max files loaded from the dataset"
    )
    parser.add_argument(
        "-l", "--llm", default=True, action="store_false",
        help="If to use an LLM to structure the output of the OCR. Default to 'True'"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Randomizer seed. Default to '42'"
    )
    args, left_argv = parser.parse_known_args()

    main(args)