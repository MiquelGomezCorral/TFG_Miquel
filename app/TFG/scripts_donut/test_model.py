import os
import sys

if __name__ == "__main__":
    curr_directory = os.getcwd()
    print("\nStarting Directory:", curr_directory)
    if not curr_directory.endswith("app"):
        if curr_directory.endswith("TFG_Miquel"):
            os.chdir("./app") 
        else: os.chdir("../") 
        print("New Directory:", os.getcwd())
    # if new_directory is not None and not curr_directory.endswith(new_directory):
    #     os.chdir(f"./{new_directory}") 
    #     print("New Directory:", os.getcwd(), "\n")
    sys.path.append(os.getcwd())

from TFG.utils.utils import print_separator, change_directory
from TFG.utils.time_traker import print_time
from TFG.scripts_dataset.validate_model import validate_model
from TFG.scripts_donut.donut_utils import from_output_to_json

import re
import time
import json
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from typing import Callable, Tuple

from donut import JSONParseEvaluator
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel



def test_model(
    model, processor, dataset_name_or_path, save_path, task_pront, 
    verbose: bool = True, max_samples: int = None,
    evaluators: list[Tuple[str, Callable[[dict,dict], float]]] = None
):
    if evaluators is None: evaluators = []
    evaluators = [('N_Tree_ED', lambda gt, pred: JSONParseEvaluator().cal_acc(pred, gt))] + evaluators
    
    print_separator(f'TESTING', sep_type="SUPER")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    output_list = []
    ground_truh_list = []
    scores_list = []

    dataset = load_dataset(dataset_name_or_path, split="test")
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # prepare encoder inputs
        pixel_values = processor(sample["image"].convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        # prepare decoder inputs
        task_prompt = task_pront
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(device)
        
        # autoregressively generate sequence
        outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # turn into JSON
        output_json = from_output_to_json(processor, outputs.sequences, second_reg=False)
        # seq = processor.batch_decode(outputs.sequences)[0]
        # seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        # seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        # seq = processor.token2json(seq)

        ground_truth_json = json.loads(sample["ground_truth"])
        ground_truth_json = ground_truth_json["gt_parse"]
        
        scores = {
            name: evaluator(ground_truth_json, output_json) for name, evaluator in evaluators 
        }

        scores_list.append(scores)
        output_list.append(output_json)
        ground_truh_list.append(ground_truth_json)


    scores = {
        "n_samples": len(scores_list),
        "mean_accuracy": np.mean([scs["N_Tree_ED"] for scs in scores_list]), 
        "ground_truths": ground_truh_list, 
        "predictions": output_list, 
        "accuracies": scores_list
    }
    
    validate_model(
        output_path=save_path,
        ground_truths=ground_truh_list,
        model_predictions=output_list,
        verbose=verbose
    )
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "predictions.json"), "w") as f:
            json.dump(scores, f)

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str,
        default="./result/training/fatura_train/fatura_train_0/"#"naver-clova-ix/donut-base-finetuned-cord-v2"
    )
    parser.add_argument(
        "--dataset_name_or_path", type=str,
        default="dataset/fatura/"  #"naver-clova-ix/cord-v1"
    )
    parser.add_argument(
        "--split", type=str,
        default="test"
    )
    parser.add_argument(
        "--task_name", type=str,
        default="fatura_test"
    )
    parser.add_argument(
        "--save_path", type=str,
        default="result/fatura_test/output.json"
    )

    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    # ================== PROCESSING =========================
    t1 = time.time()
    print_separator(f'Processing {args.task_name} for test...')

    processor = DonutProcessor.from_pretrained(args.pretrained_model_name_or_path) #"nielsr/donut-demo"
    model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model_name_or_path) #"nielsr/donut-demo"
    
    test_model(args)

    t2 = time.time()
    diff = t2-t1
    print_time(diff, space=True)
    print_separator(f'DONE!')