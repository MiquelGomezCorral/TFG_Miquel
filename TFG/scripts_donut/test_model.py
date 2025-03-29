import os
import sys
curr_directory = os.getcwd()
print("\nOld Current Directory:", curr_directory)
if not curr_directory.endswith("TFG_Miquel"):
    os.chdir("../") 
    print("New Directory:", os.getcwd())
# if new_directory is not None and not curr_directory.endswith(new_directory):
#     os.chdir(f"./{new_directory}") 
#     print("New Directory:", os.getcwd(), "\n")
sys.path.append(os.getcwd())

from TFG.scripts_dataset.utils import print_separator, change_directory, print_time

import re
import json
import torch
import numpy as np
from tqdm.auto import tqdm

from donut import JSONParseEvaluator
from datasets import load_dataset

import argparse
import time


def test_model(model, processor):
    print_separator(f'TESTING', sep_type="SUPER")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    output_list = []
    accs = []

    dataset = load_dataset("naver-clova-ix/cord-v2", split="validation")

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # prepare encoder inputs
        pixel_values = processor(sample["image"].convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        # prepare decoder inputs
        task_prompt = "<s_fatura>"
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
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = processor.token2json(seq)

        ground_truth = json.loads(sample["ground_truth"])
        ground_truth = ground_truth["gt_parse"]
        evaluator = JSONParseEvaluator()
        score = evaluator.cal_acc(seq, ground_truth)

        accs.append(score)
        output_list.append(seq)

    scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
    print(scores, f"length : {len(accs)}")





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