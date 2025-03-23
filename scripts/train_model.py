import os
import sys
from utils import print_separator, change_directory, print_time
change_directory()

import json
from datasets import load_dataset
import argparse
import time

from utils import print_separator
from sconf import Config
from train import train


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--config", type=str, required=False,
    default='config/train_cord.yaml'
  )
  parser.add_argument(
    "--pretrained_model_name_or_path", type=str, required=False,
    default="naver-clova-ix/donut-base"
  )
  parser.add_argument(
    "--dataset_name_or_path", type=str, required=False,
    default= f"[
      'dataset/new_generated_images/'
    ]" #"['naver-clova-ix/cord-v1']"
  )
  parser.add_argument(
    "--exp_version", type=str, required=False,
    default="test_experiment"
  )
  parser.add_argument(
    "--exp_name", type=str, required=False,
    default="test_experiment"
  )
  parser.add_argument(
    "--result_path", type=str, required=False,
    default='result/training/'
  )
  parser.add_argument(
    "--task_name", type=str, 
    default="fatura"
  )
  args, left_argv = parser.parse_known_args()

  if args.task_name is None:
      args.task_name = os.path.basename(args.dataset_name_or_path)

  config = Config(args.config)
  for k, v in vars(args).items():
      config[k] = v
      
  # ================== Training =========================
  t1 = time.time()
  print_separator(f'Training {args.task_name} for test...')
  
  train(config)
  
  t2 = time.time()
  diff = t2-t1
  print_time(diff, space=True)
  print_separator(f'DONE!')
  
  