import os
from utils import print_separator, change_directory, print_time
change_directory()
  
import argparse
import time

from test_donut import test


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
  
  test(args)
  
  t2 = time.time()
  diff = t2-t1
  print_time(diff, space=True)
  print_separator(f'DONE!')