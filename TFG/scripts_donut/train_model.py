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
# change_directory() #new_directory="donut"

import time
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, EarlyStopping

from test_model import test_model
from TFG.scripts_donut.tokenizer import DonutDataset, added_tokens
from TFG.scripts_donut.lightning_module import DonutModelPLModule

@dataclass
class Model_Config():
    image_size: list[int] = field(default_factory=lambda: [1280, 960]) 
    max_length: int = 768
    
    special_token: str = "<s_fatura>"

    config: dict = field(default_factory=lambda: {
        "max_epochs": 16,
        "val_check_interval": 0.2,  # how many times we want to validate during an epoch
        "check_val_every_n_epoch": 4,
        "gradient_clip_val": 1.0,
        "num_training_samples_per_epoch": 25,
        "lr": 3e-5,
        "train_batch_sizes": [8],
        "val_batch_sizes": [1],
        # "seed": 2022,
        "num_nodes": 1,
        "warmup_steps": 3,  # 10%
        "result_path": "./TFG/outputs/donut/model_output",
        "verbose": True,
    })

def train_model(args):
    MODEL_CONFIG = Model_Config()
    
    # =============================================================================
    #                            DATASET BASIC, NOT USED
    # =============================================================================
    # print_separator(f'Loadding dataset {args.dataset_name_or_path}...', sep_type="LONG")
    # dataset = load_dataset(args.dataset_name_or_path)
    
    
    # =============================================================================
    #                                   MODEL
    # =============================================================================
    print_separator(f'Loadding model {args.pretrained_model_name_or_path}...', sep_type="LONG")
    """update image_size of the encoder due to   during pre-training, a larger image size was used!"""
    
    config = VisionEncoderDecoderConfig.from_pretrained(args.pretrained_model_name_or_path)
    config.encoder.image_size = MODEL_CONFIG.image_size # (height, width)
    # update max_length of the decoder (for generation)
    config.decoder.max_length = MODEL_CONFIG.max_length
    # TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
    # https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602
    
    """CHECK FOR WARNING ABOUT THE WEIGHTS BEING WELL LOADED"""
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)
    

    # =============================================================================
    #                               DATASET PYTORCH
    # =============================================================================
    print_separator(f'Creating pytorch Dataset...', sep_type="LONG")
    """TAKE INTO ACCOUNT THAT WE MODIFY THE TOKENIZER"""
    processor.image_processor.size = MODEL_CONFIG.image_size[::-1] # should be (width, height)
    processor.image_processor.do_align_long_axis = False

    train_dataset = DonutDataset(
        dataset_name_or_path=args.dataset_name_or_path, 
        model=model,
        processor=processor,
        max_length=MODEL_CONFIG.max_length,
        split="train",
        task_start_token=MODEL_CONFIG.special_token,
        prompt_end_token=MODEL_CONFIG.special_token,
        sort_json_key=False, # cord dataset is preprocessed, so no need for this
    )

    val_dataset = DonutDataset(
        dataset_name_or_path=args.dataset_name_or_path, 
        model=model,
        processor=processor,
        max_length=MODEL_CONFIG.max_length,
        split="validation", 
        task_start_token=MODEL_CONFIG.special_token, 
        prompt_end_token=MODEL_CONFIG.special_token,
        sort_json_key=False, # cord dataset is preprocessed, so no need for this
    )
    
    print(f"Added {len(added_tokens)} tokens: \n - ", added_tokens)
    print(" - Original number of tokens:", processor.tokenizer.vocab_size)
    print(" - Number of tokens after adding special tokens:", len(processor.tokenizer))
    
    print_separator(f'Setting additional atributes...', sep_type="NORMAL")
    """nother important thing is that we need to set 2 additional attributes in the configuration of the model. This is not required, but will allow us to train the model by only providing the decoder targets, without having to provide any decoder inputs.
    The model will automatically create the `decoder_input_ids` (the decoder inputs) based on the `labels`, by shifting them one position to the right and prepending the decoder_start_token_id. I recommend checking [this video](https://www.youtube.com/watch?v=IGu7ivuy1Ag&t=888s&ab_channel=NielsRogge) if you want to understand how models like Donut automatically create decoder_input_ids - and more broadly how Donut works."""

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([MODEL_CONFIG.special_token])[0]
    
    print(" - Pad token ID:", processor.decode([model.config.pad_token_id]))
    print(" - Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))
    
    print_separator(f'Creating pytorch DataLoaders...', sep_type="LONG")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    
    # =============================================================================
    #                               TRAINING
    # =============================================================================
    print_separator(f'TRAINING', sep_type="SUPER")
    model_module = DonutModelPLModule(MODEL_CONFIG.config, processor, model, MODEL_CONFIG.max_length, train_dataloader, val_dataloader)
    
    wandb_logger = WandbLogger(project="Donut", name=args.task_name)
    early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
    trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=MODEL_CONFIG.config["max_epochs"], #get("max_epochs"),
            val_check_interval=MODEL_CONFIG.config["val_check_interval"], #get("val_check_interval"),
            check_val_every_n_epoch=MODEL_CONFIG.config["check_val_every_n_epoch"], #get("check_val_every_n_epoch"),
            gradient_clip_val=MODEL_CONFIG.config["gradient_clip_val"], #get("gradient_clip_val"),
            precision=16, # we'll use mixed precision
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[early_stop_callback],#[PushToHubCallback(), early_stop_callback],
    )

    trainer.fit(model_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    # =============================================================================
    #                               TESTING
    # =============================================================================
    # print_separator(f'TESTING', sep_type="SUPER")
    
    test_model(model, processor)



# =============================================================================
#                               MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=False,
        default="naver-clova-ix/donut-base"
    )
    parser.add_argument(
        "--dataset_name_or_path", type=str, required=False,
        default= f"datasets_finetune/outputs/FATURA" #"['naver-clova-ix/cord-v1']"
    )
    parser.add_argument(
        "--result_path", type=str, required=False,
        default='result/training/'
    )
    parser.add_argument(
        "--task_name", type=str, 
        default="fatura_train"
    )
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)
        
    # ================== Training =========================
    t1 = time.time()
    print_separator(f'Training {args.task_name}...', sep_type="SUPER")

    train_model(args)

    t2 = time.time()
    diff = t2-t1
    print_time(diff, space=True )
    print_separator(f'DONE!', sep_type="SUPER")

