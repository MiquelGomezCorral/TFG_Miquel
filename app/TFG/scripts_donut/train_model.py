import os
import sys
import argparse
from dotenv import load_dotenv

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
    
    load_dotenv()
    
    # ============================================================ 
    #                   Parse arguments
    # ============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--pretrained_model_name_or_path", type=str, required=False, default="naver-clova-ix/donut-base")
    parser.add_argument("-d", "--dataset_name_or_path", type=str, required=False, default= f"final_dataset_fatura") #"['naver-clova-ix/cord-v1']"
    parser.add_argument("-o", "--result_path", type=str, required=False, default='./TFG/outputs/donut')
    parser.add_argument("-n", "--task_name", type=str, default="fatura_train")
    parser.add_argument("-k", "--make_me_a_donut", action="store_false", default=True)
    parser.add_argument("-b", "--boom_folders", action="store_false", default=True)
    
    parser.add_argument(
        "-tr", "--train_samples", type=int, default=None,
        help="Number of samples for testing, 'None' will take al much as possible"
    )
    parser.add_argument(
        "-va", "--validation_samples", type=int, default=None,
        help="Number of samples for validation, 'None' will take al much as possible"
    )
    parser.add_argument(
        "-ts", "--test_samples", type=int, default=None,
        help="Number of samples for test, 'None' will take al much as possible"
    )
    
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)
        
        
    # ============================================================ 
    #      Start the donut animation in a separated process
    # ============================================================
    if args.make_me_a_donut:
        import multiprocessing
        from TFG.scripts_donut.donut_print import print_donut, clean_all
            
        stop_event = multiprocessing.Event()
        donut_process = multiprocessing.Process(
            target=lambda: print_donut(infinite=True),
            daemon=True # So if the main program stops it also does
        )
        donut_process.start()

import json
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint


from test_model import test_model
from TFG.scripts_donut.donut_utils import clear_folder
from TFG.scripts_donut.config import Config, Model_Config
from TFG.scripts_donut.lightning_module import DonutModelPLModule
from TFG.scripts_donut.tokenizer import DonutDataset, added_tokens
from TFG.utils.utils import print_separator, change_directory
from TFG.utils.time_traker import TimeTracker
from TFG.utils.validation_utils import validate_prediction


# =============================================================================
#                            TRAIN SCRIP UTILS
# =============================================================================
def fatura_metric(ground_truth, prediction, verbose: bool = False):
    return 1-validate_prediction(ground_truth, prediction, verbose = verbose)[2]

def train_model_from_args(args):
    # ================== Clear repo =========================
    if args.boom_folders:
        clear_folder(folder="./temp")
        clear_folder(folder="./wandb")
    # ================== Training =========================
    print_separator(f'Training {args.task_name}', sep_type="SUPER")

    train_model(args)

    print_separator(f'DONE {args.task_name}!', sep_type="SUPER")

    
# =============================================================================
#                            TRAIN SCRIP
# =============================================================================

def train_model(args):
    CONFIG = Config(
        model_trained_path = args.result_path,
        model_prediction_path = args.result_path,
        pretrained_model_name_or_path = args.pretrained_model_name_or_path,
        dataset_name_or_path = args.dataset_name_or_path,
        task_name = args.task_name,
    )
    MODEL_CONFIG = Model_Config(
        # Partial so the function can be called later but with that parameter set
        train_samples = args.train_samples,
        test_samples = args.test_samples,
        validation_samples = args.validation_samples,
        metrics = [("fatura_metric", partial(fatura_metric))] 
    )
    
    os.makedirs(args.result_path, exist_ok=True)
    
    TIME_TRAKER = TimeTracker(name="Training")
    TIME_TRAKER.start(verbose=False)
    
    # =============================================================================
    #                            DATASET BASIC, NOT USED
    # =============================================================================
    # TIME_TRAKER.track("Basic dataset load")
    # print_separator(f'Loadding dataset {args.dataset_name_or_path}...', sep_type="LONG")
    # dataset = load_dataset(args.dataset_name_or_path)
    
    # =============================================================================
    #                                   MODEL
    # =============================================================================
    print_separator(f'Loadding model {CONFIG.pretrained_model_name_or_path}...', sep_type="LONG")
    """update image_size of the encoder due to   during pre-training, a larger image size was used!"""
    
    vision_encoder_config = VisionEncoderDecoderConfig.from_pretrained(CONFIG.pretrained_model_name_or_path)
    vision_encoder_config.encoder.image_size = CONFIG.image_size # (height, width)
    # update max_length of the decoder (for generation)
    vision_encoder_config.decoder.max_length = CONFIG.max_length
    # TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
    # https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602
    
    """CHECK FOR WARNING ABOUT THE WEIGHTS BEING WELL LOADED"""
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=vision_encoder_config)
    TIME_TRAKER.track("Getting Model")
    
    
    # =============================================================================
    #                               DATASET PYTORCH
    # =============================================================================
    print_separator(f'Creating pytorch Dataset...', sep_type="LONG")
    """TAKE INTO ACCOUNT THAT WE MODIFY THE TOKENIZER"""
    processor.image_processor.size = CONFIG.image_size[::-1] # should be (width, height)
    processor.image_processor.do_align_long_axis = False

    train_dataset = DonutDataset(
        dataset_name_or_path=CONFIG.dataset_name_or_path, 
        model=model,
        processor=processor,
        max_length=CONFIG.max_length,
        split="train",
        task_start_token=CONFIG.special_token,
        prompt_end_token=CONFIG.special_token,
        sort_json_key=False, # Our Fatura dataset is preprocessed, so no need for this
        max_samples=MODEL_CONFIG.train_samples,
    )

    val_dataset = DonutDataset(
        dataset_name_or_path=CONFIG.dataset_name_or_path, 
        model=model,
        processor=processor,
        max_length=CONFIG.max_length,
        split="validation", 
        task_start_token=CONFIG.special_token, 
        prompt_end_token=CONFIG.special_token,
        sort_json_key=False, # Our Fatura dataset is preprocessed, so no need for this
        max_samples=MODEL_CONFIG.validation_samples,
    )
    
    print_separator(f'Special Tokens', sep_type="NORMAL")
    print(f"Added {len(added_tokens)} tokens: \n - ", added_tokens)
    print(" - Original number of tokens:", processor.tokenizer.vocab_size)
    print(" - Number of tokens after adding special tokens:", len(processor.tokenizer))
    
    TIME_TRAKER.track("Creating pythorch Dataset")
    
    
    print_separator(f'Setting additional atributes...', sep_type="NORMAL")
    """nother important thing is that we need to set 2 additional attributes in the configuration of the model. This is not required, but will allow us to train the model by only providing the decoder targets, without having to provide any decoder inputs.
    The model will automatically create the `decoder_input_ids` (the decoder inputs) based on the `labels`, by shifting them one position to the right and prepending the decoder_start_token_id. I recommend checking [this video](https://www.youtube.com/watch?v=IGu7ivuy1Ag&t=888s&ab_channel=NielsRogge) if you want to understand how models like Donut automatically create decoder_input_ids - and more broadly how Donut works."""

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([CONFIG.special_token])[0]
    
    print(" - Pad token ID:", processor.decode([model.config.pad_token_id]))
    print(" - Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))
    
    TIME_TRAKER.track("Setting additional atributes")
    
    
    print_separator(f'Creating pytorch Data Loaders...', sep_type="LONG")
    print(" - Creating Training Dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    print(" - Creating Validation Dataloader...")
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    TIME_TRAKER.track("Creating pytorch Data Loaders")
    
    # =============================================================================
    #                               TRAINING
    # =============================================================================
    print_separator(f'TRAINING {args.task_name}', sep_type="SUPER")
    model_module = DonutModelPLModule(
        config=MODEL_CONFIG.to_dict(), 
        processor=processor, 
        model=model, 
        max_length=CONFIG.max_length, 
        metrics=MODEL_CONFIG.metrics,
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader
    )
    
    wandb_logger = WandbLogger(project="Donut", name=CONFIG.task_name)
    early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
    
    os.makedirs('./temp', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_edit_distance',  # monitor metric validation loss
        mode='min',  # We want to minimize the edit distance
        save_top_k=CONFIG.save_top_k,  # 1 to Only save the best model
        save_last=False,  # Do not save the last model 
        dirpath='./temp',  # Directory to store checkpoints
        filename='donut_champion', 
    )
    trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=MODEL_CONFIG.max_epochs, #get("max_epochs"),
            val_check_interval=MODEL_CONFIG.val_check_interval, #get("val_check_interval"),
            check_val_every_n_epoch=MODEL_CONFIG.check_val_every_n_epoch, #get("check_val_every_n_epoch"),
            gradient_clip_val=MODEL_CONFIG.gradient_clip_val, #get("gradient_clip_val"),
            precision=16, # we'll use mixed precision
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],#[PushToHubCallback(), early_stop_callback],
    )

    trainer.fit(model_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    TIME_TRAKER.track("Training")
    
    # =============================================================================
    #                               TESTING
    # =============================================================================
    # print_separator(f'TESTING', sep_type="SUPER")
    
    # Get the scores if you want to do something with them. But by providing saving path they are saved automatically
    scores = test_model(
        model, processor, 
        save_path = CONFIG.model_prediction_path,
        dataset_name_or_path = CONFIG.dataset_name_or_path, 
        task_pront = CONFIG.special_token,
        evaluators = [("fatura_metric", partial(fatura_metric, verbose=False))], # MODEL_CONFIG.metrics
        max_samples = MODEL_CONFIG.test_samples,
    )
    
    TIME_TRAKER.track("Testing")
    
    # =============================================================================
    #                               TIMING
    # =============================================================================
    print_separator(f'TIMING {args.task_name}', sep_type="SUPER")

    metrics = TIME_TRAKER.print_metrics()
    with open(os.path.join(args.result_path, "timing.txt"), "w") as f:
        TIME_TRAKER.print_metrics(out_file = f)
    with open(os.path.join(args.result_path, "timing.json"), "w") as f:
        json.dump(metrics, f)


# =============================================================================
#                               MAIN
# =============================================================================
if __name__ == "__main__":
    # ================== Silly donut ======================
    if args.make_me_a_donut:
        stop_event.set()  # Signal the animation to stop
        donut_process.terminate()  # Kill the process
        donut_process.join()  # Ensure cleanup
        clean_all()
    
    # ================== Train ======================
    train_model_from_args(args)

