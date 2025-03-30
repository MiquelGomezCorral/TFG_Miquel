import re
import time
import math
import torch
import numpy as np
from pathlib import Path
from nltk import edit_distance

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


from TFG.scripts_donut.donut_utils import from_output_to_json
# from TFG.scripts_dataset.utils import print_time


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model, max_length, metric, train_dataloader, val_dataloader):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.max_length = max_length
        self.metric = metric
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        
        self.scores_hist = []
        

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch
        
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        # t_start = time.time()
        pixel_values, labels, grount_truths = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)
        
        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=self.max_length,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)
    
        predictions_json = []
        # for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            # Now this is done with: from_output_to_json 
            # seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            # seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        # predictions_json.append(
        #     from_output_to_json(self.processor, outputs.sequences)
        # )
        # print(f"{grount_truths = }")
        # print(f"{outputs.sequences = }")
        # print(f"{type(outputs.sequences) = }")
        
        
        predictions_json = []
        # print(f"PRE {outputs.sequences = }")
        # print(f"PRE {type(outputs.sequences) = }") 
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            seq = re.sub(r"(?:(?<=>) | (?=</s_))", "", seq)
            seq = self.processor.token2json(seq)
            
            predictions_json.append(seq)
        # predictions_json = from_output_to_json(self.processor, outputs.sequences)
        # print(f"POS {predictions_json = }")
        # print(f"POS {type(predictions_json) = }")    
        
        
        # print(f"PRE {grount_truths[0] = }")
        # print(f"PRE {type(grount_truths[0]) = }")   
        grount_truths_json = from_output_to_json(self.processor, grount_truths[0], decoded=True)
        
        # print(f"POS {grount_truths_json = }")
        # print(f"POS {type(grount_truths_json) = }")   
        # for gt in grount_truths:
        #     grount_truths_json.append(
        #         from_output_to_json(self.processor, gt)
        #     )

        scores = []
        if not isinstance(grount_truths_json, list) and not isinstance(grount_truths_json, tuple):
            grount_truths_json = [grount_truths_json]
        if not isinstance(predictions_json, list) and not isinstance(predictions_json, tuple):
            predictions_json = [predictions_json]
            
        for gt, pred in zip(grount_truths_json, predictions_json):
            # NOT NEEDED ANYMORE
            # pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # gt = re.sub(r"<.*?>", "", gt, count=1)
            # gt = gt.replace(self.processor.tokenizer.eos_token, "")
            
            scores.append(
                self.metric(gt, pred)
                # edit_distance(pred, gt) / max(len(pred), len(gt))
            )

            if self.config.get("verbose", False):
                # print(f"\n VALIDATED SAMPLE NUMBER: {idx+1}:")
                print(f" -   Prediction: {pred}")
                print(f" - Ground Truth: {gt}")
                print(f" - Loss ({self.metric.__name__}): {scores[0]:0.4f}")

        self.log("val_edit_distance", np.mean(scores))
        # print_time(time.time()-t_start, n_files=len(predictions), prefix="samples validated in:")
        
        self.scores_hist.append(scores)
        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
    
        return optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader