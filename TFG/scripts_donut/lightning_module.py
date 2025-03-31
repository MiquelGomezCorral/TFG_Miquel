import re
import time
import math
import json
import torch
import numpy as np
from pathlib import Path
from nltk import edit_distance
from typing import Callable, Iterable, Tuple

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


from TFG.scripts_donut.donut_utils import from_output_to_json
# from TFG.scripts_dataset.utils import print_time


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model, max_length, train_dataloader, val_dataloader, metrics: Iterable[Tuple[str, Callable[[dict,dict], float]]] = None):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.max_length = max_length
        if metrics is None: metrics = []
        self.metrics = metrics
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
        
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    
        predictions_json = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            seq = re.sub(r"(?:(?<=>) | (?=</s_))", "", seq)
            seq = self.processor.token2json(seq)
            
            predictions_json.append(seq)
        
        grount_truths_json = from_output_to_json(self.processor, grount_truths[0], decoded=True)

        scores = []
        # In case not more than one sample is passed per batchs 
        if not isinstance(grount_truths_json, list) and not isinstance(grount_truths_json, tuple):
            grount_truths_json = [grount_truths_json]
        if not isinstance(predictions_json, list) and not isinstance(predictions_json, tuple):
            predictions_json = [predictions_json]
            
        for gt, pred in zip(grount_truths_json, predictions_json):
            scores.append({
                "Normed_ED": edit_distance(json.dumps(gt), json.dumps(pred)) / max(len(json.dumps(gt)), len(json.dumps(pred))),
                **{name: metric(gt, pred) for name, metric in self.metrics}
            })

            if self.config.get("verbose", False):
                # print(f"\n VALIDATED SAMPLE NUMBER: {idx+1}:")
                print(f"\n - Ground Truth: {gt}")
                print(f" -   Prediction: {pred}")
                for name, metric in scores[0].items():
                    print(f" - Loss ({name}): {metric:0.4f}")

        self.log("val_edit_distance", np.mean([scs["Normed_ED"] for scs in scores]))
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