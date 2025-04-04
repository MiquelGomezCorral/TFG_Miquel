import re
import os
import torch
import shutil

def from_output_to_json(processor, outputs, decoded: bool = False, second_reg: bool = True, remove_first_tag: bool = True) -> str:
    if not decoded:
        if isinstance(outputs, list) or isinstance(outputs, tuple) or isinstance(outputs, torch.Tensor):
            # outputs = [outputs]
            seq = processor.batch_decode(outputs)[0]
        else:
            seq = processor.decode(outputs)
    else:
        seq = outputs
    
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    if remove_first_tag:
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
    if second_reg:
        seq = re.sub(r"(?:(?<=>) | (?=</s_))", "", seq)
    seq = processor.token2json(seq)
    return seq

def clear_folder(folder='./temp'):
    """Clear the contents of the temporary folder."""
    if os.path.exists(folder) and os.path.isdir(folder):
        # Remove all contents in the temp directory
        shutil.rmtree(folder)
        print(f"Cleared the {folder} folder.")
    else:
        print(f"The directory {folder} does not exist.")