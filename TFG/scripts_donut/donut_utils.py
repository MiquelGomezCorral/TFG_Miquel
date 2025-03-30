import re

import torch

def from_output_to_json(processor, outputs, decoded: bool = False) -> str:
    if not decoded:
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            # outputs = [outputs]
            seq = processor.batch_decode(outputs)[0]
        else:
            seq = processor.decode(outputs)
    else:
        seq = outputs
    
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
    seq = re.sub(r"(?:(?<=>) | (?=</s_))", "", seq)
    seq = processor.token2json(seq)
    return seq