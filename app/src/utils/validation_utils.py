<<<<<<< HEAD
import copy
import json
import string
from TFG.utils.metrics import check_date_value, levenshtein_distance, levenshtein_similarity, precision_recall_f1, token_precision_recall_f1


keys = ['all', 'buyer', 'address', 'date', 'currency', 'subtotal', 'discount', 'tax', 'total']
base_scores: dict[str, dict[str, float]] = {
    key: {
        "tp":0,"fp":0,"fn":0,
        "accuracy":0,
        "precision":0,"recall":0,"fscore":0, 
        "token_precision":0,"token_recall":0,"token_fscore":0
    } 
    for key in keys
}
"""
    tp: Those fields predicted and predicted PROPERLY.
    fp: Those fields predicted and predicted WRONG.
    tp: Those fields NOT predicted.
"""


def lowercase_keys(pred):
    if isinstance(pred, list):
        pred = pred[0]
    return {
        k.lower(): v for k, v in pred.items()
    }


def validate_prediction(gt, pred, verbose: bool = False):
    """
        Recieve json str or dict ground trugths and model predictions and outputs the corresponding metrics
    """
    scores = copy.deepcopy(base_scores)
    total_keys: int = len(gt)
    n_correct: int = 0
    mistaken_keys: list = list()
    
    # =============================
    #           CHECK INPUT
    # =============================
    if not isinstance(gt, dict):
        if len(gt) == 0: 
            print("WARNING, EMPTY 'Ground Truth':", gt)
            return scores, False, 0
        gt = json.loads(gt)
    if not isinstance(pred, dict):
        if len(pred) == 0: 
            print("WARNING, EMPTY 'Prediction':", pred)
            return scores, False, 0
        pred = json.loads(pred)

    # =============================
    #           UNIFY KEYS 
    # =============================
    gt = lowercase_keys(gt)
    pred = lowercase_keys(pred)
    # =============================
    #         VALIDATE ANSWER
    # =============================
    for key_gt, val_gt in gt.items():
        if key_gt not in pred:
            scores[key_gt].update({
                "tp": 0, "fp": 0, "fn": 1, # False negative because the field was there but the model didn't predict it
            })
            mistaken_keys.append(key_gt)
        else:
            correct = validate_answer(key_gt, val_gt, pred[key_gt])
            accuracy, token_precision, token_recall, token_fscore = token_precision_recall_f1(val_gt, pred[key_gt])
            
            if not correct: 
                mistaken_keys.append(key_gt)
            else:
                n_correct += 1 
            scores[key_gt].update({
                "tp": int(correct), "fp": int(not correct), "fn":0,
                "accuracy": accuracy,
                "token_precision": token_precision, "token_recall": token_recall, "token_fscore": token_fscore,
            })
    
    if verbose:
        print(" - Mistakes:", mistaken_keys)
    
    accuracy = n_correct / total_keys
    all_correct = len(mistaken_keys) == 0
    gt_str = " ".join([str(val) for val in gt.values()])
    pred_str = " ".join([str(val) for val in pred.values()])
    _, token_precision, token_recall, token_fscore = token_precision_recall_f1(gt_str, pred_str)
    
    scores["all"].update({
        "tp": sum(sub_scores["tp"] for k, sub_scores in scores.items() if k!='all'),
        "fp": sum(sub_scores["fp"] for k, sub_scores in scores.items() if k!='all'),
        "fn": sum(sub_scores["fn"] for k, sub_scores in scores.items() if k!='all'),
        "accuracy": accuracy,
        "token_precision": token_precision, "token_recall": token_recall, "token_fscore": token_fscore,
    })
    return scores, all_correct, accuracy, mistaken_keys
        

def validate_answer(key_gt, val_gt, val_pred) -> bool: 
    val_gt, val_pred = str(val_gt), str(val_pred)
    if isinstance(val_gt, str):
        val_gt = val_gt.lower()
    if isinstance(val_pred, str):
        val_pred = val_pred.lower()
    
    if key_gt == "date":
        return check_date_value(val_gt, val_pred, verbose=True)
    
    if key_gt == "currency":
        usd = ["$", "usd"]
        eur = ["€", "eur"]
        if val_gt in usd:
            return val_pred in usd
        elif val_gt in eur:
            return val_pred in eur
        else: return False
        
    if key_gt == "address":
        # Remove all puntuation
        val_gt =   " ".join(val_gt.split()).translate(str.maketrans('', '', string.punctuation))
        val_pred = " ".join(val_pred.split()).translate(str.maketrans('', '', string.punctuation))
    
    # THIS SHOULD NOT BE DONE: THE MODEL SHOULD BE ABLE TO SPECIFY IF A FIELD APPEARS OR NOT
    if key_gt in ["discount", "tax", "subtotal", "total"]:
        valid_values = [None, 'none', '', 0.0]
        if val_gt in valid_values:
            return val_pred in valid_values
        else:
            val_gt = float(val_gt)
            try:
                val_pred = float(val_pred)
            except ValueError as e:
                return False
    
    if key_gt == "shopping_or_tax":
        return True
        
    return val_gt == val_pred



# ======================================================
#                TORELANCE EDIT DISTANCE
# ======================================================



def validate_prediction_ed(gt, pred, edit_distance: int, verbose: bool = False):
    """
        Recieve json str or dict ground trugths and model predictions and outputs the corresponding metrics
    """
    scores = copy.deepcopy(base_scores)

    total_keys: int = len(gt)
    n_correct: int = 0
    mistaken_keys: list = list()
    
    # =============================
    #           CHECK INPUT
    # =============================
    if not isinstance(gt, dict):
        if len(gt) == 0: 
            print("WARNING, EMPTY 'Ground Truth':", gt)
            return scores, False, 0
        gt = json.loads(gt)
    if not isinstance(pred, dict):
        if len(pred) == 0: 
            print("WARNING, EMPTY 'Prediction':", pred)
            return scores, False, 0
        pred = json.loads(pred)

    # =============================
    #           UNIFY KEYS 
    # =============================
    gt = lowercase_keys(gt)
    pred = lowercase_keys(pred)

    # =============================
    #         VALIDATE ANSWER
    # =============================
    for key_gt, val_gt in gt.items():
        if key_gt not in pred:
            scores[key_gt].update({
                "tp": 0, "fp": 0, "fn": 1, # False negative because the field was there but the model didn't predict it
            })
            mistaken_keys.append(key_gt)
        else:
            correct = validate_answer_ed(key_gt, val_gt, pred[key_gt], edit_distance)
            accuracy, token_precision, token_recall, token_fscore = token_precision_recall_f1(val_gt, pred[key_gt])
            
            if not correct: 
                mistaken_keys.append(key_gt)
            else:
                n_correct += 1 
            scores[key_gt].update({
                "tp": int(correct), "fp": int(not correct), "fn": 0,
                "accuracy": accuracy,
                "token_precision": token_precision, "token_recall": token_recall, "token_fscore": token_fscore,
            })
    if verbose:
        print(" - Mistakes:", mistaken_keys)
    
    accuracy = n_correct / total_keys
    all_correct = len(mistaken_keys) == 0
    gt_str = " ".join([str(val) for val in gt.values()])
    pred_str = " ".join([str(val) for val in pred.values()])
    _, token_precision, token_recall, token_fscore = token_precision_recall_f1(gt_str, pred_str)
    
    scores["all"].update({
        "tp": sum(sub_scores["tp"] for k, sub_scores in scores.items() if k!='all'),
        "fp": sum(sub_scores["fp"] for k, sub_scores in scores.items() if k!='all'),
        "fn": sum(sub_scores["fn"] for k, sub_scores in scores.items() if k!='all'),
        "accuracy": accuracy,
        "token_precision": token_precision, "token_recall": token_recall, "token_fscore": token_fscore,
    })
    return scores, all_correct, accuracy, mistaken_keys
        
def validate_answer_ed(key_gt, val_gt, val_pred, edit_distance) -> bool: 
    """ Leivenstein distance applied to: Dates, address, buyer


    Args:
        key_gt (str): groun truth key
        val_gt (any): ground truth value
        val_pred (any): prediction value
        edit_distance (int): max edit distance accepted

    Returns:
        bool: if prediction is accepted or not
    """
    val_gt, val_pred = str(val_gt), str(val_pred)
    if isinstance(val_gt, str):
        val_gt = val_gt.lower()
    if isinstance(val_pred, str):
        val_pred = val_pred.lower()
    
    # THIS SHOULD NOT BE DONE: THE MODEL SHOULD BE ABLE TO SPECIFY IF A FIELD APPEARS OR NOT
    if key_gt in ["currency", "discount", "tax", "subtotal", "total", "shopping_or_tax"]:
        return validate_answer(key_gt, val_gt, val_pred)
    
    if key_gt == "date":
        check = check_date_value(val_gt, val_pred, verbose=True) 
        return check or levenshtein_distance(val_gt, val_pred) <= edit_distance
        
=======
import copy
import json
import string
from TFG.utils.metrics import check_date_value, levenshtein_distance, levenshtein_similarity, precision_recall_f1, token_precision_recall_f1


keys = ['all', 'buyer', 'address', 'date', 'currency', 'subtotal', 'discount', 'tax', 'total']
base_scores: dict[str, dict[str, float]] = {
    key: {
        "tp":0,"fp":0,"fn":0,
        "accuracy":0,
        "precision":0,"recall":0,"fscore":0, 
        "token_precision":0,"token_recall":0,"token_fscore":0
    } 
    for key in keys
}
"""
    tp: Those fields predicted and predicted PROPERLY.
    fp: Those fields predicted and predicted WRONG.
    tp: Those fields NOT predicted.
"""


def lowercase_keys(pred):
    if isinstance(pred, list):
        pred = pred[0]
    return {
        k.lower(): v for k, v in pred.items()
    }


def validate_prediction(gt, pred, verbose: bool = False):
    """
        Recieve json str or dict ground trugths and model predictions and outputs the corresponding metrics
    """
    scores = copy.deepcopy(base_scores)
    total_keys: int = len(gt)
    n_correct: int = 0
    mistaken_keys: list = list()
    
    # =============================
    #           CHECK INPUT
    # =============================
    if not isinstance(gt, dict):
        if len(gt) == 0: 
            print("WARNING, EMPTY 'Ground Truth':", gt)
            return scores, False, 0
        gt = json.loads(gt)
    if not isinstance(pred, dict):
        if len(pred) == 0: 
            print("WARNING, EMPTY 'Prediction':", pred)
            return scores, False, 0
        pred = json.loads(pred)

    # =============================
    #           UNIFY KEYS 
    # =============================
    gt = lowercase_keys(gt)
    pred = lowercase_keys(pred)
    # =============================
    #         VALIDATE ANSWER
    # =============================
    for key_gt, val_gt in gt.items():
        if key_gt not in pred:
            scores[key_gt].update({
                "tp": 0, "fp": 0, "fn": 1, # False negative because the field was there but the model didn't predict it
            })
            mistaken_keys.append(key_gt)
        else:
            correct = validate_answer(key_gt, val_gt, pred[key_gt])
            accuracy, token_precision, token_recall, token_fscore = token_precision_recall_f1(val_gt, pred[key_gt])
            
            if not correct: 
                mistaken_keys.append(key_gt)
            else:
                n_correct += 1 
            scores[key_gt].update({
                "tp": int(correct), "fp": int(not correct), "fn":0,
                "accuracy": accuracy,
                "token_precision": token_precision, "token_recall": token_recall, "token_fscore": token_fscore,
            })
    
    if verbose:
        print(" - Mistakes:", mistaken_keys)
    
    accuracy = n_correct / total_keys
    all_correct = len(mistaken_keys) == 0
    gt_str = " ".join([str(val) for val in gt.values()])
    pred_str = " ".join([str(val) for val in pred.values()])
    _, token_precision, token_recall, token_fscore = token_precision_recall_f1(gt_str, pred_str)
    
    scores["all"].update({
        "tp": sum(sub_scores["tp"] for k, sub_scores in scores.items() if k!='all'),
        "fp": sum(sub_scores["fp"] for k, sub_scores in scores.items() if k!='all'),
        "fn": sum(sub_scores["fn"] for k, sub_scores in scores.items() if k!='all'),
        "accuracy": accuracy,
        "token_precision": token_precision, "token_recall": token_recall, "token_fscore": token_fscore,
    })
    return scores, all_correct, accuracy, mistaken_keys
        

def validate_answer(key_gt, val_gt, val_pred) -> bool: 
    val_gt, val_pred = str(val_gt), str(val_pred)
    if isinstance(val_gt, str):
        val_gt = val_gt.lower()
    if isinstance(val_pred, str):
        val_pred = val_pred.lower()
    
    if key_gt == "date":
        return check_date_value(val_gt, val_pred, verbose=True)
    
    if key_gt == "currency":
        usd = ["$", "usd"]
        eur = ["€", "eur"]
        if val_gt in usd:
            return val_pred in usd
        elif val_gt in eur:
            return val_pred in eur
        else: return False
        
    if key_gt == "address":
        # Remove all puntuation
        val_gt =   " ".join(val_gt.split()).translate(str.maketrans('', '', string.punctuation))
        val_pred = " ".join(val_pred.split()).translate(str.maketrans('', '', string.punctuation))
    
    # THIS SHOULD NOT BE DONE: THE MODEL SHOULD BE ABLE TO SPECIFY IF A FIELD APPEARS OR NOT
    if key_gt in ["discount", "tax", "subtotal", "total"]:
        valid_values = [None, 'none', '', 0.0]
        if val_gt in valid_values:
            return val_pred in valid_values
        else:
            val_gt = float(val_gt)
            try:
                val_pred = float(val_pred)
            except ValueError as e:
                return False
    
    if key_gt == "shopping_or_tax":
        return True
        
    return val_gt == val_pred



# ======================================================
#                TORELANCE EDIT DISTANCE
# ======================================================



def validate_prediction_ed(gt, pred, edit_distance: int, verbose: bool = False):
    """
        Recieve json str or dict ground trugths and model predictions and outputs the corresponding metrics
    """
    scores = copy.deepcopy(base_scores)

    total_keys: int = len(gt)
    n_correct: int = 0
    mistaken_keys: list = list()
    
    # =============================
    #           CHECK INPUT
    # =============================
    if not isinstance(gt, dict):
        if len(gt) == 0: 
            print("WARNING, EMPTY 'Ground Truth':", gt)
            return scores, False, 0
        gt = json.loads(gt)
    if not isinstance(pred, dict):
        if len(pred) == 0: 
            print("WARNING, EMPTY 'Prediction':", pred)
            return scores, False, 0
        pred = json.loads(pred)

    # =============================
    #           UNIFY KEYS 
    # =============================
    gt = lowercase_keys(gt)
    pred = lowercase_keys(pred)

    # =============================
    #         VALIDATE ANSWER
    # =============================
    for key_gt, val_gt in gt.items():
        if key_gt not in pred:
            scores[key_gt].update({
                "tp": 0, "fp": 0, "fn": 1, # False negative because the field was there but the model didn't predict it
            })
            mistaken_keys.append(key_gt)
        else:
            correct = validate_answer_ed(key_gt, val_gt, pred[key_gt], edit_distance)
            accuracy, token_precision, token_recall, token_fscore = token_precision_recall_f1(val_gt, pred[key_gt])
            
            if not correct: 
                mistaken_keys.append(key_gt)
            else:
                n_correct += 1 
            scores[key_gt].update({
                "tp": int(correct), "fp": int(not correct), "fn": 0,
                "accuracy": accuracy,
                "token_precision": token_precision, "token_recall": token_recall, "token_fscore": token_fscore,
            })
    if verbose:
        print(" - Mistakes:", mistaken_keys)
    
    accuracy = n_correct / total_keys
    all_correct = len(mistaken_keys) == 0
    gt_str = " ".join([str(val) for val in gt.values()])
    pred_str = " ".join([str(val) for val in pred.values()])
    _, token_precision, token_recall, token_fscore = token_precision_recall_f1(gt_str, pred_str)
    
    scores["all"].update({
        "tp": sum(sub_scores["tp"] for k, sub_scores in scores.items() if k!='all'),
        "fp": sum(sub_scores["fp"] for k, sub_scores in scores.items() if k!='all'),
        "fn": sum(sub_scores["fn"] for k, sub_scores in scores.items() if k!='all'),
        "accuracy": accuracy,
        "token_precision": token_precision, "token_recall": token_recall, "token_fscore": token_fscore,
    })
    return scores, all_correct, accuracy, mistaken_keys
        
def validate_answer_ed(key_gt, val_gt, val_pred, edit_distance) -> bool: 
    """ Leivenstein distance applied to: Dates, address, buyer


    Args:
        key_gt (str): groun truth key
        val_gt (any): ground truth value
        val_pred (any): prediction value
        edit_distance (int): max edit distance accepted

    Returns:
        bool: if prediction is accepted or not
    """
    val_gt, val_pred = str(val_gt), str(val_pred)
    if isinstance(val_gt, str):
        val_gt = val_gt.lower()
    if isinstance(val_pred, str):
        val_pred = val_pred.lower()
    
    # THIS SHOULD NOT BE DONE: THE MODEL SHOULD BE ABLE TO SPECIFY IF A FIELD APPEARS OR NOT
    if key_gt in ["currency", "discount", "tax", "subtotal", "total", "shopping_or_tax"]:
        return validate_answer(key_gt, val_gt, val_pred)
    
    if key_gt == "date":
        check = check_date_value(val_gt, val_pred, verbose=True) 
        return check or levenshtein_distance(val_gt, val_pred) <= edit_distance
        
>>>>>>> db2f1eff (changed root folder by default for imports)
    return levenshtein_distance(val_gt, val_pred) <= edit_distance