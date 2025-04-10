from typing import Dict, Callable, Iterable
from nltk import edit_distance
from dataclasses import dataclass, field, asdict

@dataclass
class Model_Config:
    # ================================================
    #                   FIELDS
    # ================================================
    max_epochs: int = 300
    val_check_interval: float = 0.25  # how many times we want to validate during an epoch
    check_val_every_n_epoch: int = 2
    gradient_clip_val: float = 1.0
    num_training_samples_per_epoch: int = 100
    lr: float = 3e-5
    train_batch_sizes: tuple[int] = (8,) #list[int] = field(default_factory=lambda: [8])
    val_batch_sizes: tuple[int] = (1,) #list[int] = field(default_factory=lambda: [1])
    
    num_nodes: int = 1
    warmup_steps: int = 10  # 10% of epochs
    
    seed: int = 42
    verbose: bool = True
    
    train_samples: int = None
    validation_samples: int = None
    test_samples: int = None
    
    
    metrics: Iterable[Callable[[dict, dict], float]] = (
        # ("Normed_ED", lambda ground_truth, prediction: edit_distance(ground_truth, prediction) / max(len(ground_truth), len(prediction)))
    )

    # ================================================
    #                   Methods
    # ================================================
    def update_from_dict(self, config_dict: Dict):
        """Method to update configuration from a dictionary

        Args:
            config_dict (Dict): Input dictionary
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # Convert the Config object to a dictionary
    def to_dict(self):
        """Generate a dict from the object

        Returns:
            dict: generated dict
        """
        return asdict(self)


@dataclass
class Config:
    model_trained_path: str = "./TFG/outputs/donut/model_output"
    model_prediction_path: str = "./TFG/outputs/donut/model_prediction"
    pretrained_model_name_or_path: str = "naver-clova-ix/donut-base"
    dataset_name_or_path: str = "datasets_finetune/outputs/FATURA"
    task_name: str = "fatrua_train"
    
    image_size: tuple[int, int] = (1280, 960)
    max_length: int = 384 # 128 * 3
    save_top_k: int = 1
    special_token: str = "<s_fatura>"
    # config: Model_Config = field(default_factory=Model_Config)  # Nested Config dataclass

    # Method to update Model_Config from a dictionary
    def update_from_dict(self, config_dict: Dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(value, dict) and hasattr(self, key):  # Handle nested dict for `config`
                    getattr(self, key).update_from_dict(value)
                else:
                    setattr(self, key, value)

    # Convert the Model_Config object to a dictionary
    def to_dict(self):
        return asdict(self)

