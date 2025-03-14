import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

# ===========================================
#             Factura class
# ===========================================

class Factura:
    def __init__(self):
        self.buyer: str = ""
        self.address: str = ""
        self.date: str = ""
        self.shopping_or_tax: bool = None  # True for Shopping, False for Tax
        self.currency: str = ""
        self.subtotal: float = 0.0
        self.discount: float = 0.0
        self.tax: float = 0.0
        self.total: float = 0.0
        self.products: List[Product] = []

    def __repr__(self):
        return f"Factura(buyer={self.buyer}, address={self.address}, date={self.date}, shopping_or_tax={self.shopping_or_tax}, currency={self.currency}, subtotal={self.subtotal}, discount={self.discount}, tax={self.tax}, total={self.total}, products={self.products})"

    def to_dict(self):
        """
        Convert the Factura object to a dictionary, including converting products to dictionaries.
        """
        return {
            "buyer": self.buyer,
            "address": self.address,
            "date": self.date,
            "shopping_or_tax": self.shopping_or_tax,
            "currency": self.currency,
            "subtotal": self.subtotal,
            "discount": self.discount,
            "tax": self.tax,
            "total": self.total,
            "products": [Product.convert_to_dict(product) for product in self.products]  # Convert products to dictionaries
        }

    @classmethod  
    def from_dict(cls, data: dict):
        """
        Create a Factura object from a dictionary.
        """
        factura = cls()
        factura.buyer = data.get("buyer", "")
        factura.address = data.get("address", "")
        factura.date = data.get("date", "")
        factura.shopping_or_tax = data.get("shopping_or_tax", None)
        factura.currency = data.get("currency", "")
        factura.subtotal = data.get("subtotal", 0.0)
        factura.discount = data.get("discount", 0.0)
        factura.tax = data.get("tax", 0.0)
        factura.total = data.get("total", 0.0)
        factura.products = data.get("products", [])
        return factura
        
class Product:
    def __init__(self):
        self.description: str = ""
        self.quntity: int = 0
        self.unite_price: float = 0.0
        self.total_price: float = 0.0
    
    def to_dict(self):
        """
        Convert the Product object to a dictionary.
        """
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total_price": self.total_price
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Product":
        """
        Create a Product instance from a dictionary.
        
        Args:
            data (Dict[str, Any]): A dictionary containing the attributes to populate the Product object.

        Returns:
            Product: A new Product instance populated with the data from the dictionary.
        """
        product = cls()
        product.description = data.get("description", "")
        product.quantity = data.get("quantity", 0)
        product.unit_price = data.get("unit_price", 0.0)
        product.total_price = data.get("total_price", 0.0)
        return product
    
    @staticmethod
    def convert_to_dict(item: Any) -> Dict[str, Any]:
        """
        Converts a Product object to a dictionary or returns the item if it's already a dictionary.
        
        Args:
            item (Any): Either a Product object or a dictionary to be converted.
            
        Returns:
            Dict[str, Any]: The dictionary representation of the item.
        """
        if isinstance(item, Product):
            return item.to_dict()
        elif isinstance(item, dict):
            return item  # If it's already a dictionary, return it as-is.
        else:
            raise ValueError("Item must be either a Product object or a dictionary.")
        

# ===========================================
#             Management logic
# ===========================================

def load_ensure_parse_save_files(args) -> None:
    """
    Loads files from the specified dataset path, extracts relevant data, 
    and saves the parsed results.

    Args:
        args: An object containing the following attributes:
            - dataset_path (str): Path to the dataset folder.
            - save_path (str): Path where parsed files will be saved.
            - seed (int): Random seed for reproducibility.
            - n_files (int): Number of files to process (optional).

    Returns:
        None
    """
    random.seed(args.seed)

    pre_parsed_files: List[tuple] = []
    files: List[str] = os.listdir(args.dataset_path)  # Lists all files and folders
    
    # file_ids = random.choices(list(range(200)), k=args.n_files) # Instances go from 0 to 199
    
    print(f"Processing {len(files)} Files...")
    for i, file_name in enumerate(files):
        file_path = os.path.join(args.dataset_path, file_name)
        with open(file_path) as f:
            print(f" - {i+1}: Processing {file_name}...")
            
            data = json.load(f)            
            pre_parsed_files.append((file_name, data["original_data"]))

    print(f"\nSaving files...")
    
    save_and_parse_files(pre_parsed_files, args.save_path, args.test_split)

    print(f"DONE!")


def save_and_parse_files(pre_parsed_files: List[Tuple["str", Factura]], save_path: str, test_split: Optional[float] = None) -> None:
    """
    Saves and parses files, either saves then on a single folder or with a test/train split.

    Args:
        pre_parsed_files (List[Tuple[str, Factura]]): List of tuples containing file names and parsed data.
        save_path (str): Path where files should be saved.
        test_split (Optional[float]): Fraction of data to be used for testing (if applicable).

    Returns:
        None
    """    
    if test_split is None:
        save_and_parse_files_normal(pre_parsed_files, save_path)
    else: 
        save_and_parse_files_split(pre_parsed_files, save_path, test_split)
        
        
# ===========================================
#             Saving Logic
# ===========================================

def save_and_parse_files_normal(pre_parsed_files: List[Tuple["str", Factura]], save_path: str) -> None:
    """
    Saves and parses files in a single folder for both readable and metadata data.

    Args:
        pre_parsed_files (List[Tuple[str, Factura]]): A list of tuples where each tuple contains 
                                                   the file name and its parsed data.
        save_path (str): The directory where the files should be saved.

    Returns:
        None
    """
    save_path_redeable: str = os.path.join(save_path, "redeable")
    save_path_metadata: str = os.path.join(save_path, "metadata")
    file_path_metadata: str = os.path.join(save_path_metadata, "metadata.jsonl")
    
    os.makedirs(save_path_metadata, exist_ok=True)        
    os.makedirs(save_path_redeable, exist_ok=True)        
        
    with open(file_path_metadata, "w") as out_metadata:
        for file_name, pre_parsed_data in pre_parsed_files:
            pre_parsed_data = pre_parsed_data if isinstance(pre_parsed_data, Factura) else Factura.from_dict(pre_parsed_data) 
            
            file_name_jpg: str = ".".join(file_name.split(".")[:-1]) + ".jpg" # Change extension of the file name to jpg
            
            parsed_data_redeable: dict = parse_file_redeable(file_name_jpg, pre_parsed_data)
            parsed_data_metadata: dict = parse_file_metadata(file_name_jpg, pre_parsed_data)
        
            file_path_redeable = os.path.join(save_path_redeable, file_name)   
            with open(file_path_redeable, "w") as out_redeable:
                json.dump(parsed_data_redeable, out_redeable, indent=4, ensure_ascii=False)
                
            out_metadata.write(json.dumps(parsed_data_metadata) + "\n")
            
            
def save_and_parse_files_split(pre_parsed_files: List[Tuple["str",Factura]], save_path: str, test_split: float) -> None:
    """
    Saves, parses and splits the files into training and testing sets, storing each set
    in different folders for both readable and metadata data.

    Args:
        pre_parsed_files (List[Tuple[str, Factura]]): A list of tuples where each tuple contains 
                                                   the file name and its parsed data.
        save_path (str): The directory where the files should be saved.
        test_split (float): The fraction of data to be used for the test set.

    Returns:
        None
    """
    save_path_redeable: str = os.path.join(save_path, "redeable")
    save_path_metadata_test: str = os.path.join(save_path, "test")
    save_path_metadata_train: str = os.path.join(save_path, "train")
    
    os.makedirs(save_path_redeable, exist_ok=True)        
    os.makedirs(save_path_metadata_test, exist_ok=True)
    os.makedirs(save_path_metadata_train, exist_ok=True)

        
    n: int = int(len(pre_parsed_files) * test_split)
    test_pre_parsed_files: List[Tuple[str, dict]] = pre_parsed_files[:n]
    train_pre_parsed_files: List[Tuple[str, dict]] = pre_parsed_files[n:]
    
        
    with open(os.path.join(save_path_metadata_test, "metadata.jsonl"), "w") as out_metadata:
        for file_name, pre_parsed_data in test_pre_parsed_files:
            pre_parsed_data = pre_parsed_data if isinstance(pre_parsed_data, Factura) else Factura.from_dict(pre_parsed_data) 
            
            file_name_jpg: str = ".".join(file_name.split(".")[:-1]) + ".jpg" # Change extension of the file name to jpg
            
            parsed_data_redeable: dict = parse_file_redeable(file_name_jpg, pre_parsed_data)
            parsed_data_metadata: dict = parse_file_metadata(file_name_jpg, pre_parsed_data)

            file_path_redeable = os.path.join(save_path_redeable, file_name)   
            with open(file_path_redeable, "w") as out_redeable:
                json.dump(parsed_data_redeable, out_redeable, indent=4, ensure_ascii=False)
                
            out_metadata.write(json.dumps(parsed_data_metadata) + "\n")
            
    with open(os.path.join(save_path_metadata_train, "metadata.jsonl"), "w") as out_metadata:
        for file_name, pre_parsed_data in train_pre_parsed_files:
            pre_parsed_data = pre_parsed_data if isinstance(pre_parsed_data, Factura) else Factura.from_dict(pre_parsed_data) 
            
            file_name_jpg: str = ".".join(file_name.split(".")[:-1]) + ".jpg" # Change extension of the file name to jpg
            
            parsed_data_redeable: dict = parse_file_redeable(file_name_jpg, pre_parsed_data)
            parsed_data_metadata: dict = parse_file_metadata(file_name_jpg, pre_parsed_data)
        
            file_path_redeable = os.path.join(save_path_redeable, file_name)   
            with open(file_path_redeable, "w") as out_redeable:
                json.dump(parsed_data_redeable, out_redeable, indent=4, ensure_ascii=False)
                
            out_metadata.write(json.dumps(parsed_data_metadata) + "\n")
    
    
# ===========================================
#                Parsers
# ===========================================

def parse_file_metadata(file_name: str, pre_parsed_data: Factura) -> Dict[str, Any]:
    """
    Parses data for a given file by structuring the pre-parsed data into a dictionary 
    with 'file_name' and 'ground_truth' (in JSON format) so Donut model can reade it.

    Args:
        file_name (str): The name of the file being processed.
        pre_parsed_data (Factura): The pre-parsed data that will be included as 'gt_parse' 
                                          in the ground truth.

    Returns:
        Dict[str, Any]: A dictionary containing the 'file_name' and 'ground_truth', where 
                        'ground_truth' is a JSON string representing the pre-parsed data.
    """
    ground_truth: Dict[str, Any] = {"gt_parse": pre_parsed_data.to_dict()}
    
    parsed_data: Dict[str, Any] = {
        "file_name": file_name,
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False)
    }
    return parsed_data


def parse_file_redeable(file_name: str, pre_parsed_data: Factura) -> Dict[str, Any]:
    """
    Parses a file and returns a readable version of the parsed data, which includes 
    both 'original_data' and the 'ground_truth' parsed for the Donut Model.

    Args:
        file_name (str): The name of the file being processed.
        pre_parsed_data (Factura): The pre-parsed data to be included in the 'original_data' field.

    Returns:
        Dict[str, Any]: A dictionary containing both the 'ground_truth' (as returned by 
                        `parse_file_metadata`) and 'original_data'.
    """
    parsed_data: Dict[str, Any] = parse_file_metadata(file_name, pre_parsed_data)
    parsed_data["original_data"] = pre_parsed_data.to_dict()
    return parsed_data


# ===========================================
#                Main
# ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", type=float)
    parser.add_argument("--dataset_path", type=str, default="datasets_finetune/outputs/FATURA/redeable")
    parser.add_argument("--save_path", type=str, default="datasets_finetune/outputs/FATURA")
    parser.add_argument("--seed", type=int, default=42)
    args, left_argv = parser.parse_known_args()

    load_ensure_parse_save_files(args)
