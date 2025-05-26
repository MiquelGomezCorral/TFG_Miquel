
from typing import Any, Dict

from file_class import Factura


# ===========================================
#             TEMPLATE MANAGER
# ===========================================    

def extract_json(data: dict, template: int) -> dict:
    """
    Extracts JSON data based on the specified template and returns a pre-parsed dictionary.

    Args:
        data (dict): The raw JSON data to be parsed.
        template (int): The template number that determines how to parse the data.
    
    Returns:
        dict: The parsed data in the desired structure.
    """
    
    pre_parsed_factura: Factura = Factura()
    
    if template == 1:
        return extract_template_1(data, pre_parsed_factura) # Full Full
    elif template == 3:
        return extract_template_3(data, pre_parsed_factura) # -subtotal, discount
    elif template == 5:
        return extract_template_5(data, pre_parsed_factura) # -discount ? Due Balue as Total
    elif template == 6:
        return extract_template_6(data, pre_parsed_factura) # -discount
    elif template == 8:
        return extract_template_8(data, pre_parsed_factura) # Saturated layour -discount, tax 

# ===========================================
#             TEMPLATE PARSERS
# ===========================================   

def extract_template_1(data: Dict[str, Any], parsed_factura: Factura) -> Factura:
    """
    Extracts and parses the information from the raw data for template 1 and fills the parsed_factura structure.

    Args:
        data (Dict[str, Any]): The raw data from the template.
        parsed_factura (Factura): A Factura object to be populated with the structure of the extracted information.
    
    Returns:
        Factura: The updated `parsed_factura` object with parsed values for each field.
    """  
    parsed_factura.buyer = data["BUYER"]["text"].split("\n")[0].split(":")[1] # Bill to:James Miller -> "James Miller"
    parsed_factura.address = " ".join(data["BUYER"]["text"].split("\n")[1:3]) # 41839 Lee Terrace Apt. 982\nLake Gregoryland, WV 71038 US -> One line
    parsed_factura.date = data["DATE"]["text"].split(": ")[1] # Date: 20-Mar-2008
    
    # parsed_factura.shopping_or_tax = 'commercial' in data["TITLE"]["text"].lower()
    
    parsed_factura.subtotal = float(data["SUB_TOTAL"]["text"].split()[-2])
    
    parsed_factura.discount = float(data["DISCOUNT"]["text"].split()[-1])
    parsed_factura.tax = float(data["TAX"]["text"].split()[-2])
    
    parsed_factura.currency = data["TOTAL"]["text"].split()[-1] # €, EUR, $, USD...
    parsed_factura.total = float(data["TOTAL"]["text"].split()[-2])
    
    return parsed_factura
    
def extract_template_3(data: Dict[str, Any], parsed_factura: Factura) -> Factura:
    """
    Extracts and parses the information from the raw data for template 1 and fills the parsed_factura structure.

    Args:
        data (Dict[str, Any]): The raw data from the template.
        parsed_factura (Factura): A Factura object to be populated with the structure of the extracted information.
    
    Returns:
        Factura: The updated `parsed_factura` object with parsed values for each field.
    """  
    parsed_factura.buyer = data["BILL_TO"]["text"].split("\n")[1] # BBILL_TO:\nAmanda Snow-> "JAmanda Snow"
    parsed_factura.address = " ".join(data["BILL_TO"]["text"].split("\n")[2:4]) # 41839 Lee Terrace Apt. 982\nLake Gregoryland, WV 71038 US -> One line
    parsed_factura.date = data["DATE"]["text"].split(": ")[1] # Date: 20-Mar-2008
    
    # parsed_factura.shopping_or_tax = 'commercial' in data["TITLE"]["text"].lower()
    
    parsed_factura.subtotal = None # it doesn't has
    
    parsed_factura.discount = None # it doesn't has
    parsed_factura.tax = float(data["TAX"]["text"].split()[-2])
    
    parsed_factura.currency = data["TOTAL"]["text"].split()[-1] # €, EUR, $, USD...
    parsed_factura.total = float(data["TOTAL"]["text"].split()[-2])
    
    return parsed_factura
    
def extract_template_5(data: Dict[str, Any], parsed_factura: Factura) -> Factura:
    """
    Extracts and parses the information from the raw data for template 1 and fills the parsed_factura structure.

    Args:
        data (Dict[str, Any]): The raw data from the template.
        parsed_factura (Factura): A Factura object to be populated with the structure of the extracted information.
    
    Returns:
        Factura: The updated `parsed_factura` object with parsed values for each field.
    """  
    parsed_factura.buyer = data["BUYER"]["text"].split("\n")[0].split(":")[1] # Bill :James Miller -> "James Miller"
    parsed_factura.address = " ".join(data["BUYER"]["text"].split("\n")[1:3]) # 41839 Lee Terrace Apt. 982\nLake Gregoryland, WV 71038 US -> One line
    parsed_factura.date = data["DATE"]["text"].split(": ")[1] # Date: 20-Mar-2008
    
    # parsed_factura.shopping_or_tax = False # All are Taxes
    
    parsed_factura.subtotal = float(data["SUB_TOTAL"]["text"].split()[-2])
    
    parsed_factura.discount = None # it doesn't has
    parsed_factura.tax = float(data["TAX"]["text"].split()[-2])
    
    parsed_factura.currency = data["TOTAL"]["text"].split()[-1] # €, EUR, $, USD...
    parsed_factura.total = float(data["TOTAL"]["text"].split()[-2])
    
    return parsed_factura
    
def extract_template_6(data: Dict[str, Any], parsed_factura: Factura) -> Factura:
    """
    Extracts and parses the information from the raw data for template 1 and fills the parsed_factura structure.

    Args:
        data (Dict[str, Any]): The raw data from the template.
        parsed_factura (Factura): A Factura object to be populated with the structure of the extracted information.
    
    Returns:
        Factura: The updated `parsed_factura` object with parsed values for each field.
    """  
    parsed_factura.buyer = data["BUYER"]["text"].split("\n")[0].split(":")[1] # Bill to:James Miller -> "James Miller"
    parsed_factura.address = " ".join(data["BUYER"]["text"].split("\n")[1:3]) # 41839 Lee Terrace Apt. 982\nLake Gregoryland, WV 71038 US -> One line
    parsed_factura.date = data["DATE"]["text"].split(": ")[1] # Date: 20-Mar-2008
    
    # parsed_factura.shopping_or_tax = 'commercial' in data["TITLE"]["text"].lower()
    
    parsed_factura.subtotal = float(data["SUB_TOTAL"]["text"].split()[-2])
    
    parsed_factura.discount = None # it doesn't has
    parsed_factura.tax = float(data["TAX"]["text"].split()[-2])
    
    parsed_factura.currency = data["TOTAL"]["text"].split()[-1] # €, EUR, $, USD...
    parsed_factura.total = float(data["TOTAL"]["text"].split()[-2])
    
    return parsed_factura
    
def extract_template_8(data: Dict[str, Any], parsed_factura: Factura) -> Factura:
    """
    Extracts and parses the information from the raw data for template 1 and fills the parsed_factura structure.

    Args:
        data (Dict[str, Any]): The raw data from the template.
        parsed_factura (Factura): A Factura object to be populated with the structure of the extracted information.
    
    Returns:
        Factura: The updated `parsed_factura` object with parsed values for each field.
    """  
    parsed_factura.buyer = data["BUYER"]["text"].split("\n")[0].split(":")[1] # Bill to:James Miller -> "James Miller"
    parsed_factura.address = " ".join(data["BUYER"]["text"].split("\n")[1:3]) # 41839 Lee Terrace Apt. 982\nLake Gregoryland, WV 71038 US -> One line
    parsed_factura.date = data["DATE"]["text"].split(": ")[1] # Date: 20-Mar-2008
    
    # parsed_factura.shopping_or_tax = True # All are Commercial shopping
    
    parsed_factura.subtotal = float(data["SUB_TOTAL"]["text"].split()[-2])
    
    parsed_factura.discount = None # it doesn't has
    parsed_factura.tax = None # it doesn't has
    
    parsed_factura.currency = data["TOTAL"]["text"].split()[-1] # €, EUR, $, USD...
    parsed_factura.total = float(data["TOTAL"]["text"].split()[-2])
    
    return parsed_factura