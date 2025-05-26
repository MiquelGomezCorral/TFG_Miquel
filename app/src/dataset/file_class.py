from typing import Any, Dict, List

# ===========================================
#             Factura class
# ===========================================

class Factura:
    def __init__(self):
        self.buyer: str = ""
        self.address: str = ""
        self.date: str = ""
        # self.shopping_or_tax: bool = None  # True for Commercial, False for Tax
        self.currency: str = ""
        self.subtotal: float = 0.0
        self.discount: float = 0.0
        self.tax: float = 0.0
        self.total: float = 0.0
        # self.products: List[Product] = []

    def __repr__(self):
        return f"Factura(buyer={self.buyer}, address={self.address}, date={self.date}, currency={self.currency}, subtotal={self.subtotal}, discount={self.discount}, tax={self.tax}, total={self.total})"#, products={self.products})" shopping_or_tax={self.shopping_or_tax},

    def to_dict(self):
        """
        Convert the Factura object to a dictionary, including converting products to dictionaries.
        """
        return {
            "buyer": self.buyer,
            "address": self.address,
            "date": self.date,
            # "shopping_or_tax": self.shopping_or_tax,
            "currency": self.currency,
            "subtotal": self.subtotal,
            "discount": self.discount,
            "tax": self.tax,
            "total": self.total,
            # "products": [Product.convert_to_dict(product) for product in self.products]  # Convert products to dictionaries
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
        # factura.shopping_or_tax = data.get("shopping_or_tax", None)
        factura.currency = data.get("currency", "")
        factura.subtotal = data.get("subtotal", 0.0)
        factura.discount = data.get("discount", 0.0)
        factura.tax = data.get("tax", 0.0)
        factura.total = data.get("total", 0.0)
        # factura.products = data.get("products", [])
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
        