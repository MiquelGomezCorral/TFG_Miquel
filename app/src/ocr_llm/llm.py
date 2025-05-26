from pydantic import BaseModel, Field
from ocr_llm_module.llm.azure.azure_openai import AzureOpenAILanguageModel

# Define the structure of the response from the LLM
class LLMStructuredResponse(BaseModel):
    """
    Structure for the response from the LLM
    """
    buyer: str | None = Field(
        title="Buyer",
        description="The name of the buyer on the invoice. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    address: str | None = Field(
        title="Address",
        description="The address of the buyer (Strictly use the same commas and format as seen in the document). IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    date: str | None = Field(
        title="Date",
        description="The date of the invoice in YYYY-MM-DD format. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    currency: str | None = Field(
        title="Currency",
        description="The currency used in the invoice. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    subtotal: float | None = Field(
        title="Subtotal",
        description="The subtotal amount before discounts and tax. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    discount: float | None = Field(
        title="Discount",
        description="The discount applied to the invoice. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    tax: float | None = Field(
        title="Tax",
        description="The total tax applied to the invoice. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )
    total: float | None = Field(
        title="Total",
        description="The final total amount after tax and discounts. IF IT DOES NOT APPEAR, 'None' should be assigned to this values."
    )




def document_to_llm(llm_client: AzureOpenAILanguageModel, document_content: str):
    prompt: str = f"""
    Given the following invoice:
    <document>
    {document_content}
    </document>

    Extract the following details:
    - Buyer name
    - Buyer address
    - Invoice date (format: YYYY-MM-DD)
    - Transaction type (Shopping or Tax-related; return True for Shopping, False for Tax)
    - Currency used
    - Subtotal amount
    - Discount applied (Total ammount, not percentage)
    - Tax amount (Total ammount, not percentage)
    - Total amount

    Please provide ONLY the requested data in a structured format.
    """

    # Get the response from the LLM in a structured format
    llm_output: LLMStructuredResponse = (
        llm_client.get_llm_response_with_structured_response(
            prompt=prompt,
            schema=LLMStructuredResponse
        )
    )
    return llm_output

