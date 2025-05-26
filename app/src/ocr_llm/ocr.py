import io
import json
from ocr_llm_module.ocr.azure.document_intelligence import AzureDocumentIntelligenceClient

def document_to_orc(ocr_client: AzureDocumentIntelligenceClient, file_io: io.BytesIO, prebuilt_model: str = None):
    """
    Processes a document using an OCR client and extracts text and structured fields.

    Args:
        ocr_client (AzureDocumentIntelligenceClient): The client used to perform OCR using Azure's Document Intelligence API.
        file_io (io.BytesIO): A file-like object containing the document to be processed (e.g., a PDF or image).
        prebuilt_model (str, optional): If specified, uses this prebuilt model for OCR (e.g., "prebuilt-invoice", "prebuilt-layout").

    Returns:
        tuple:
            - raw_lines (str): All the lines of text from the document, separated by page breaks.
            - document_content (str): Full STRUCTURED text content of the document, also separated by page breaks.
            - pages (list[str]): List of page-wise STRUCTURED strings representing the text on each page.
            - fields_content (dict): Dictionary of extracted fields and their corresponding values.
            - json_output (dict): Raw JSON-like dictionary output from the OCR result containing structured data.
    """

    # IMPORTANT TO CHANGE THE TYPE OF DOCUMENT THAT THE OCR HAS TO READ.
    
    if prebuilt_model is None:
        pages, result = ocr_client.read_document(file_io)
    else:
        pages, result = ocr_client.read_document(file_io, prebuilt_model=prebuilt_model)

    # check_orc_output(result)

    page_break = "\n\n ------- PAGE BREAK ------- \n\n"
    # print(result)
    raw_lines = result.content
    document_content: str = page_break.join(pages)
    # fields_content, json_output = get_fields_from_result(result)
    # print(f"{fields_content = }")
    

    return raw_lines, document_content, pages, None, None #fields_content, json_output 



def get_fields_from_result(result) -> tuple[dict, str]:
    """Convers the Result object return from a azure document intelligence OCR into a dict and a json format with just the field
    Example of results fields
    (
        'Buyer', 
        {
            'type': 'string',
            'valueString': 'Andre Quinn', 
            'content': 'Andre Quinn', 
            'boundingRegions': [
                {
                    'pageNumber': 1,
                    'polygon': [
                        2.718,
                        3.2488,
                        3.6761,
                        3.2532,
                        3.6754,
                        3.4154,
                        2.7173,
                        3.411
                    ]
                }
            ],
            'confidence': 0.906,
            'spans': [
                {
                    'offset': 128,
                    'length': 11
                }
            ]
        }
    )
    Args:
        result (_type_): Result object from a azure document intelligence OCR

    Returns:
        tuple[dict, str]: Structured output fields in dict and in json format
    """
    fields_content = []
    if not result.documents:
        return fields_content, dict()
    
    documents_fields = [document.fields.items() for document in result.documents]
    for document_fields in documents_fields:
        fields_dict = dict()
        for fields in document_fields:
            # 
            field_name = f"value{fields[1]['type'].capitalize()}" # make the first letter of the type capillar so it ends like 'valueString' for example
            if field_name in  fields[1]:
                fields_dict[fields[0]] = fields[1][field_name]
            elif "content" in fields[1]:
                print(f" - Formating key for {fields[0]} didn't work, trying content of:{fields}")
                fields_dict[fields[0]] = fields[1]["content"]
            else:
                print(f" - Field {fields[0]} not found at: {fields[1]}")
                fields_dict[fields[0]] = None
            
            # print(f" - Fields: {fields}")
        fields_content.append(fields_dict)
        
    json_output = json.dumps(fields_content, indent=4)
    
    return fields_content, json_output


def check_orc_output(result):
    # poller = document_intelligence_client.begin_analyze_document(
    #     model_id, AnalyzeDocumentRequest(url_source=formUrl)
    # )
    if result.documents:
        for idx, document in enumerate(result.documents):
            print(f"--------Analyzing document #{idx + 1}--------")
            print(f"Document has type {document.doc_type}")
            print(f"Document has confidence {document.confidence}")
            print(f"Document was analyzed by model with ID {result.model_id}")
            for name, field in document.fields.items():
                print(f"......found field of type '{field.type}' with value '{field.content}' and with confidence {field.confidence}")

    if result.pages:
        for page in result.pages:
            print(f"\nLines found on page {page.page_number}")
            for line in page.lines:
                print(f" - Line '{line.content.encode('utf-8')}'")
            for word in page.words:
                print(f" - Word '{word.content.encode('utf-8')}' has a confidence of {word.confidence}")
            if page.selection_marks:
                for selection_mark in page.selection_marks:
                    print(f" - Selection mark is '{selection_mark.state}' and has a confidence of {selection_mark.confidence}")

    if result.tables:
        for i, table in enumerate(result.tables):
            print(f"\nTable {i + 1} can be found on page:")
            for region in table.bounding_regions:
                print(f"...{region.page_number}")
            for cell in table.cells:
                print(f"...Cell[{cell.row_index}][{cell.column_index}] has content '{cell.content.encode('utf-8')}'")

    print("-----------------------------------")