    
import io
import json
from ocr_llm_module.ocr.azure.document_intelligence import AzureDocumentIntelligenceClient

def document_to_orc(ocr_client: AzureDocumentIntelligenceClient, file_io: io.BytesIO, prebuilt_model: str = None):
    # IMPORTANT TO CHANGE THE TYPE OF DOCUMENT THAT THE OCR HAS TO READ.
    
    if prebuilt_model is None:
        pages, result = ocr_client.read_document(file_io)
    else:
        pages, result = ocr_client.read_document(file_io, prebuilt_model=prebuilt_model)

    # check_orc_outpu(result)
    fields_content, json_output = get_fields_from_result(result)
    # print(f"{fields_content = }")
    
    document_content: str = "\n\n ------- PAGE BREAK ------- \n\n".join(pages)

    return document_content, pages, fields_content, json_output 



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
    
    for idx, document in enumerate(result.documents):
        print("--------Analyzing document #{}--------".format(idx + 1))
        print("Document has type {}".format(document.doc_type))
        print("Document has confidence {}".format(document.confidence))
        print("Document was analyzed by model with ID {}".format(result.model_id))
        for name, field in document.fields.items():
            print("......found field of type '{}' with value '{}' and with confidence {}".format(field.type, field.content, field.confidence))


    # iterate over tables, lines, and selection marks on each page
    for page in result.pages:
        print("\nLines found on page {}".format(page.page_number))
        for line in page.lines:
            print("...Line '{}'".format(line.content.encode('utf-8')))
        for word in page.words:
            print(
                "...Word '{}' has a confidence of {}".format(
                    word.content.encode('utf-8'), word.confidence
                )
            )
        if page.selection_marks:
            for selection_mark in page.selection_marks:
                print(
                    "...Selection mark is '{}' and has a confidence of {}".format(
                        selection_mark.state, selection_mark.confidence
                    )
                )

    for i, table in enumerate(result.tables):
        print("\nTable {} can be found on page:".format(i + 1))
        for region in table.bounding_regions:
            print("...{}".format(i + 1, region.page_number))
        for cell in table.cells:
            print(
                "...Cell[{}][{}] has content '{}'".format(
                    cell.row_index, cell.column_index, cell.content.encode('utf-8')
                )
            )
    print("-----------------------------------")