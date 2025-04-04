# Azure Document Intelligence Module

This module provides a comprehensive interface for Azure Document Intelligence services, enabling the extraction and reconstruction of text content from various document types while preserving document structure and detecting elements like tables.

## Overview

The `AzureDocumentIntelligenceClient` class connects to Azure Document Intelligence services and provides methods for:

- Transforming PDF documents into structured plain text
- Extracting text while preserving layout information
- Reconstructing document layout and structure
- Detecting and formatting tables within documents
- Processing specialized document types (invoices, receipts, etc.)
- Here is an [example](../../../examples/ex2_ocr/) of how to use the module.


## Key Features

### 1. Document Reading Capabilities
- Process general documents with layout preservation
- Extract text from specialized document types (invoices, receipts, ID documents)
- Configure reading parameters like character spacing and line detection
- Maintain document structure including paragraphs, tables, and spacing

### 2. Table Detection and Formatting
- Automatically detect tabular content in documents
- Format tables using grid-style representation
- Preserve table structure separate from regular text content

### 3. Document Type-Specific Processing

The module implements specialized methods for different document types:

```python
# Process a general document
pages = client.read_document(file_stream)

# Process specialized document types
invoice_text = client.read_invoice(invoice_file)
receipt_text = client.read_receipt(receipt_file)
id_text = client.read_credit_card(card_image)
layout_text = client.read_layout(document_file)
```

### 4. Text Positioning and Layout Reconstruction

- Sophisticated positional analysis for text elements
- Line detection and alignment calculation
- Proper spacing and indentation preservation
- Overlap detection to prevent duplicate text extraction

## Internal Processing Algorithm

1. **Element Detection**: Identifies bounding boxes for words and tables
2. **Table Extraction**: Processes tables separately with proper formatting
3. **Text Ordering**: Uses line slope detection to determine which words belong on the same line
4. **Element Sorting**: Orders elements by page number, y-position, and x-position
5. **Layout Reconstruction**: Adds appropriate spacing and line breaks to maintain document structure

## Environment Setup

The module requires the following environment variables or constructor arguments:

| Environment Variable | Constructor Parameter | Description |
|---------------------|----------------------|-------------|
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | `azure_endpoint` | Azure Document Intelligence service endpoint |
| `AZURE_DOCUMENT_INTELLIGENCE_API_KEY` | `azure_api_key` | API key for authentication |
| - | `max_overlap_between_elements` | How much closer two elements can be before they are considered they are the same element (default: 0.05) |

Example initialization:

```python
client = AzureDocumentIntelligenceClient(
	azure_endpoint="https://your-endpoint.cognitiveservices.azure.com/",
	azure_api_key="your_api_key_here",
	max_overlap_between_elements=0.05
)
```


### How does it work?

1. The libraries contained in requirements.txt must be installed in a docker. This way compatibility problems are avoided, ensuring that you have the same development environment.

2. To execute the script, indicate in the variable "example_file_path" the path of the .pdf file to transform. In "example_output_file_path" indicate the path of the .txt file where the result will be saved.

3. Finally, to run tests, simply call the script:
	> python3 path/to/file/DocumentIntelligence.py

### Internal functioning.

1. The program begins by detecting the different bounding boxes for each of the words.

2. From these, detected tables are stored with their content and position.

3. Next, the ordering of the different words is carried out. This is a key point when preserving the document structure. The most important thing here is to detect which words belong to the same line. For this, we use the slope of the line formed by the different cells, so that if a line passes through the cell we are analyzing, we will replace its minimum "y" (the upper position of the cell) with the "y" assigned to the line. If this does not happen, its minimum "y" will be replaced by its midpoint.

4. Next, ordering is performed according to the page number, the "y" position of the upper left corner and the "x" position of the upper left corner, for each of the detected elements. The processing done previously allows us to have proper word ordering at this point.

5. Finally, indentation and line spacing are performed. To correctly perform these two, we need to store both the horizontal and vertical position of the previously processed word. We also use two constants, the number of characters per line (px_por_linea) and the number of lines per page (linea_por_pag), for which using 150 and 60 respectively has given the best results. With these parameters, along with the width and height of the page, we can determine how many pixels a line occupies and how many pixels it has.

	First, the line spacing to be added must be considered. With the previous vertical position and the upper "y" of the current word, we will obtain the number of "\n" to add. This case is only taken into account if the previous word and the current one are not considered to be on the same line.

	To perform correct indentation, the number of pixels per line is obtained. This way, with the previous horizontal position and the left "x" of the current word, we will get the number of spaces to assign.

### Important considerations.

1. Position of tables: despite respecting the structure of the tables, it does not take into account the original position of the tables. That is, even if the table is on the right, it will not position it there, it will only print it respecting the order of appearance.

2. Document inclination: it has a certain tolerance for documents with inclined lines, but does not ensure a correct solution in these cases.

3. Text with different sizes: finding very different letter sizes on the same line can also lead to incorrect positions.

4. Handwritten documents: handwriting can result in letters that stretch to upper or lower lines, which could lead to errors in the results.

5. Vertical text: Sometimes, it's possible to find vertical text such as those found in the legends of some graphs, or clarifications in the margins of the pages. This type of text can significantly break the structure of the document.
