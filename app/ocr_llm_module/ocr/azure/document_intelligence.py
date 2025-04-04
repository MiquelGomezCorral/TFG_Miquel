import os
import io
from tabulate import tabulate
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient


class AzureDocumentIntelligenceClient():
    """
    AzureDocumentIntelligenceClient provides an interface to interact
    with Azure's Document Intelligence services to extract and reconstruct text
    content from various document types (e.g., invoices, layouts, receipts, and credit cards).

    Attributes:

        azure_client (DocumentIntelligenceClient):
            An instance used to communicate with the Azure Document Intelligence service,
            initialized with the specified endpoint and API key.

        max_overlap_between_elements (float):
            The maximum allowable overlap between text elements. This value is used
            to determine whether a newly detected text element overlaps with any existing
            elements (such as tables) and should be excluded from the main text extraction.

    Methods:
        __init__(azure_endpoint: Optional[str], azure_api_key: Optional[str], max_overlap_between_elements: float = 0.05)
            Initializes the AzureDocumentIntelligenceClient instance.
            The constructor retrieves the Azure endpoint and API key either from the provided arguments or from the
            environment variables "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" and "AZURE_DOCUMENT_INTELLIGENCE_API_KEY".
            Raises:
                ValueError: If either the azure_endpoint or azure_api_key is not provided.
        _is_overlapping_with_element(position: List[int | float], element: List[Dict[str, List[int | float]]]) -> bool
            Determines if a given text element (defined by its position and bounding coordinates) overlaps with any
            existing elements (typically tables).
                position:
                    A list containing the page number and bounding coordinates of the new element.
                element:
                    A list of dictionaries, each representing an existing element with a "position" key that holds
                    its bounding coordinates.
                True if the new element overlaps with any of the existing elements; False otherwise.
        read_document(file: io.BytesIO, prebuilt_model: str = "prebuilt-read", max_characters_per_line: int = 150) -> List[str]
            Processes a document by analyzing its contents using Azure's prebuilt models (text, tables, etc.).
            It reconstructs the page layout based on the detected positions of text and tables.
                file:
                    A BytesIO stream containing the document (typically PDF) to be analyzed.
                prebuilt_model:
                    The name of the prebuilt model to be used (e.g., "prebuilt-read", "prebuilt-invoice", "prebuilt-layout").
                    Defaults to "prebuilt-read".
                max_characters_per_line:
                    The maximum number of characters per line; used for spacing calculations during text reconstruction.
                A list of strings, each representing the reassembled textual content of a page from the document.
        read_invoice(file: io.BytesIO) -> List[str]
            Extracts text from an invoice document using the "prebuilt-invoice" model.
                file:
                    A BytesIO stream of the invoice document.
                A list of strings with the extracted content from the invoice.
        read_layout(file: io.BytesIO) -> List[str]
            Extracts text from a layout document using the "prebuilt-layout" model.
                file:
                    A BytesIO stream of the layout document.
                A list of strings with the extracted content from the layout.
        read_receipt(file: io.BytesIO) -> List[str]
            Extracts text from a receipt document using the "prebuilt-receipt" model.
                file:
                    A BytesIO stream of the receipt document.
                A list of strings with the extracted content from the receipt.
        read_credit_card(file: io.BytesIO) -> List[str]
            Extracts text from a credit card document using the "prebuilt-idDocument" model.
                file:
                    A BytesIO stream of the credit card document.
                A list of strings with the extracted content from the credit card document.
    Usage:
        To use the AzureDocumentIntelligenceClient, initialize it either by directly passing the required
        endpoint and API key or by setting the corresponding environment variables. Once initialized, utilize
        one of the read_* methods to analyze and extract text from the specified document type.
    Note:
        This client depends on proper Azure credentials and endpoint configuration to perform document analysis.
    """

    def __init__(
        self,
        azure_endpoint: str | None = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", None),
        azure_api_key: str | None = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", None),
        max_overlap_between_elements: float = 0.05,
    ):

        self.max_overlap_between_elements = max_overlap_between_elements

        if azure_endpoint is None:
            raise ValueError(
                "Azure Endpoint is required to use Azure Document Intelligence. "
                "Please set the 'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT' environment"
                " variable or pass it as an argument in the constructor."
            )

        if azure_api_key is None:
            raise ValueError(
                "Azure API Key is required to use Azure Document Intelligence. "
                "Please set the 'AZURE_DOCUMENT_INTELLIGENCE_API_KEY' environment"
                " variable or pass it as an argument in the constructor."
            )

        self.azure_client = DocumentIntelligenceClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_api_key),
        )

    def _is_overlapping_with_element(
        self,
        position: list[int | float],
        element: list[dict[str, list[int | float]]]
    ) -> bool:
        """
            Check if new element overlaps with any of the existing elements
            Args:
                - position (list): The position of the new element
                - element (list): The list of existing elements
                - max_overlap (float): The maximum overlap allowed between the new element and the existing element

            Returns:
                - bool: True if the new element overlaps with any of the existing elements, False otherwise
        """
        margin = -self.max_overlap_between_elements
        page_number: int = int(position[0])
        line_top_left_x: float = position[1]
        line_top_left_y: float = position[2]
        line_bottom_right_x: float = position[5]
        line_bottom_right_y: float = position[6]

        for table in element:
            table_page_number: int = int(table["position"][0])
            table_top_left_x: float = table["position"][1]
            table_top_left_y: float = table["position"][2]
            table_bottom_right_x: float = table["position"][5]
            table_bottom_right_y: float = table["position"][6]

            if page_number == table_page_number and not (
                line_bottom_right_x < table_top_left_x - margin
                or line_top_left_x > table_bottom_right_x + margin
                or line_bottom_right_y < table_top_left_y - margin
                or line_top_left_y > table_bottom_right_y + margin
            ):
                return True

        return False

    def read_document(
        self,
        file: io.BytesIO,
        prebuilt_model: str = "prebuilt-read",
        max_characters_per_line: int = 150
    ) -> list[str]:
        """
        Read the text from a document.
        Parameters:
            prebuilt_model (str): The prebuilt model to be used for reading the document.
            (e.g. "prebuilt-invoice, prebuilt-layout, etc.")

            document (io.BytesIO): The document to be read.
        Returns:
            list[str]: The text extracted from the document.
        """
        minimum_aprox_line_per_page: int = 60

        poller = self.azure_client.begin_analyze_document(
            prebuilt_model,
            body=file,
            content_type="application/pdf",
        )

        result = poller.result()

        sections = []
        tables = []

        # read tables first
        for table in result.tables:  # type: ignore
            row_count = table.row_count
            column_count = table.column_count

            # Initialize the table
            matrix = [
                [
                    ""
                    for _ in range(column_count)
                ] for _ in range(row_count)
            ]

            for cell in table.cells:
                i = cell.row_index
                j = cell.column_index
                matrix[i][j] = cell.content

            formatted_table = tabulate(matrix, tablefmt="grid")

            page_number = int(
                table.bounding_regions[0].page_number)  # type: ignore
            bounding_box = table.bounding_regions[0].polygon   # type: ignore
            position = (  # type: ignore
                page_number, bounding_box[0], bounding_box[1], bounding_box[2],
                bounding_box[3], bounding_box[4], bounding_box[5], bounding_box[6], bounding_box[7]
            )
            tables.append(  # type: ignore
                {
                    "type": "table",
                    "content": formatted_table,
                    "position": position
                }
            )

        page_width = []
        page_height = []
        orden_y = [[] for _ in range(len(result.pages))]  # type:ignore

        # read the main content
        for page_idx, page in enumerate(result.pages):
            page_width.append(page.width)  # type: ignore
            page_height.append(page.height)  # type: ignore
            for word in page.words:  # type: ignore
                page_number = page_idx + 1
                bounding_box = word.polygon
                ymin = None  # type:ignore
                x = bounding_box[0]  # type:ignore

                for recta in orden_y[page_idx]:  # type:ignore
                    m = (recta[1][1] - recta[0][1]) / (recta[1][0] - recta[0][0])  # type:ignore
                    y = m * (x - recta[0][0]) + recta[0][1]  # type:ignore

                    if (y >= bounding_box[1] and y <= bounding_box[5]):  # type:ignore
                        ymin = y  # type:ignore
                        break

                if ymin is None:
                    ymin = bounding_box[1] + (bounding_box[5] - bounding_box[1]) / 2  # type:ignore
                    orden_y[page_idx].append(  # type:ignore
                        ((bounding_box[0], ymin), (bounding_box[4], ymin)))  # type:ignore

                position = (  # type: ignore
                    page_number, bounding_box[0], ymin, bounding_box[2],  # type: ignore
                    bounding_box[3], bounding_box[4], bounding_box[5],  # type: ignore
                    bounding_box[6], bounding_box[7]  # type: ignore
                )

                # Check if the current line overlaps with any of the tables

                overlap = self._is_overlapping_with_element(
                    position, tables  # type: ignore
                )

                if not overlap:
                    sections.append(  # type: ignore
                        {
                            "type": "text",
                            "content": word.content,
                            "position": position
                        }
                    )

        sections.extend(tables)  # type: ignore
        sections.sort(key=lambda x: (  # type: ignore
            x["position"][0], x["position"][2], x["position"][1]  # type: ignore
        ))

        pages: list[str] = ["" for _ in range(len(result.pages))]
        page_space, page_alt1, page_alt2 = [
            0
        ] * len(result.pages), [0] * len(result.pages), [0] * len(result.pages)

        for section in sections:  # type: ignore
            page_number = section["position"][0]  # type: ignore
            page_index = page_number - 1  # type: ignore
            content = section["content"]  # type: ignore

            if section["type"] != "table":
                dif1 = (section["position"][6] - section["position"][2]) / 3  # type:ignore
                dif2 = (page_alt2[page_index] - page_alt1[page_index]) / 3  # type:ignore
                if (
                    section["position"][6] - dif1 > page_alt2[page_index]  # type:ignore
                        or section["position"][2] + dif1 < page_alt1[page_index]  # type:ignore
                ) and (
                    page_alt1[page_index] + dif2 < section["position"][2]  # type:ignore
                        or page_alt2[page_index] - dif2 > section["position"][6]  # type:ignore
                ):  # type:ignore
                    height_page = (  # type: ignore
                        page_height[page_index] / minimum_aprox_line_per_page  # type: ignore
                    )
                    height_spaces = section["position"][2] - page_alt2[page_index]  # type: ignore
                    h = int(height_spaces / height_page)  # type: ignore

                    pages[page_index] += "\n"
                    for _ in range(h - 1):
                        pages[page_index] += "\n"
                    page_space[page_index] = 0

                px_letra = page_width[page_index] / max_characters_per_line  # type: ignore
                px_sp = section["position"][1] - page_space[page_index]  # type: ignore
                space = int(px_sp / px_letra)  # type: ignore

                pages[page_index] += " "
                for _ in range(space - 1):  # type: ignore
                    pages[page_index] += " "
                pages[page_index] += content  # type: ignore
            else:
                pages[page_index] += "\n\n" + content + "\n\n"  # type: ignore
            page_space[page_index] = section["position"][5]  # type: ignore
            page_alt1[page_index] = section["position"][2]  # type: ignore
            page_alt2[page_index] = section["position"][6]  # type: ignore

        return pages

    def read_invoice(self, file: io.BytesIO) -> list[str]:
        """
        Read the text from an invoice.
        Parameters:
            file (io.BytesIO): The invoice to be read.
        Returns:
            list[str]: The text extracted from the invoice.
        """
        return self.read_document(file, "prebuilt-invoice")

    def read_layout(self, file: io.BytesIO) -> list[str]:
        """
        Read the text from a layout.
        Parameters:
            file (io.BytesIO): The layout to be read.
        Returns:
            list[str]: The text extracted from the layout.
        """
        return self.read_document(file, "prebuilt-layout")

    def read_receipt(self, file: io.BytesIO) -> list[str]:
        """
        Read the text from a receipt.
        Parameters:
            file (io.BytesIO): The receipt to be read.
        Returns:
            list[str]: The text extracted from the receipt.
        """
        return self.read_document(file, "prebuilt-receipt")

    def read_credit_card(self, file: io.BytesIO) -> list[str]:
        """
        Read the text from a credit card.
        Parameters:
            file (io.BytesIO): The credit card to be read.
        Returns:
            list[str]: The text extracted from the credit card.
        """
        return self.read_document(file, "prebuilt-idDocument")
