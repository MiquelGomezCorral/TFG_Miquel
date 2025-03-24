import io
import time
from typing import List

import img2pdf  # type: ignore
from fastapi import APIRouter, File, UploadFile

from src.api.dto.response.bill import CnmcBillDataResponseDTO
from src.controller.octopus.document_processor import DocumentProcessor
from src.schemas.octopus.bill import CnmcBillData


router = APIRouter()

document_processor = DocumentProcessor()


@router.post("/process_bill", name="process_bill")
async def process_bill(*, file: UploadFile = File(...)) -> CnmcBillDataResponseDTO:
    """
    Process a bill and return the extracted data as a response
    """
    file_io = io.BytesIO(file.file.read())

    current_time_seconds: float = time.time()

    bill: CnmcBillData | None = None

    bill, _ = document_processor.process_document(file_io)

    bill_response = CnmcBillDataResponseDTO(**bill.dict())
    bill_response.processing_time_in_seconds = round(time.time() - current_time_seconds, 2)

    return bill_response


@router.post("/process_image_bill", name="process_image_bill")
async def process_image_bill(*, files: List[UploadFile] = File(...)) -> CnmcBillDataResponseDTO:
    """
    Given a list of images, process them as a bill and return the extracted data as a response
    """
    # Concatenate all images into a single PDF file
    pdf_file = io.BytesIO()
    a4inpt = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))  # type: ignore
    pdf_file.write(
        img2pdf.convert(  # type: ignore
            [file.file.read() for file in files],
            layout_fun=img2pdf.get_layout_fun(  # type: ignore
                pagesize=a4inpt,
            ),
        )  # type: ignore
    )
    pdf_file.seek(0)  # Reset the file pointer to ensure the file is read from the beginning

    return await process_bill(file=UploadFile(filename="bill.pdf", file=pdf_file))
