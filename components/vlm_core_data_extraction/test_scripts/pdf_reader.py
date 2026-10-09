import pymupdf

def read_pdf_text(filePath):
    doc = pymupdf.open(filePath)

    for page in doc:
        return page.get_text()


def read_pdf_drawings(filePath):
    doc = pymupdf.open(filePath)
    page = doc[0]

    drawings = page.get_drawings()
    return drawings

filePath = "assets/original_cvs/Joppe_Iemants_CV.pdf"
pdf_text = read_pdf_text(filePath)
pdf_drawings = read_pdf_drawings(filePath)
print(pdf_drawings)