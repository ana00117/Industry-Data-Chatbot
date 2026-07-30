from pypdf import PdfReader


def extract_pages_from_pdf(file_bytes, file_name):
    reader = PdfReader(file_bytes)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "source": file_name,
                "page": page_number
            })
    return pages


def extract_pages_from_uploaded_files(uploaded_files):
    all_pages = []
    for uploaded_file in uploaded_files:
        pages = extract_pages_from_pdf(uploaded_file, uploaded_file.name)
        all_pages.extend(pages)
    return all_pages
