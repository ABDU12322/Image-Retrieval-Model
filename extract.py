import pypdf
import docx

with open("extracted_texts.txt", "w", encoding="utf-8") as f:
    try:
        pdf = pypdf.PdfReader('Research_MultiModal_Approach.pdf')
        pdf_text = '\n'.join([page.extract_text() for page in pdf.pages])
        f.write('--- PDF ---\n')
        f.write(pdf_text[:4000])
        f.write('\n\n')
    except Exception as e:
        f.write(f'PDF Error: {e}\n')

    try:
        doc = docx.Document('Report ML project.docx')
        doc_text = '\n'.join([p.text for p in doc.paragraphs])
        f.write('--- DOCX ---\n')
        f.write(doc_text[:4000])
    except Exception as e:
        f.write(f'DOCX Error: {e}\n')
