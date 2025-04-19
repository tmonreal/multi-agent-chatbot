import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def read_doc(file_path):
    file_loader = PyPDFLoader(file_path)
    document = file_loader.load()
    print(f"Loaded document with {len(document)} pages")
    return document

def chunk_data_with_llm_sections_groq(docs, llm, chunk_size=1000, chunk_overlap=100):
    """
    Uses the Groq LLM to classify each paragraph of Jorge's CV into semantic sections.
    Returns a list of Documents, each labeled by section.
    """
    text = "\n".join(doc.page_content for doc in docs)

    # Split into paragraph-like blocks
    blocks = re.split(r"\n\s*\n", text)

    labeled_blocks = []
    for i, block in enumerate(blocks):
        cleaned_block = block.strip()
        if not cleaned_block or len(cleaned_block.split()) < 5:
            continue  # skip very short or empty lines

        prompt = f"""
        Eres un clasificador de secciones para currículums (CVs) escritos en español.
        A partir del siguiente bloque de texto, clasifica el contenido en UNA de las siguientes categorías:

        - datos personales
        - formación académica
        - experiencia laboral
        - certificaciones
        - habilidades
        - publicaciones
        - proyectos
        - desconocido

        Responde solamente con la categoría.

        Bloque:
        \"\"\"
        {cleaned_block}
        \"\"\"
        """

        result = llm.invoke(prompt)
        label = getattr(result, "content", str(result)).strip().lower()

        # fallback to unknown if not a known label
        valid_labels = ["personal info", "education", "work experience", "certifications", "skills", "publications", "projects"]
        if label not in valid_labels:
            label = "unknown"

        labeled_blocks.append((label, cleaned_block))

    # Group blocks by label
    grouped_sections = {}
    for label, block in labeled_blocks:
        grouped_sections.setdefault(label, []).append(block)

    # Chunk within each label group
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_docs = []

    for label, blocks in grouped_sections.items():
        full_text = "\n\n".join(blocks)
        for chunk in splitter.split_text(full_text):
            content = f"[{label.upper()}]\n{chunk.strip()}"
            final_docs.append(Document(page_content=content, metadata={"section": label}))

    return final_docs

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    """
    Custom chunking for a CV using numbered section headers (e.g., 1, 2, ...)
    """
    text = "\n".join([doc.page_content for doc in docs])

    # Match lines like "12.1 ACTIVIDADES ACTUALES"
    pattern = r"(?=^\d{1,2}(?:\.\d+)?\s+[A-ZÁÉÍÓÚÑ\s]+$)"
    matches = list(re.finditer(pattern, text, re.MULTILINE))

    sections = []

    # ✅ Add preamble (before first section)
    if matches and matches[0].start() > 0:
        preamble_text = text[:matches[0].start()].strip()
        if preamble_text:
            sections.append(("INTRODUCCIÓN", preamble_text))

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            title_line = section_text.splitlines()[0].strip()
            sections.append((title_line, section_text))

    # Chunk inside each section
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = []

    for title, section_text in sections:
        splits = splitter.split_text(section_text)
        for chunk in splits:
            chunked_docs.append(Document(page_content=f"{title}\n{chunk}"))

    return chunked_docs

def chunk_data_sectionwise(docs, chunk_size=500, chunk_overlap=100):
    text = "\n".join([doc.page_content for doc in docs])

    section_titles = [
        "Contact",
        "Education",
        "Experience",
        "Courses and Certifications",
        "Certifications",
        "Languages",
        "Projects",
        "Top Skills",
        "Skills",
        "Awards"
    ]

    pattern = r"(?=^(" + "|".join(re.escape(title) for title in section_titles) + r")\s*$)"
    matches = list(re.finditer(pattern, text, re.MULTILINE))

    section_docs = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            section_name = matches[i].group(1).strip()
            section_docs.append((section_name, section_text))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    chunked_docs = []
    for section_name, section_text in section_docs:
        sub_chunks = splitter.split_text(section_text)
        for chunk_text in sub_chunks:
            lines = chunk_text.strip().splitlines()
            first_line = lines[0].strip() if lines else ""
            if section_name.lower() in first_line.lower():
                full_text = chunk_text.strip()
            else:
                full_text = f"{section_name}\n{chunk_text.strip()}"
            chunked_docs.append(Document(page_content=full_text, metadata={"section": section_name}))

    return chunked_docs