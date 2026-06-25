import spacy
from Trail.FSC.src.Chunking.chunk_paragraph import content_aware_chunk

nlp = spacy.load("en_core_web_sm")

def extract_propositions(text):
    paragraphs = content_aware_chunk(text)
    propositions = []

    for para in paragraphs:
        # Skip lists
        if para.strip().startswith(("-", "1.")):
            continue

        doc = nlp(para)
        for sent in doc.sents:
            propositions.append(sent.text.strip())

    return propositions
