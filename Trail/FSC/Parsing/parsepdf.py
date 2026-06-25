# import pdfplumber

# def parse_pdf(path_to_pdf):
#     text = ""

#     with pdfplumber.open(path_to_pdf) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"

#     return text.strip()

import pdfplumber
import re

def remove_special_chars(text):  #garbage tokens (boxes)
    return re.sub(r'[^\x00-\x7F]+', '', text)

def fix_spaced_letters(text):
    # Join single spaced letters: C H O → CHO
    return re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', text)

def fix_repeated_chars(text):
    # Replace repeated characters (e.g. EEEEE → E)
    return re.sub(r'(.)\1{2,}', r'\1', text)

def is_valid_text(line):
    words = line.split()
    
    # Too short
    if len(words) < 3:
        return False

    # Too many symbols
    symbol_ratio = sum(1 for c in line if not c.isalnum() and c != ' ') / len(line)
    
    if symbol_ratio > 0.3:
        return False

    return True

def remove_diagram_noise(text):
    # Remove lines with too many numbers/symbols
    if re.search(r'[\=\+\-\*/]{2,}', text):
        return ""

    # Remove weird chemical spacing like C H O (optional)
    if re.match(r'^(\w\s){3,}', text):
        return ""

    return text

def clean_text(text):
    text = fix_repeated_chars(text)
    text = fix_spaced_letters(text)
    text = remove_special_chars(text)

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Apply diagram filtering
    if not is_valid_text(text):
        return ""

    text = remove_diagram_noise(text)

    return text


def remove_duplicates_preserve_order(text_list):
    seen = set()
    result = []
    for item in text_list:
        if item not in seen and item != "":
            seen.add(item)
            result.append(item)
    return result


def extract_columns(page):
    """
    Split page into left & right columns
    """
    width = page.width

    # Define column bounding boxes
    left_bbox = (0, 0, width / 2, page.height)
    right_bbox = (width / 2, 0, width, page.height)

    left_text = page.within_bbox(left_bbox).extract_text() or ""
    right_text = page.within_bbox(right_bbox).extract_text() or ""

    return left_text, right_text


def parse_pdf(file_path):
    pages_data = []
    full_text_parts = []
    global_seen = set()

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):

            left_text, right_text = extract_columns(page)

            # Process columns sequentially
            combined = (left_text + "\n" + right_text).split("\n")

            cleaned_lines = []

            for line in combined:
                cleaned = clean_text(line)

                if cleaned and cleaned not in global_seen:
                    cleaned_lines.append(cleaned)
                    global_seen.add(cleaned)

            cleaned_lines = remove_duplicates_preserve_order(cleaned_lines)

            page_text = " ".join(cleaned_lines)

            page_content = {
                "page_number": i + 1,
                "text": page_text
            }

            pages_data.append(page_content)

            if page_text:
                full_text_parts.append(f"[Page {i+1}]\n{page_text}")

    combined_text = "\n\n".join(full_text_parts)

    return {
        "doc_id": file_path,
        "text": combined_text,
        "pages": pages_data,
        "metadata": {
            "total_pages": len(pages_data),
            "source": "pdf"
        }
    }


# -------- RUN --------
if __name__ == "__main__":
    file_path = "D://topicTrace//Trail//FSC//Samples//pdfs//NCERT_Class12_chem.pdf"

    result = parse_pdf(file_path)

    print("Pages:", result["metadata"]["total_pages"])
    print("\nPreview:\n")
    print(result["text"][:500])

    with open("parsed_pdf_output.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])