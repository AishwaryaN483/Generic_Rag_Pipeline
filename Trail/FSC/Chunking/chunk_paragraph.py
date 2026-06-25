import re

def content_aware_chunk(text, max_words=300):
    sections = re.split(
        r'\n(?=\d+(?:\.\d+)*\s+[A-Z])',
        text
    )

    chunks = []
    buffer = ""

    for sec in sections:
        if not sec or not sec.strip():
            continue

        words = sec.split()

        if len(words) > max_words:
            sentences = re.split(r'(?<=[.!?])\s+', sec)
            for s in sentences:
                if len((buffer + " " + s).split()) > max_words:
                    chunks.append(buffer.strip())
                    buffer = s
                else:
                    buffer += " " + s
        else:
            if len((buffer + " " + sec).split()) > max_words:
                chunks.append(buffer.strip())
                buffer = sec
            else:
                buffer += " " + sec

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if len(c.split()) > 30]
