import re

def clean_text(text: str) -> str:
    """
    Clean text extracted from a PDF.

    The cleaner:
    - Normalizes line endings
    - Removes leading/trailing whitespace
    - Removes excessive spaces
    - Removes excessive blank lines
    - Preserves meaningful text, numbers, punctuation, and paragraphs
    """
    
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    
    # Remove leading/trailing whitespace
    lines = [line.strip() for line in text.split("\n")]
    
    # Remove consecutive blank lines
    cleaned_lines = []
    previous_blank = False
    
    for line in lines:
        if line == "":
            if not previous_blank:
                cleaned_lines.append(line)
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False
    
    text = "\n".join(cleaned_lines)
    
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[\t]+", " ", text)
    
    # Keep at most one completely blank line between paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()