import sys
import fitz  # PyMuPDF
import json
import nltk
import os
import io
import contextlib
import re

# --- NLTK Data Path Setup ---
# Try to use a path within the venv first, then user's home.
nltk_data_candidates = []
if 'VIRTUAL_ENV' in os.environ:
    nltk_data_candidates.append(os.path.join(os.environ['VIRTUAL_ENV'], 'nltk_data'))
nltk_data_candidates.append(os.path.join(os.path.expanduser("~"), "nltk_data"))

configured_nltk_download_dir = None
for path_candidate in nltk_data_candidates:
    try:
        if not os.path.exists(path_candidate):
            os.makedirs(path_candidate, exist_ok=True)  # exist_ok=True for safety
            print(f"Created NLTK data directory: {path_candidate}", file=sys.stderr)

        # Ensure this path is in nltk.data.path and prioritized
        if path_candidate not in nltk.data.path:
            nltk.data.path.insert(0, path_candidate)  # Prepend to give it priority

        configured_nltk_download_dir = path_candidate  # Select the first successfully created/accessed path
        print(f"NLTK data path configured to prioritize download and lookup at: {configured_nltk_download_dir}", file=sys.stderr)
        break
    except OSError as e:
        print(f"Warning: Could not create or access NLTK data directory {path_candidate}: {e}", file=sys.stderr)

if not configured_nltk_download_dir:
    print(f"Warning: Could not establish a preferred NLTK data directory. Using NLTK defaults for downloads.", file=sys.stderr)
print(f"Current NLTK data search paths: {nltk.data.path}", file=sys.stderr)
# --- End NLTK Data Path Setup ---

def download_nltk_package(package_id, friendly_name, download_dir_override=None):
    """
    Downloads an NLTK package (e.g., 'punkt').
    Ensures all download messages go to stderr.
    Returns True if download command is successful, False otherwise.
    """
    print(f"Attempting to download NLTK package '{friendly_name}' (ID: {package_id})...", file=sys.stderr)
    if download_dir_override:
        print(f"Attempting download to directory: {download_dir_override}", file=sys.stderr)

    capture_stream = io.StringIO()
    download_successful = False
    try:
        # Redirect stdout/stderr during nltk.download to capture any output
        with contextlib.redirect_stdout(capture_stream), contextlib.redirect_stderr(capture_stream):
            # nltk.download() returns True for success, False for failure (if raise_on_error=False)
            if nltk.download(package_id, download_dir=download_dir_override, quiet=True, raise_on_error=False):
                download_successful = True

        if download_successful:
            print(f"NLTK download command for '{friendly_name}' reported success.", file=sys.stderr)
        else:
            print(f"NLTK download command for '{friendly_name}' reported failure.", file=sys.stderr)

        # Log captured output from NLTK to stderr for debugging
        nltk_output = capture_stream.getvalue()
        if nltk_output and nltk_output.strip():
            print(f"Output from NLTK during '{friendly_name}' download attempt:\n{nltk_output}", file=sys.stderr)
            if not download_successful and "Problem with server or network connection" in nltk_output:
                print("Hint: NLTK download failure might be due to network issues or the NLTK server.", file=sys.stderr)

    except Exception as e:
        # This would catch errors if raise_on_error=True, or other unexpected errors
        print(f"Exception during NLTK download command for '{friendly_name}': {e}", file=sys.stderr)
        nltk_output = capture_stream.getvalue()
        if nltk_output and nltk_output.strip():
            print(f"Captured output from NLTK for '{friendly_name}' before exception:\n{nltk_output}", file=sys.stderr)

    return download_successful

def clean_text(text):
    """
    Cleans the text by removing unnecessary whitespace and non-printable characters.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\f', '', text)
    return text.strip()

def extract_text_and_metadata(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        all_chunks = []
        metadata = doc.metadata
        processed_metadata = {}
        for key, value in metadata.items():
            if value is None:
                processed_metadata[key] = ""
            else:
                processed_metadata[key] = str(value)

        print(f"PDF Metadata: {processed_metadata}", file=sys.stderr)
        print(f"Total Pages in PDF: {len(doc)}", file=sys.stderr)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text = page.get_text("text")
            print(f"Page {page_num + 1}: Raw text length: {len(raw_text)}", file=sys.stderr)
            cleaned_text = clean_text(raw_text)
            print(f"Page {page_num + 1}: Cleaned text length: {len(cleaned_text)}", file=sys.stderr)
            text += cleaned_text + "\n"

            if cleaned_text:
                page_chunks = intelligent_chunking(cleaned_text, pdf_path)
                print(f"Page {page_num + 1}: Generated {len(page_chunks[0])} chunks.", file=sys.stderr)
                all_chunks.extend(page_chunks[0])

        doc.close()
        print(f"Total extracted characters: {len(text)}", file=sys.stderr)
        print(f"Total chunks generated: {len(all_chunks)}", file=sys.stderr)
        return text, processed_metadata, all_chunks, None
    except Exception as e:
        return "", {}, [], f"Error extracting text/metadata: {str(e)}"

def intelligent_chunking(text, source_document_name, max_chunk_size=512, overlap=50):
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        return [], f"NLTK sent_tokenize failed: {str(e)}"

    chunks = []
    current_chunk_tokens = []
    current_chunk_text = ""

    for sentence in sentences:
        sentence_tokens = nltk.word_tokenize(sentence)

        if len(current_chunk_tokens) + len(sentence_tokens) <= max_chunk_size:
            current_chunk_tokens.extend(sentence_tokens)
            current_chunk_text += (" " if current_chunk_text else "") + sentence
        else:
            if current_chunk_text:
                chunks.append({"content": current_chunk_text, "source_document": source_document_name})

            if len(sentence_tokens) > max_chunk_size:
                print(f"Warning: Sentence from '{source_document_name}' exceeds max_chunk_size and will be truncated.", file=sys.stderr)
                truncated_sentence_text = " ".join(sentence_tokens[:max_chunk_size])
                chunks.append({"content": truncated_sentence_text, "source_document": source_document_name})
                current_chunk_tokens = []
                current_chunk_text = ""
            else:
                current_chunk_tokens = sentence_tokens
                current_chunk_text = sentence

    if current_chunk_text:
        chunks.append({"content": current_chunk_text, "source_document": source_document_name})

    return chunks, None

def main(pdf_file_path, original_file_name):
    nltk_error_message = None
    try:
        nltk.sent_tokenize("Test sentence for punkt.")
        print("NLTK 'punkt' tokenizer is available and functional.", file=sys.stderr)
    except LookupError as e_initial_lookup:
        print(f"NLTK 'punkt' tokenizer not found or not functional ({e_initial_lookup}). Attempting to download 'punkt' package.", file=sys.stderr)
        if download_nltk_package('punkt', 'punkt tokenizer', download_dir_override=configured_nltk_download_dir):
            print("NLTK 'punkt' package download command reported success. Verifying functionality...", file=sys.stderr)
            try:
                nltk.sent_tokenize("Test sentence after punkt download attempt.")
                print("NLTK 'punkt' tokenizer is now available and functional after download.", file=sys.stderr)
            except LookupError as e_after_download:
                nltk_error_message = f"NLTK 'punkt' package download reported success, but tokenizer still not functional: {str(e_after_download)}. This might indicate an incomplete download, issues with NLTK data paths ({nltk.data.path}), or that the downloaded package is corrupted. Check stderr for NLTK's own download messages."
            except Exception as e_verify_general:
                nltk_error_message = f"Error verifying 'punkt' functionality after download attempt: {str(e_verify_general)}"
        else:
            nltk_error_message = "Failed to download NLTK 'punkt' package. Please check network connectivity, NLTK download server status, and write permissions to NLTK data directories. Captured output from NLTK (if any) was printed to stderr."
    except Exception as e_initial_general:
        nltk_error_message = f"Unexpected error during initial NLTK setup: {str(e_initial_general)}"

    if nltk_error_message:
        result = {"FileName": original_file_name, "Content": "", "Chunks": [], "Metadata": {}, "Error": nltk_error_message}
        print(json.dumps(result), file=sys.stdout)
        sys.exit(0)

    full_text, metadata, chunks_data, error = extract_text_and_metadata(pdf_file_path)
    if error:
        result = {"FileName": original_file_name, "Content": "", "Chunks": [], "Metadata": metadata or {}, "Error": error}
        print(json.dumps(result), file=sys.stdout)
        sys.exit(0)

    output = {
        "FileName": original_file_name,
        "Content": full_text,
        "Chunks": chunks_data,
        "Metadata": metadata,
        "Error": None
    }
    print(json.dumps(output), file=sys.stdout)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        error_output = {
            "FileName": "",
            "Content": "",
            "Chunks": [],
            "Metadata": {},
            "Error": "Usage: python pdf_processor.py <path_to_pdf_file> <original_file_name>"
        }
        print(json.dumps(error_output), file=sys.stdout)
        sys.exit(0)

    pdf_path_arg = sys.argv[1]
    original_file_name_arg = sys.argv[2]
    main(pdf_path_arg, original_file_name_arg)
