import os
from dotenv import load_dotenv

load_dotenv()

try:  # Prefer the new google-genai SDK; fall back to google-generativeai if available.
    from google import genai  # type: ignore[attr-defined]
    _USE_CLIENT_API = True
except ImportError:
    try:
        import google.generativeai as genai  # type: ignore[import-not-found]
        _USE_CLIENT_API = False
    except ImportError as exc:
        raise ImportError(
            "Missing Google Gemini SDK. Install `google-genai` or `google-generativeai`."
        ) from exc

format_prompt = """
You are given raw text from a Japanese book (a novel or story).
Your task is ONLY to format the text into a clean book-like structure.

Rules:
1. Insert one line break between paragraphs.
2. At the beginning of each paragraph, insert one full-width space character (全角スペース1文字分).
3. Remove unnecessary line breaks inside sentences.
4. If there are unnatural word breaks or typos caused by OCR, fix them using context.
5. Do NOT summarize or shorten the content. Keep all original content.

最終出力は本文の整形後テキストのみを出力してください。
余計な説明や注釈は一切不要です。
"""

summary_prompt_template = """
You are given a chapter from a Japanese novel or story (already formatted).
Your task is to summarize it into a polished, story-like text.

Use the following chapter title exactly as provided:
{chapter_title}

**Output structure (must follow exactly):**

1. First line: the chapter title as an H2 heading.
   * If Google Docs: apply **HEADING_2** style to the title line.
2. One blank line.
3. The summary body text (Japanese), 3000–4000 characters, formatted per the rules below.

**Critical constraints (do not violate):**

* Do **not** stop after writing the heading. Writing only the heading is invalid.
* Always include the full summary body after the blank line.
* If the input lacks a clear chapter title, infer a concise title from the content (e.g., main topic or scene) and still output it as H2.

**Rules for the summary body:**

1. Length: **3000–4000 Japanese characters** (strict).
2. Include all key events, characters, and emotional flow.
3. Ensure the writing is natural, grammatically correct, and coherent.
4. Preserve the original style, tone, and atmosphere.
5. Japanese book-style formatting:

   * Insert **one line break between paragraphs**.
   * At the beginning of each paragraph, insert **one full-width space character**（全角スペース1文字分）.
   * Remove unnecessary line breaks inside sentences.

**Output requirements:**

* 最終出力は必ず日本語で書き、整理された完成版のみを出力してください。
* 字数は必ず3000字以上4000字以内にしてください。
* 出力は本文の内容のみとし、余計な説明や注釈、注意書き、メタコメントは一切付けないでください。
"""

def call_model():

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "'GEMINI_API_KEY' setting is missing in environment variables."
        )

    model_name = os.getenv("GEMINI_MODEL")
    if not model_name:
        raise ValueError("'GEMINI_MODEL' setting is missing in environment variables.")

    if _USE_CLIENT_API:
        client = genai.Client(api_key=api_key)
        return client, model_name

    genai.configure(api_key=api_key)
    generative_model = genai.GenerativeModel(model_name=model_name)
    return generative_model, model_name

def format_text(raw_text: str) -> str:
    runner, model_name = call_model()

    full_prompt = f"{format_prompt}\n\nHere is the raw text:\n{raw_text}"
    if _USE_CLIENT_API:
        response = runner.models.generate_content(model=model_name, contents=full_prompt)
    else:
        response = runner.generate_content(full_prompt)
    return response.text

def summarize_text(formatted_text: str, chapter_title: str) -> str:
    runner, model_name = call_model()

    summary_prompt = summary_prompt_template.format(chapter_title=chapter_title)
    full_prompt = f"{summary_prompt}\n\nHere is the formatted text:\n{formatted_text}"
    if _USE_CLIENT_API:
        response = runner.models.generate_content(model=model_name, contents=full_prompt)
    else:
        response = runner.generate_content(full_prompt)
    response_text = response.text.strip()

    heading_line = f"## {chapter_title}".strip()
    lines = response_text.splitlines()
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("##"):
            lines[0] = heading_line
        else:
            lines.insert(0, heading_line)
    else:
        lines = [heading_line]

    body_lines = [line.rstrip() for line in lines[1:]]

    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    normalized_body = "\n".join(body_lines)
    if normalized_body:
        final_text = f"{heading_line}\n\n{normalized_body}"
    else:
        final_text = heading_line

    return final_text
