#### テキスト整形 & 章分割

import re
#from ocr_utils import extract_book_title_with_size


def normalize_linebreaks(text: str) -> str:
    """句点・感嘆符・疑問符の後の改行は残し、それ以外は削除"""
    # 「。」「！」「？」＋改行を一時タグに置換
    text = re.sub(r"([。！？])\n", r"\1<KEEP_NEWLINE>", text)
    # 残りの改行は削除
    text = text.replace("\n", "")
    # 一時タグを改行に戻す
    text = text.replace("<KEEP_NEWLINE>", "\n")
    return text


def clean_text(text: str, book_title: str) -> str:
    """不要部分削除"""
    # 書籍タイトルの繰り返しを削除
    text = re.sub(r"NETFLIXの最強人事戦略.*?(~自由と責任の文化を築く~)?", "", text)
    #text = re.sub(rf"{re.escape(book_title)}.*", "", text)
    #pattern = rf"{re.escape(book_title)}[ 　]*(〜.*?〜|~.*?~)?"
    #pattern = rf"{re.escape(book_title)}[\s　]*(〜[\s\S]*?〜|~[\s\S]*?~)?"
    #text = re.sub(pattern, "", text)
    # 読了時間や％表示を削除
    text = re.sub(r"本を読み終えるまで:.*", "", text)
    text = re.sub(r"\d+%", "", text)

    # 改行処理
    text = normalize_linebreaks(text)

    # 複数スペースを1つに
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def split_by_chapter(text: str):
    """序章・第〇章・プロローグ・エピローグごとに分割"""
    parts = []
    current_title = None
    current_text = ""

    chapter_pattern = re.compile(r"(プロローグ|序章|第[0-9一二三四五六七八九十]+章|エピローグ|あとがき)")

    for line in text.splitlines():
        m = chapter_pattern.search(line)
        if m:
            if current_title and current_text.strip():
                parts.append((current_title, current_text.strip()))
            current_title = m.group(1)
            current_text = line
        else:
            current_text += "\n" + line

    if current_title and current_text.strip():
        parts.append((current_title, current_text.strip()))

    return parts
