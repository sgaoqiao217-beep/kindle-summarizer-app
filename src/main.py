import argparse
import json
import os
import re
import unicodedata
from typing import List, Optional, Tuple

from tqdm import tqdm

from google_api_utils_old import get_credentials, create_google_doc
from llm_utils import format_text, summarize_text
from ocr_utils import extract_info_type1, split_by_chapter_and_save_list
from sort_image_file import normalize_filenames
from text_utils import clean_text


BASE_DIR = os.path.dirname(__file__)
DEFAULT_IMAGE_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "images"))
DEFAULT_OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "ocr_extract_info"))
DEFAULT_INFO_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "info.json")
HEADING_KEYS = ("Title", "Chapter", "Subtitle")

def fix_broken_chapter_tokens(text: str) -> str:
    """OCRで改行や空白でバラけた『第…章』『序章/終章/プロローグ/エピローグ』を修復する"""
    if not isinstance(text, str):
        return text

    t = text

    # OCRされたBodyの冒頭だけを整形する。本文途中の改行には触れない。

    # 1) 「第」「数字」「章」が分断されているケースを冒頭でのみ修復する
    chapter_match = re.match(
        r'^(?P<prefix>\s*)第(?P<num_block>(?:\s|[0-9０-９一二三四五六七八九十百千〇零])+?)章',
        t,
        flags=re.DOTALL
    )
    if chapter_match:
        prefix = chapter_match.group("prefix")
        num_block = chapter_match.group("num_block")
        num = re.sub(r'\s+', '', num_block)
        rest = t[chapter_match.end():]
        t = f"{prefix}第{num}章{rest}"

    # 2) 「序\n章」「終\n章」「プロ\nローグ」「エピ\nローグ」を冒頭でのみ結合
    for pattern, replacement in (
        (r'^(?P<prefix>\s*)序\s+章', '序章'),
        (r'^(?P<prefix>\s*)終\s+章', '終章'),
        (r'^(?P<prefix>\s*)プロ\s+ローグ', 'プロローグ'),
        (r'^(?P<prefix>\s*)エピ\s+ローグ', 'エピローグ'),
    ):
        special_match = re.match(pattern, t, flags=re.DOTALL)
        if special_match:
            prefix = special_match.group("prefix")
            rest = t[special_match.end():]
            t = f"{prefix}{replacement}{rest}"
            break

    # 3) 章見出し（＋任意のサブタイトル）を冒頭1行に収め、直後を改行にする
    heading_match = re.match(
        r'^(?P<prefix>\s*)(?P<heading>('
        r'第[0-9０-９一二三四五六七八九十百千〇零]+章'
        r'|序章|終章|プロローグ|エピローグ'
        r')(?:[ \t\u3000：:・\-][^\n]*)?)',
        t
    )
    if heading_match:
        prefix = heading_match.group("prefix")
        heading = re.sub(r'[ \t\u3000]+', ' ', heading_match.group("heading")).strip()
        rest = t[heading_match.end():]
        rest = rest.lstrip(' \t\u3000')
        if rest and rest[0] != '\n':
            rest = '\n' + rest
        if not rest:
            rest = '\n'
        t = f"{prefix}{heading}{rest}"

    # 4) 章タイトルの判定対象を Body 先頭行に限定するため、2行目以降の行頭へスペースを付与
    first_newline = t.find('\n')
    if first_newline != -1 and first_newline + 1 < len(t):
        head = t[:first_newline + 1]
        tail = t[first_newline + 1:]
        tail = re.sub(r'\n(?=\S)', '\n ', tail)
        if tail and not tail[0].isspace():
            tail = ' ' + tail
        t = head + tail

    return t

def normalize_heading_block(text: str) -> str:
    """章タイトルなどで改行・全角/半角が混在している場合に整形する"""
    if not isinstance(text, str):
        return text
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"(?<![。．！？.!?：:])\n(?!\n)", "", normalized)
    normalized = re.sub(r"[ \t\u3000]+", " ", normalized).strip()
    return normalized

# 既存: _normalize_info_headings の直下あたりに追加
def _normalize_info_all(info: dict) -> dict:
    """Title/Chapter/Subtitle の正規化に加えて Body の章トークン割れも修復"""
    if not isinstance(info, dict):
        return info
    # 見出し系
    info = _normalize_info_headings(info)
    # Body の「第\n7章」「序\n章」等を修復
    if "Body" in info and info["Body"]:
        info["Body"] = fix_broken_chapter_tokens(info["Body"])
    return info



def _collect_image_files(image_dir: str) -> List[str]:
    supported_exts = {".jpg", ".jpeg", ".png"}
    try:
        files = [name for name in os.listdir(image_dir) if os.path.splitext(name)[1].lower() in supported_exts]
    except FileNotFoundError:
        return []
    return [os.path.join(image_dir, name) for name in sorted(files)]


def _normalize_info_headings(info: dict) -> dict:
    for key in HEADING_KEYS:
        if key in info and info[key]:
            info[key] = normalize_heading_block(info[key])
    return info


def sort_images(image_dir: str) -> List[str]:
    normalize_filenames(image_dir)
    image_paths = _collect_image_files(image_dir)

    print(f"並び替え対象: {len(image_paths)} 枚")
    for path in image_paths[:5]:
        print("  ->", path)
    if len(image_paths) > 5:
        print("  ...")
    print("並び替え完了")
    return image_paths


def generate_info_json(image_dir: str, output_dir: str, image_paths: Optional[List[str]] = None) -> Tuple[str, List[dict]]:
    os.makedirs(output_dir, exist_ok=True)

    if image_paths is None:
        image_paths = sort_images(image_dir)

    if not image_paths:
        print("OCR対象の画像が見つかりませんでした。")
        return os.path.join(output_dir, "info.json"), []

    print("OCR化開始")
    info_records: List[dict] = []
    for img_path in tqdm(image_paths, desc="OCR", unit="page"):
        info = extract_info_type1(img_path)
        if info:
            info_records.append(_normalize_info_all(info))

    output_path = os.path.join(output_dir, "info.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info_records, f, ensure_ascii=False, indent=2)

    print(f"info.json を保存しました: {output_path}")
    return output_path, info_records


def split_chapters(info_json_path: str, output_dir: str, info_list: Optional[List[dict]] = None) -> List[str]:
    if info_list is None:
        if not os.path.exists(info_json_path):
            raise FileNotFoundError(f"info.json が見つかりません: {info_json_path}")
        with open(info_json_path, "r", encoding="utf-8") as f:
            info_list = json.load(f)

    if not isinstance(info_list, list):
        raise ValueError("info.json の形式が不正です (list ではありません)")

    info_list = [_normalize_info_all(rec) for rec in info_list]
    split_by_chapter_and_save_list(info_list, output_dir)

    chapter_files = [
        os.path.join(output_dir, name)
        for name in sorted(os.listdir(output_dir))
        if name.endswith(".json") and name != "info.json"
    ]
    print(f"章ごとのJSONを {len(chapter_files)} 件出力しました。")
    return chapter_files


def process_chapters_with_llm(output_dir: str, info_json_path: str = DEFAULT_INFO_JSON) -> None:
    if not os.path.exists(info_json_path):
        raise FileNotFoundError(f"info.json が見つかりません: {info_json_path}")

    with open(info_json_path, "r", encoding="utf-8") as f:
        info_list = json.load(f)

    book_title = "書籍タイトル不明"
    if info_list:
        title_candidate = info_list[0].get("Title")
        if title_candidate:
            book_title = normalize_heading_block(title_candidate)
    print(f"抽出された書籍タイトル: {book_title}")

    chapter_entries = []
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(".json"):
            continue
        if name == "info.json" or name.startswith("info"):
            continue

        chapter_path = os.path.join(output_dir, name)
        try:
            with open(chapter_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"章データ読込エラー ({name}): {exc}")
            continue

        if not isinstance(data, dict):
            print(f"章データ形式不正のためスキップ: {name}")
            continue

        chapter_entries.append((chapter_path, data))

    if not chapter_entries:
        print("章ごとのJSONが見つかりませんでした。先に分割処理を実行してください。")
        return

    chapters = []
    total_chapter_files = len(chapter_entries)
    for idx, (chapter_path, data) in enumerate(chapter_entries, 1):
        print(
            f"[{idx}/{total_chapter_files}] 章データ読込: {os.path.basename(chapter_path)}",
            flush=True,
        )
        raw_chapter_title = data.get("Chapter", "無題の章")
        chapter_title = normalize_heading_block(raw_chapter_title)
        body = data.get("Body", "")
        body = fix_broken_chapter_tokens(body)
        chapter_text = clean_text(body, book_title)
        chapters.append((chapter_title, chapter_text))

    creds = get_credentials()
    formatted_folder_name = f"{book_title}_整形"
    summary_folder_name = f"{book_title}_要約"

    for chapter_title, chapter_text in chapters:
        print(f"LLM処理中: {chapter_title}")
        try:
            formatted = format_text(chapter_text)
            create_google_doc(formatted_folder_name, chapter_title, formatted, creds)
            print(f"整形完了: {chapter_title}")

            summary = summarize_text(formatted, chapter_title)
            create_google_doc(summary_folder_name, chapter_title, summary, creds)
            print(f"要約完了: {chapter_title}")
        except Exception as exc:
            print(f"要約処理エラー ({chapter_title}): {exc}")

    print("LLMによる処理が完了しました。")


def run_all(image_dir: str, output_dir: str) -> None:
    image_paths = sort_images(image_dir)
    info_json_path, info_list = generate_info_json(image_dir, output_dir, image_paths=image_paths)
    split_chapters(info_json_path, output_dir, info_list=info_list)
    process_chapters_with_llm(output_dir, info_json_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCRTool pipeline controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_sort = subparsers.add_parser("sort-images", help="画像ファイル名の整形と並び替え")
    parser_sort.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)

    parser_info = subparsers.add_parser("generate-info", help="OCRを実行しinfo.jsonを出力")
    parser_info.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser_info.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    parser_split = subparsers.add_parser("split-chapters", help="info.jsonから章ごとに分割")
    parser_split.add_argument("--info-json", default=DEFAULT_INFO_JSON)
    parser_split.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    parser_llm = subparsers.add_parser("process-llm", help="章ごとのJSONに整形・要約を適用")
    parser_llm.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser_llm.add_argument("--info-json", default=DEFAULT_INFO_JSON)

    parser_all = subparsers.add_parser("all", help="全ての処理を順番に実行")
    parser_all.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser_all.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sort-images":
        sort_images(args.image_dir)
    elif args.command == "generate-info":
        generate_info_json(args.image_dir, args.output_dir)
    elif args.command == "split-chapters":
        split_chapters(args.info_json, args.output_dir)
    elif args.command == "process-llm":
        process_chapters_with_llm(args.output_dir, args.info_json)
    elif args.command == "all":
        run_all(args.image_dir, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
