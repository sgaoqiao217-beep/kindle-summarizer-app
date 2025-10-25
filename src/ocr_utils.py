import os
import re
import glob
import json
import unicodedata
from collections import defaultdict
from google.cloud import vision
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv
from google.oauth2 import service_account
import streamlit as st


def _vision_client_from_secrets():
    raw = st.secrets["GOOGLE_CREDENTIALS"]
    info = json.loads(raw) if isinstance(raw, str) else raw
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    project_id = info.get("project_id")
    if project_id:
        creds = creds.with_quota_project(project_id)
    return vision.ImageAnnotatorClient(credentials=creds)

"""画像から Title, Body, Left, Right を抽出して dict で返す"""
def extract_info_type1(image_path: str):

    client = _vision_client_from_secrets()
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    annotations = response.text_annotations

    if not annotations:
        return None

    # 文字ごとのバウンディングボックス
    boxes = []
    for ann in annotations[1:]:
        verts = ann.bounding_poly.vertices
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        boxes.append({
            "text": ann.description,
            "left": min(xs), "right": max(xs),
            "top": min(ys), "bottom": max(ys)
        })

    if not boxes:
        return None

    # Title（最上段）
    min_top = min(b["top"] for b in boxes)
    title_group = [b for b in boxes if abs(b["top"] - min_top) < 20]
    title_bottom = max(b["bottom"] for b in title_group)

    # Page number（最下段）
    max_bottom = max(b["bottom"] for b in boxes)
    page_group = [b for b in boxes if abs(b["bottom"] - max_bottom) < 20]
    page_top = min(b["top"] for b in page_group)

    img_width = max(b["right"] for b in boxes)
    left_page = [b for b in page_group if b["right"] < img_width/3]
    right_page = [b for b in page_group if b["left"] > img_width*2/3]

    # Body
    body_group = [b for b in boxes if b["top"] > title_bottom and b["bottom"] < page_top]
    body_sorted = sorted(body_group, key=lambda x: -x["left"])
    
    columns = defaultdict(list)
    for b in body_sorted:
        col_key = int(b["left"] / 50)
        columns[col_key].append(b)

    column_texts = []
    for col in sorted(columns.keys(), reverse=True):
        col_boxes = sorted(columns[col], key=lambda b: b["top"])
        col_text = "".join(b["text"] for b in col_boxes)
        column_texts.append(col_text)

    body_text = "\n".join(column_texts)

    info = {
        "Filename": os.path.basename(image_path),
        "Title": "".join(b["text"] for b in sorted(title_group, key=lambda x: x["left"])),
        "Body": body_text,
        "Left": "".join(b["text"] for b in sorted(left_page, key=lambda x: x["left"])),
        "Right": "".join(b["text"] for b in sorted(right_page, key=lambda x: x["left"])),
    }

    return info

def save_merged_chapter(title, chapter, text_lines, filename_list, output_dir):
    """章データを JSON ファイルに保存（複数ファイル合併版）"""
    chapter_data = {
        "FilenameList": filename_list,       # 合併した元ファイル名リスト
        "Title": title,
        "Chapter": chapter,
        "Body": "\n".join(text_lines)
    }
    
    safe_name = re.sub(r"[\\/:*?\"<>|]", "_", chapter)
    file_path = os.path.join(output_dir, f"{safe_name}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chapter_data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {file_path}")

def split_by_chapter_and_save_list(info_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 書籍によって変更する章タイトル（行頭からマッチさせる）
    # chapter_pattern = re.compile(
    #     r"^(?P<heading>"
    #     r"第(?:[1-9][0-9]*|[０-９]+|[一二三四五六七八九十百千〇零]+)章(?:[ \t\u3000：:\-・][^\n]*)?"
    #     r"|(?:プロローグ|序章|終章|エピローグ|結論)(?:[ \t\u3000：:\-・][^\n]*)?"
    #     r")$",
    #     re.MULTILINE,
    # )

    # 本書（画像の目次）向け: 行頭から
    # 1) アラビア数字(半角/全角)で始まる章タイトル
    #    例: "1　中国のスプートニク的瞬間" / "２ 中国の…"
    #    数字の後に任意の区切り（空白・全角空白・句読点・コロン等）→ タイトル
    # 2) 特別見出し（はじめに/序文/序章/プロローグ/終章/エピローグ/結論/あとがき/謝辞/原註/注/解説 など）
    # 3) 万一に備えて「第○章」表記もフォールバックで許容
    chapter_pattern = re.compile(
        r"^(?P<heading>"
        r"(?:"  # ---- (A) 数字で始まる章 ----
            r"(?:[0-9]{1,3}|[０-９]{1,3})"                     # 1～3桁の半角/全角数字
            r"[ \t\u3000]*[\.．、:：\-・]?"                    # 区切り（任意）
            r"[ \t\u3000]*"                                   # 空白（任意）
            r"[^\n]+?"                                        # 章のタイトル本体
        r")"
        r"|(?:"  # ---- (B) 特別見出し ----
            r"(?:はじめに|序章|序文|序言|まえがき|プロローグ"
            r"|終章|エピローグ|結論|あとがき|解説|謝辞|原註|原注|注)"
            r"(?:[ \t\u3000：:・\-][^\n]*)?"                  # サブタイトル等が続く場合も許容
        r")"
        r"|(?:"  # ---- (C) フォールバック: 第○章 ----
            r"第(?:[0-9０-９]+|[一二三四五六七八九十百千〇零]+)章"
            r"(?:[ \t\u3000：:・\-][^\n]*)?"
        r")"
        r")$",
        re.MULTILINE,
    )


    current_chapter = None
    current_text = []
    current_files = []
    current_title = None

    for info in info_list:
        body = info["Body"].strip()
        if not body:
            continue

        matches = list(chapter_pattern.finditer(body))

        if not matches:
            # ➡️ 同一章の続き
            if current_chapter:
                current_text.append(body)
                if info["Filename"] not in current_files:
                    current_files.append(info["Filename"])
                if not current_title:
                    current_title = info["Title"]
            continue

        last_index = 0
        for idx, match in enumerate(matches):
            heading_text = match.group("heading").strip()
            start_index = match.start()

            if current_chapter and current_text:
                trailing = body[last_index:start_index].strip()
                if trailing:
                    current_text.append(trailing)
                    if info["Filename"] not in current_files:
                        current_files.append(info["Filename"])

                save_merged_chapter(
                    current_title,
                    current_chapter,
                    current_text,
                    current_files,
                    output_dir,
                )

            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
            chapter_chunk = body[match.start():next_start].strip()

            current_chapter = heading_text
            current_text = [chapter_chunk] if chapter_chunk else []
            current_files = [info["Filename"]]
            current_title = info["Title"] or current_title
            last_index = next_start

    # 最後の章を保存
    if current_chapter and current_text:
        save_merged_chapter(
            current_title,
            current_chapter,
            current_text,
            current_files,
            output_dir
        )
