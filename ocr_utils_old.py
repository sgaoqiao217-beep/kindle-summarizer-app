import os
import re
import glob
import json
from collections import defaultdict
from google.cloud import vision
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv

# def detect_text(path, client):
#     """指定した画像をOCRしてテキストを返す"""
#     with open(path, "rb") as f:
#         content = f.read()
#     image = vision.Image(content=content)

#     response = client.document_text_detection(image=image)
#     if response.full_text_annotation:
        
#         # texts = response.full_text_annotation.text

#         # for text in texts:
#         #         print(f'\nText: {text.description}')
#         #         vertices = ['({},{})'.format(v.x, v.y) for v in text.bounding_poly.vertices]
#         #         print('bounds: {}'.format(','.join(vertices)))

#         return response.full_text_annotation.text
#     return ""


"""画像から Title, Body, Left, Right を抽出して dict で返す"""
def extract_info_type1(image_path: str):

    client = vision.ImageAnnotatorClient()
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

def split_by_chapter_and_save(info: dict, output_dir: str):
    """info の Body を章ごとに分割し、章ごとに JSON ファイル保存"""
    os.makedirs(output_dir, exist_ok=True)

    chapter_pattern = re.compile(r"(第[0-9一二三四五六七八九十]+章.*|序章.*|結論.*)")
    body_text = info["Body"]

    current_chapter = None
    current_text = []

    for line in body_text.splitlines():
        m = chapter_pattern.match(line.strip())
        if m:
            # 直前の章を保存
            if current_chapter and current_text:
                save_single_chapter(info, current_chapter, current_text, output_dir)
            # 新しい章開始
            current_chapter = m.group(1).strip()
            current_text = []
        else:
            current_text.append(line)

    # 最後の章を保存
    if current_chapter and current_text:
        save_single_chapter(info, current_chapter, current_text, output_dir)

def save_single_chapter(info, chapter, text_lines, output_dir):
    """章データを JSON ファイルに保存"""
    chapter_data = {
        "Filename": info["Filename"],
        "Title": info["Title"],
        "Chapter": chapter,
        "Body": "\n".join(text_lines),
        "Left": info["Left"],
        "Right": info["Right"],
    }

    # ファイル名を安全にする（スペースや記号を置換）
    safe_name = re.sub(r"[\\/:*?\"<>|]", "_", chapter)
    file_path = os.path.join(output_dir, f"{safe_name}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chapter_data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {file_path}")
