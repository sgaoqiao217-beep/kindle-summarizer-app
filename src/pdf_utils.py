import io
from typing import List, Tuple, Optional
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT

def register_jp_font(font_name: str, font_path: str):
    """日本語フォントを登録（最初に1回だけ呼ぶ）"""
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except Exception as e:
        raise RuntimeError(f"フォント登録に失敗: {e}")

def build_pdf_bytes(
    title: str,
    chapters: List[Tuple[str, str]],  # [(見出しH2, 本文)]
    font_name: str = "NotoSansJP",
    margin_mm: int = 15,
) -> bytes:
    """
    章ごとにH2見出し＋本文でPDFを生成してbytesを返す。
    - chapters: [(heading, body_text)]
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin_mm * mm,
        rightMargin=margin_mm * mm,
        topMargin=margin_mm * mm,
        bottomMargin=margin_mm * mm,
        title=title,
        author="Kindle Summarizer",
    )

    styles = getSampleStyleSheet()
    # ベースを日本語フォントで上書き
    base = ParagraphStyle(
        "Base",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=18,            # 行間（日本語は広めが見やすい）
        spaceAfter=6,
        alignment=TA_JUSTIFY,  # 両端揃え（お好みで）
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base,
        fontSize=16,
        leading=22,
        spaceBefore=8,
        spaceAfter=10,
        alignment=TA_LEFT,
    )
    body = ParagraphStyle(
        "Body",
        parent=base,
        fontSize=11,
        leading=18,
    )

    story = []
    # タイトルページ簡易版（不要なら削除）
    if title:
        story.append(Paragraph(title, ParagraphStyle("Title", parent=base, fontSize=20, leading=26)))
        story.append(Spacer(1, 8 * mm))

    for idx, (heading, text) in enumerate(chapters, start=1):
        if heading:
            story.append(Paragraph(heading, h2))
            story.append(Spacer(1, 2 * mm))
        # 要件どおりの段落整形（先頭全角スペース等）をPDF Paragraphに流し込む場合、
        # 内部でHTML風タグが解釈されるので、「<」「&」は適宜エスケープするのが安全。
        def esc(s: str) -> str:
            return (s.replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;"))
        # 既に「段落ごとに1行空け＋段頭全角スペース」を整形済みなら、そのまま段落化
        paragraphs = [p for p in text.split("\n") if p.strip() != ""]
        for p in paragraphs:
            story.append(Paragraph(esc(p), body))
        # 章の後ろに余白
        story.append(Spacer(1, 6 * mm))
        # 次章は次ページ開始にしたい場合は PageBreak を挿入
        if idx < len(chapters):
            story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()

