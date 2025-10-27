import argparse
import os
import time
from typing import Optional, Tuple

import pyautogui as pag

SAVE_DIR = "screenshots"
DEFAULT_TOTAL_PAGES = 20  # 撮るページ数の既定値
INTERVAL_SEC = 1.8        # ページ送りの待ち（Kindle描画待ちを含む）
START_DELAY = 5           # 実行後、Kindle に切り替える猶予
PAGE_TURN_CHOICES = ("right", "left")
DEFAULT_PAGE_TURN_KEY = "left"


def main(page_turn_key: Optional[str] = None, total_pages: Optional[int] = None) -> None:
    key = (page_turn_key or DEFAULT_PAGE_TURN_KEY).strip().lower()
    if key not in PAGE_TURN_CHOICES:
        raise ValueError(f"Unsupported page turn key '{page_turn_key}'. Use one of {PAGE_TURN_CHOICES}.")

    pages = total_pages if total_pages is not None else DEFAULT_TOTAL_PAGES
    if pages <= 0:
        raise ValueError("total_pages must be a positive integer.")

    os.makedirs(SAVE_DIR, exist_ok=True)
    pag.FAILSAFE = True      # 画面左上にマウスを移動で即中断可
    pag.PAUSE = 0.05

    print(f"開始 {START_DELAY}s 後に撮影します。Kindle 画面を前面にしてください。")
    print(f"ページ送りキー: {key}")
    print(f"撮影枚数: {pages}")
    time.sleep(START_DELAY)

    for i in range(1, pages + 1):
        path = os.path.join(SAVE_DIR, f"page_{i:04d}.png")
        img = pag.screenshot()            # 画面全体
        img.save(path)
        print(f"[{i}/{pages}] saved -> {path}")

        pag.press(key)
        time.sleep(INTERVAL_SEC)

    print("完了")


def _resolve_runtime_config() -> Tuple[str, Optional[int]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--page-turn-key",
        choices=PAGE_TURN_CHOICES,
        help="Kindle のページ送りに使うキー（デフォルト: right）",
    )
    parser.add_argument(
        "--total-pages",
        type=int,
        help="スクリーンショットを撮るページ数（デフォルト: 20）",
    )
    args = parser.parse_args()

    env_key = os.environ.get("PAGE_TURN_KEY", "")
    env_key = env_key.strip().lower() if env_key else ""

    env_pages = os.environ.get("TOTAL_PAGES")
    total_pages_env: Optional[int] = None
    if env_pages:
        try:
            total_pages_env = int(env_pages)
        except ValueError:  # ユーザーが誤った値を入れた場合はここで止める
            raise ValueError("Environment variable TOTAL_PAGES must be an integer.")

    key = args.page_turn_key or (env_key if env_key in PAGE_TURN_CHOICES else DEFAULT_PAGE_TURN_KEY)
    total_pages = args.total_pages if args.total_pages is not None else total_pages_env

    return key, total_pages


if __name__ == "__main__":
    key, total = _resolve_runtime_config()
    main(key, total)
