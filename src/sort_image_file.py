import os
import re
from natsort import natsorted

def normalize_filenames(directory: str):
    """
    スクリーンショット画像のファイル名を揃えて並び替える
    無印 → " 0" にリネーム
    """

    files = os.listdir(directory)
    renamed_files = []

    for f in files:
        # スクリーンショットファイルだけ対象にする
        if not f.startswith("スクリーンショット"):
            continue

        base, ext = os.path.splitext(f)

        # 正規表現で "日時 + (オプションの連番)" を検出
        m = re.match(r"(スクリーンショット \d{4}-\d{2}-\d{2} \d{2}\.\d{2}\.\d{2})(?: (\d+))?", base)
        if m:
            datetime_part = m.group(1)
            number_part = m.group(2)

            if number_part is None:  # 無印 → 0
                new_name = f"{datetime_part} 0{ext}"
            else:
                new_name = f"{datetime_part} {number_part}{ext}"

            # リネーム実行
            old_path = os.path.join(directory, f)
            new_path = os.path.join(directory, new_name)

            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {f} → {new_name}")

            renamed_files.append(new_name)

    # 自然順でソート
    sorted_files = natsorted(renamed_files)
    return sorted_files



