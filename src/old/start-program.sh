python main.py sort-images --image-dir ../images
python main.py generate-info --image-dir ../images --output-dir ../ocr_extract_info
python main.py split-chapters --info-json ../ocr_extract_info/info.json --output-dir ../ocr_extract_info
python main.py process-llm --output-dir ../ocr_extract_info --info-json ../ocr_extract_info/info.json

# 連続実行
python main.py all --image-dir ../images --output-dir ../ocr_extract_info

