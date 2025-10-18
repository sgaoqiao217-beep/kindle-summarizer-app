#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}"/.. && pwd)"
PYTHON_EXECUTABLE="${PYTHON:-python}"
MAIN_SCRIPT="${SCRIPT_DIR}/main.py"

IMAGE_DIR_DEFAULT="${PROJECT_ROOT}/images"
OUTPUT_DIR_DEFAULT="${PROJECT_ROOT}/ocr_extract_info"
INFO_JSON_DEFAULT="${OUTPUT_DIR_DEFAULT}/info.json"

usage() {
  cat <<USAGE
Usage: $(basename "$0") <command> [args]

Commands:
  sort-images            Run image normalisation and sorting
  generate-info          Run OCR to create info.json
  split-chapters         Split info.json into chapter JSON files
  process-llm            Format and summarise chapter JSON files
  all                    Execute the full pipeline

Environment overrides:
  IMAGE_DIR   Path to images directory (default: ${IMAGE_DIR_DEFAULT})
  OUTPUT_DIR  Path to output directory (default: ${OUTPUT_DIR_DEFAULT})
  INFO_JSON   Path to info.json (default: ${INFO_JSON_DEFAULT})

Extra args after the command are forwarded to python.
USAGE
}

command=${1:-}
if [[ -z "${command}" || "${command}" == "-h" || "${command}" == "--help" ]]; then
  usage
  exit 0
fi
shift

IMAGE_DIR_PATH="${IMAGE_DIR:-${IMAGE_DIR_DEFAULT}}"
OUTPUT_DIR_PATH="${OUTPUT_DIR:-${OUTPUT_DIR_DEFAULT}}"
INFO_JSON_PATH="${INFO_JSON:-${INFO_JSON_DEFAULT}}"

case "${command}" in
  sort-images)
    exec "${PYTHON_EXECUTABLE}" "${MAIN_SCRIPT}" sort-images --image-dir "${IMAGE_DIR_PATH}" "$@"
    ;;
  generate-info)
    exec "${PYTHON_EXECUTABLE}" "${MAIN_SCRIPT}" generate-info --image-dir "${IMAGE_DIR_PATH}" --output-dir "${OUTPUT_DIR_PATH}" "$@"
    ;;
  split-chapters)
    exec "${PYTHON_EXECUTABLE}" "${MAIN_SCRIPT}" split-chapters --info-json "${INFO_JSON_PATH}" --output-dir "${OUTPUT_DIR_PATH}" "$@"
    ;;
  process-llm)
    exec "${PYTHON_EXECUTABLE}" "${MAIN_SCRIPT}" process-llm --output-dir "${OUTPUT_DIR_PATH}" --info-json "${INFO_JSON_PATH}" "$@"
    ;;
  all)
    exec "${PYTHON_EXECUTABLE}" "${MAIN_SCRIPT}" all --image-dir "${IMAGE_DIR_PATH}" --output-dir "${OUTPUT_DIR_PATH}" "$@"
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage
    exit 1
    ;;
 esac
