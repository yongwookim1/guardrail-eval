#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_mir_data.sh [options]

Clone MIR source datasets with Git LFS/Xet and dump them into datasets/mir/.

Options:
  --cache-root PATH      Local cache root for cloned dataset repos.
  --output-root PATH     Output root for datasets/mir/images and datasets/mir/texts.
  --textvqa-split NAME   TextVQA split to pull and dump. Default: validation
  --cnndm-version VER    CNN/DailyMail version dir to pull. Default: 3.0.0
  --cnndm-split NAME     CNN/DailyMail split to pull and dump. Default: validation
  --max-images N         Optional cap on dumped unique images.
  --max-texts N          Optional cap on dumped text files.
  --overwrite            Overwrite existing dumped files.
  --skip-download        Reuse existing local clones and only run the dump step.
  -h, --help             Show this help message.
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing required command: $cmd" >&2
    exit 1
  fi
}

sync_repo() {
  local repo_url="$1"
  local dest_dir="$2"
  local ref_name="$3"

  if [[ ! -d "$dest_dir/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --branch "$ref_name" --single-branch "$repo_url" "$dest_dir"
    return
  fi

  git -C "$dest_dir" remote set-url origin "$repo_url"
  git -C "$dest_dir" checkout "$ref_name"
  git -C "$dest_dir" pull --ff-only origin "$ref_name"
}

pull_large_files() {
  local dest_dir="$1"
  local include_pattern="$2"

  git -C "$dest_dir" lfs pull --include="$include_pattern" --exclude=""
}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

CACHE_ROOT="$REPO_ROOT/.cache/mir_sources"
OUTPUT_ROOT="$REPO_ROOT/datasets/mir"
TEXTVQA_SPLIT="validation"
CNNDM_VERSION="3.0.0"
CNNDM_SPLIT="validation"
MAX_IMAGES=""
MAX_TEXTS=""
OVERWRITE=0
SKIP_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-root)
      CACHE_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --textvqa-split)
      TEXTVQA_SPLIT="$2"
      shift 2
      ;;
    --cnndm-version)
      CNNDM_VERSION="$2"
      shift 2
      ;;
    --cnndm-split)
      CNNDM_SPLIT="$2"
      shift 2
      ;;
    --max-images)
      MAX_IMAGES="$2"
      shift 2
      ;;
    --max-texts)
      MAX_TEXTS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_cmd git
require_cmd python3

if ! git lfs version >/dev/null 2>&1; then
  echo "missing required command: git lfs" >&2
  exit 1
fi

git lfs install
if command -v git-xet >/dev/null 2>&1 || git xet --version >/dev/null 2>&1; then
  git xet install
else
  echo "warning: git-xet not found; continuing through the Git LFS bridge." >&2
fi

TEXTVQA_REPO="https://huggingface.co/datasets/lmms-lab/textvqa"
CNNDM_REPO="https://huggingface.co/datasets/abisee/cnn_dailymail"
TEXTVQA_DIR="$CACHE_ROOT/textvqa"
CNNDM_DIR="$CACHE_ROOT/cnn_dailymail"

mkdir -p "$CACHE_ROOT"

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  sync_repo "$TEXTVQA_REPO" "$TEXTVQA_DIR" "main"
  sync_repo "$CNNDM_REPO" "$CNNDM_DIR" "main"

  pull_large_files "$TEXTVQA_DIR" "data/${TEXTVQA_SPLIT}-*.parquet,default/${TEXTVQA_SPLIT}/*.parquet"
  pull_large_files "$CNNDM_DIR" "${CNNDM_VERSION}/${CNNDM_SPLIT}-*.parquet"
fi

DUMP_CMD=(
  python3
  "$REPO_ROOT/scripts/dump_mir_data.py"
  --textvqa-root "$TEXTVQA_DIR"
  --cnndm-root "$CNNDM_DIR"
  --output-root "$OUTPUT_ROOT"
  --textvqa-split "$TEXTVQA_SPLIT"
  --cnndm-version "$CNNDM_VERSION"
  --cnndm-split "$CNNDM_SPLIT"
)

if [[ -n "$MAX_IMAGES" ]]; then
  DUMP_CMD+=(--max-images "$MAX_IMAGES")
fi
if [[ -n "$MAX_TEXTS" ]]; then
  DUMP_CMD+=(--max-texts "$MAX_TEXTS")
fi
if [[ "$OVERWRITE" -eq 1 ]]; then
  DUMP_CMD+=(--overwrite)
fi

"${DUMP_CMD[@]}"
