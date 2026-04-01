#!/usr/bin/env bash
#
# hf_hidden_gems.sh - Find undervalued "hidden gem" models on Hugging Face
#
# This script identifies models with high likes-to-downloads ratios,
# indicating quality models that haven't gone viral yet.
#
# Usage: hf_hidden_gems.sh [OPTIONS]
#
# Options:
#   -n, --limit N         Number of models to fetch (default: 100)
#   -m, --min-downloads N Minimum downloads to consider (default: 100)
#   -t, --top N           Show top N gems (default: 20)
#   -p, --pipeline TAG    Filter by pipeline tag (e.g., text-generation)
#   -s, --sort-by FIELD   Sort by: ratio (default), likes, downloads, trending
#   -r, --reverse         Reverse sort order
#   --json                Output as JSON array
#   -h, --help            Show this help message
#
# Examples:
#   hf_hidden_gems.sh                           # Find top 20 hidden gems
#   hf_hidden_gems.sh -n 500 -t 50              # Search 500 models, show top 50
#   hf_hidden_gems.sh -p text-generation        # Only text generation models
#   hf_hidden_gems.sh --json | jq '.[] | {id, ratio}'  # JSON output for piping
#
# Environment:
#   HF_TOKEN    Hugging Face API token (optional but recommended for higher rate limits)

set -euo pipefail

# Defaults
LIMIT=100
MIN_DOWNLOADS=100
TOP=20
PIPELINE=""
SORT_BY="ratio"
REVERSE=0
JSON_OUTPUT=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--limit)
      LIMIT="$2"
      shift 2
      ;;
    -m|--min-downloads)
      MIN_DOWNLOADS="$2"
      shift 2
      ;;
    -t|--top)
      TOP="$2"
      shift 2
      ;;
    -p|--pipeline)
      PIPELINE="$2"
      shift 2
      ;;
    -s|--sort-by)
      SORT_BY="$2"
      shift 2
      ;;
    -r|--reverse)
      REVERSE=1
      shift
      ;;
    --json)
      JSON_OUTPUT=1
      shift
      ;;
    -h|--help)
      sed -n '/^#/p' "$0" | sed 's/^# //; s/^#$//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use --help for usage information" >&2
      exit 1
      ;;
  esac
done

# Build API URL
API_URL="https://huggingface.co/api/models?limit=${LIMIT}"
if [[ -n "$PIPELINE" ]]; then
  API_URL="${API_URL}&pipeline_tag=${PIPELINE}"
fi

# Fetch models
declare -a HEADERS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HEADERS=("-H" "Authorization: Bearer ${HF_TOKEN}")
fi

MODELS=$(curl -s "${HEADERS[@]}" "$API_URL")

# Check if we got valid JSON
if ! echo "$MODELS" | jq -e '.' > /dev/null 2>&1; then
  echo "Error: Failed to fetch models from API" >&2
  exit 1
fi

# Calculate hidden gem score and filter
RESULTS=$(echo "$MODELS" | jq --arg min_dl "$MIN_DOWNLOADS" '
  [.[] 
  | select(.downloads != null and .likes != null)
  | select(.downloads >= ($min_dl | tonumber))
  | {
      id,
      likes,
      downloads,
      ratio: (.likes / .downloads),
      pipeline_tag: (.pipeline_tag // "unknown"),
      library_name: (.library_name // "unknown"),
      createdAt: (.createdAt // "unknown"),
      trendingScore: (.trendingScore // 0),
      tags
    }
  ]'
)

# Sort results
SORT_KEY=".ratio"
case "$SORT_BY" in
  likes) SORT_KEY=".likes" ;;
  downloads) SORT_KEY=".downloads" ;;
  trending) SORT_KEY=".trendingScore" ;;
  ratio|*) SORT_KEY=".ratio" ;;
esac

if [[ "$REVERSE" -eq 1 ]]; then
  SORTED=$(echo "$RESULTS" | jq "sort_by($SORT_KEY)")
else
  SORTED=$(echo "$RESULTS" | jq "sort_by($SORT_KEY) | reverse")
fi

# Take top N
TOP_RESULTS=$(echo "$SORTED" | jq ".[:$TOP]")

# Output
if [[ "$JSON_OUTPUT" -eq 1 ]]; then
  echo "$TOP_RESULTS" | jq '.'
else
  # Pretty table output
  echo ""
  printf "┌%-50s┬%10s┬%12s┬%10s┬%20s┐\n" "──────────────────────────────────────────────────" "──────────" "────────────" "──────────" "────────────────────"
  printf "│%-50s│%10s│%12s│%10s│%20s│\n" " Model ID" " Likes" "Downloads" "Ratio" "Pipeline Tag"
  printf "├%-50s┼%10s┼%12s┼%10s┼%20s┤\n" "──────────────────────────────────────────────────" "──────────" "────────────" "──────────" "────────────────────"
  
  echo "$TOP_RESULTS" | jq -r '.[] | 
    "\(.id)|\(.likes)|\(.downloads)|\(.ratio)|\(.pipeline_tag)"' | \
  while IFS='|' read -r id likes downloads ratio pipeline; do
    # Truncate long IDs
    if [[ ${#id} -gt 48 ]]; then
      id="${id:0:45}..."
    fi
    # Format ratio to 4 decimal places
    formatted_ratio=$(printf "%.4f" "$ratio")
    printf "│%-50s│%10s│%12s│%10s│%20s│\n" " $id" " $likes" " $downloads" " $formatted_ratio" " $pipeline"
  done
  
  printf "└%-50s┴%10s┴%12s┴%10s┴%20s┘\n" "──────────────────────────────────────────────────" "──────────" "────────────" "──────────" "────────────────────"
  echo ""
  echo "Hidden Gem Score = Likes / Downloads (higher = more undervalued)"
  echo "Found $(echo "$RESULTS" | jq 'length') models with ≥$MIN_DOWNLOADS downloads, showing top $TOP by $SORT_BY"
fi
