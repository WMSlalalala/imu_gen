#!/usr/bin/env bash
set -euo pipefail

interval_seconds="${1:-1800}"
if [[ ! "$interval_seconds" =~ ^[0-9]+$ ]] || (( interval_seconds < 60 )); then
  echo "error: interval must be an integer >= 60 seconds" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while true; do
  if ! bash "$script_dir/update_snapshot.sh"; then
    echo "sync failed at $(date '+%Y-%m-%d %H:%M:%S %Z'); will retry" >&2
  fi
  sleep "$interval_seconds"
done
