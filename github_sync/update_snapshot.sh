#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -d .git ]]; then
  echo "error: $repo_root is not initialized as a Git repository" >&2
  exit 2
fi

git add --all -- \
  .gitignore README.md github_sync \
  trajectory_humanization_full_20260722_v16_numeric_recovery \
  trajectory_estimator_pack_20260721 \
  trajectory_pad_supplement_20260722 \
  android_duration_time_fixed_20260720/imu_release_20260721

if git diff --cached --quiet; then
  echo "no tracked source/document changes"
else
  stamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  git commit -m "sync: formal IMU/trajectory snapshot $stamp"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "error: origin is not configured; local commit retained" >&2
  exit 3
fi

git push origin HEAD:main
