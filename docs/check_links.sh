#!/usr/bin/env bash
# Re-verify every URL in the docs. Prints status code + URL; anything not 200 should be removed.
cd "$(dirname "$0")/.."
grep -ohE 'https?://[^ )|>`]+' docs/*.md README.md SYLLABUS.md data/README.md | sort -u | while read -r u; do
  echo "$(curl -sL -o /dev/null -w '%{http_code}' -A 'Mozilla/5.0' --max-time 20 "$u") $u"; done
