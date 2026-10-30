#!/usr/bin/env bash
# Usage (from repo root, with `python3 -m http.server 8765` running there): bash site/tests/smoke.sh
B=${B:-http://127.0.0.1:8765}; fail=0; n=0
for f in site/*.html site/days/*.html site/weeks/*.html site/exams/*.html site/assets/*; do
  c=$(curl -s -o /dev/null -w '%{http_code}' "$B/$f"); n=$((n+1)); [ "$c" = 200 ] || { echo "FAIL $c $f"; fail=$((fail+1)); }
done
# every relative link in generated pages resolves (code/, weeks/*.md, pdf/)
links=$(grep -oh 'href="[^"#:]*"' site/*.html site/*/*.html | sort -u | wc -l)
for p in site/*.html site/*/*.html; do d=$(dirname "$p")
  grep -o 'href="[^"#:]*"' "$p" | sed 's/href="//;s/"$//' | while read -r h; do [ -e "$d/$h" ] || echo "BROKEN $p -> $h"; done
done > /tmp/broken.txt
echo "pages+assets: $n checked, $fail non-200; unique relative hrefs: $links; broken: $(wc -l < /tmp/broken.txt)"
head /tmp/broken.txt; [ $fail = 0 ] && [ ! -s /tmp/broken.txt ]
