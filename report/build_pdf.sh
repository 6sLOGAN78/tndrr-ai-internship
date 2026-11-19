#!/usr/bin/env bash
# Edit report.md, then run ./build_pdf.sh to produce report.pdf (needs python3-markdown + google-chrome)
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
import markdown
body = markdown.markdown(open("report.md").read(), extensions=["tables", "fenced_code"])
css = """
@page { size: A4; margin: 13mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #111; }
h1 { font-size: 14pt; margin: 0 0 2px; } h2 { font-size: 11.5pt; margin: 10px 0 3px; border-bottom: 1px solid #999; }
h3 { font-size: 9.5pt; margin: 7px 0 2px; } p, ul { margin: 3px 0; } ul { padding-left: 16px; }
table { border-collapse: collapse; margin: 4px 0; font-size: 9pt; } th, td { border: 1px solid #bbb; padding: 2px 5px; text-align: left; }
th { background: #eee; } code { font-size: 8pt; background: #f3f3f3; }
"""
open("report.html", "w").write(f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>")
PY
google-chrome --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=report.pdf report.html 2>/dev/null
python3 -c "import re;d=open('report.pdf','rb').read();print('pages:',len(re.findall(rb'/Type\s*/Page[^s]',d)))"
