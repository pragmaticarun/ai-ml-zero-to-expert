# Week PDFs

Generated with `markdown` + `xhtml2pdf` (Helvetica). Some Unicode maths glyphs are ASCII-folded. Spot-check: `pdftotext pdf/week-01.pdf - | head` and `pdftotext pdf/week-26.pdf - | head` show expected titles.

Pandoc/`pdflatex` were not available without TinyTeX; `pypandoc_binary` alone cannot emit PDF without a LaTeX engine.
