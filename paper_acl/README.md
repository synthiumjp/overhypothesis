# 8-page ACL version (anonymised for review)

- `main.tex`: main text, written for ACL Rolling Review (two-column ACL template, review mode).
- `appendix.tex`: generated from the long version by `python paper_acl/tools/build_appendix.py paper_latex/body.tex paper_acl/appendix.tex --trim` (registration record A, deviations B, model settings C, data H and language I; the earlier study's other appendices stay in the long version, supplied as supplementary material; identifying links removed). Other appendices: `build_appendix_recut.py` (L), `build_appendix_a9.py` (M), `build_appendix_a10.py` (N). Figures: `scripts/figure1_carry_design.py`, `scripts/amendment9_figures.py`, `scripts/amendment10_figures.py`.
- Build: `latexmk -pdf main.tex`.
- The long, named version is in `paper_latex/`.
