# 8-page ACL version (anonymised for review)

- `main.tex`: main text, written for ACL Rolling Review (two-column ACL template, review mode).
- `appendix.tex`: generated from the long version by `tools/build_appendix.py paper_latex/body.tex paper_acl/appendix.tex` (Appendices A-H as in the long version; I-K hold Study 1-3 detail moved out of the main text; identifying links removed).
- Build: `latexmk -pdf main.tex`.
- The long, named version is in `paper_latex/`.
