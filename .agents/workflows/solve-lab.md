---
description: Automatically complete any newly uploaded AI-101L laboratory session end-to-end
---

# /solve-lab Workflow

1. Detect target laboratory notebook (e.g., `Lab 04.ipynb` or user-specified lab).
2. Execute the corresponding builder or `python solve_lab.py --lab <number>`.
3. Verify that all models, figures, notebooks, and reports exist in `labXX/`.
4. Update `docs/GUIDE.md` and project tracking.
5. Push all deliverables to GitHub `main` branch.
