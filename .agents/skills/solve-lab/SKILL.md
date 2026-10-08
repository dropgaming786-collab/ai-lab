---
name: solve-lab
description: Automatically inspects newly uploaded laboratory notebooks, solves all in-lab exercises and home assignments, builds models, exports figures, writes report.md, updates docs/GUIDE.md, and pushes everything to GitHub with a single command.
---

# Single-Command Laboratory Automation Skill (`/solve-lab`)

Use this skill whenever the user asks to solve a newly added lab (e.g., "solve lab 04", "do lab 5", or typing `/solve-lab`).

## Autonomous Execution Runbook

When invoked, the agent performs the complete end-to-end laboratory completion process automatically:

### 1. Discovery & Notebook Inspection
1. Scan `E:\Ai lab` for new or target notebooks: `Lab XX.ipynb` or `Lab_XX_*.ipynb`.
2. Parse all sections, in-lab exercises, home tasks, and deliverables table.
3. Identify required datasets, libraries, and model architectures.

### 2. Workspace & Deliverable Structure Setup
1. Create the dedicated folder `labXX/` and subfolder `labXX/figures/`.
2. Build an automated, reproducible builder `labXX/build_deliverables.py`.

### 3. Model Training, Preprocessing & Evaluation
1. Implement all data pipelines and splits (`train_test_split`).
2. Train required regression/classification/multimodal models.
3. Compute metrics ($R^2$, accuracy, precision, recall, F1, confusion matrix).
4. Persist trained models/pipelines to disk (e.g. `*.pkl` with `joblib`, `*.npz` with `numpy`).
5. Export high-resolution diagnostic plots to `labXX/figures/*.png`.

### 4. Solve All In-Lab Exercises & Home Tasks
1. Execute and format in-lab exercise code and explanations.
2. Formulate and solve the home assignment.
3. Patch the laboratory notebook with executed cells, outputs, and written markdown answers.
4. Synchronize the executed notebook with the root `Lab XX.ipynb`.

### 5. Report & Documentation Generation
1. Write a comprehensive, rigorous academic report `labXX/report.md` covering methodology, metrics, coefficient analysis, and interpretations.
2. Update the master documentation `docs/GUIDE.md` with the new laboratory's complete technical breakdown.

### 6. Git Synchronization & Verification
1. Run `git status` to verify clean staging.
2. Stage all deliverables: `git add labXX/ docs/GUIDE.md Lab*.ipynb solve_lab.py`.
3. Commit with a descriptive message: `git commit -m "Lab XX: complete all deliverables, exercises, and reports"`.
4. Push to remote: `git push origin main`.
5. Verify clean working tree and notify the user with a concise summary.
