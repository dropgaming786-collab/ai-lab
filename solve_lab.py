"""
solve_lab.py
------------
Master One-Command Automated Laboratory Pipeline Runner for AI-101L.

Usage Examples:
    python solve_lab.py --lab 4             # Build Lab 04 deliverables and solve exercises
    python solve_lab.py --lab 4 --push      # Build Lab 04 and automatically push to GitHub
    python solve_lab.py --all               # Re-build all labs (01, 02, 03, 04)
    python solve_lab.py --auto              # Auto-detect newly added lab and complete it
"""

import os
import sys
import glob
import re
import argparse
import subprocess
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))

LAB_SCRIPTS = {
    1: os.path.join(ROOT, "scratch", "build_lab01.py"), # or inspect.py
    2: os.path.join(ROOT, "lab02", "build_deliverables.py"),
    3: os.path.join(ROOT, "lab03", "build_deliverables.py"),
    4: os.path.join(ROOT, "lab04", "build_deliverables.py"),
}

def run_command(cmd, desc=""):
    if desc:
        print(f"\n>>> {desc}")
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=ROOT)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        return False
    return True

def run_lab(lab_num, push=False):
    print("=" * 70)
    print(f"AUTOMATED RUNNER: LABORATORY {lab_num:02d}")
    print("=" * 70)

    script = LAB_SCRIPTS.get(lab_num)
    if not script or not os.path.exists(script):
        # Fallback check inside lab directory
        alt_script = os.path.join(ROOT, f"lab{lab_num:02d}", "build_deliverables.py")
        if os.path.exists(alt_script):
            script = alt_script
        else:
            print(f"[ERROR] No build script found for Laboratory {lab_num}")
            return False

    success = run_command(f"python \"{script}\"", f"Executing Lab {lab_num:02d} Builder")
    if not success:
        return False

    # Also run exercise solver for Lab 03 if applicable
    if lab_num == 3:
        ex3 = os.path.join(ROOT, "lab03", "solve_exercises.py")
        if os.path.exists(ex3):
            run_command(f"python \"{ex3}\"", "Running Lab 03 Exercise Solver")

    print(f"\n[SUCCESS] Laboratory {lab_num:02d} completed successfully!")

    if push:
        git_push(lab_num)

    return True

def git_push(lab_num):
    print("\n" + "=" * 70)
    print(f"AUTOMATED GIT PUSH FOR LAB {lab_num:02d}")
    print("=" * 70)
    run_command("git status", "Checking Git Status")
    run_command("git add .", "Staging all changes")
    commit_msg = f"Lab {lab_num:02d}: complete all deliverables, exercises, and reports"
    run_command(f'git commit -m "{commit_msg}"', "Committing changes")
    run_command("git push origin main", "Pushing to GitHub remote")
    print("\n[SUCCESS] Pushed all deliverables to GitHub!")

def auto_detect_new_lab():
    print("Scanning for newly uploaded laboratory notebooks...")
    notebooks = glob.glob(os.path.join(ROOT, "Lab*.ipynb"))
    detected = []
    for nb in notebooks:
        match = re.search(r"Lab[_\s]*0?(\d+)", os.path.basename(nb), re.IGNORECASE)
        if match:
            num = int(match.group(1))
            detected.append((num, nb))
    detected.sort()
    print(f"Detected notebooks: {[f'Lab {n}' for n, _ in detected]}")
    if detected:
        latest_num = detected[-1][0]
        print(f"Auto-selected latest lab: Lab {latest_num:02d}")
        return latest_num
    return None

def main():
    parser = argparse.ArgumentParser(description="Master AI-101L Laboratory Runner")
    parser.add_argument("--lab", type=int, help="Lab number to build (e.g. 1, 2, 3, 4)")
    parser.add_argument("--all", action="store_true", help="Run all available labs in sequence")
    parser.add_argument("--auto", action="store_true", help="Auto-detect and run latest uploaded lab")
    parser.add_argument("--push", action="store_true", help="Automatically commit and push to Git")

    args = parser.parse_args()

    if args.all:
        for num in sorted(LAB_SCRIPTS.keys()):
            run_lab(num)
        if args.push:
            git_push("01-04")
    elif args.lab:
        run_lab(args.lab, push=args.push)
    elif args.auto:
        lab_num = auto_detect_new_lab()
        if lab_num:
            run_lab(lab_num, push=args.push)
    else:
        # Default behavior: run latest detected or print help
        lab_num = auto_detect_new_lab()
        if lab_num:
            run_lab(lab_num, push=False)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
