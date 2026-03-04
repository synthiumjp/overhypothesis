"""Patch: fix sklearn issues in study1_linear_probe.py"""
path = r'C:\overhypothesis\scripts\study1_linear_probe.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Replace LogisticRegression params
old = """    clf = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        C=1.0,
        random_state=42
    )"""

new = """    clf = LogisticRegression(
        max_iter=5000,
        solver='saga',
        C=0.1,
        penalty='l2',
        random_state=42,
    )"""

if old in content:
    content = content.replace(old, new)
    print("Fixed LogisticRegression params")
else:
    print("WARNING: LogisticRegression block not found")

# Fix 2: Reduce CV folds for small classes
old2 = """    if n_samples < cv_folds * n_classes:
        cv_folds = min(cv_folds, n_samples // max(n_classes, 1))
        if cv_folds < 2:
            cv_folds = 2"""

new2 = """    # Use 3-fold to handle class imbalance (some shapes have only 2 items)
    cv_folds = 3
    if n_samples < cv_folds * n_classes:
        cv_folds = 2"""

if old2 in content:
    content = content.replace(old2, new2)
    print("Fixed CV folds")
else:
    print("WARNING: CV folds block not found")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("DONE")
