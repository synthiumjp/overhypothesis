"""Patch: suppress warnings and add error handling"""
path = r'C:\overhypothesis\scripts\study1_linear_probe.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add warning suppression at the top after imports
old_import = "try:\n    from sklearn.linear_model import LogisticRegression"
new_import = "import warnings\nwarnings.filterwarnings('ignore', category=UserWarning, module='sklearn')\n\ntry:\n    from sklearn.linear_model import LogisticRegression"

if old_import in content:
    content = content.replace(old_import, new_import)
    print("Added warning suppression")

# Wrap the per-seed body in try/except
old_extract = """            # Extract hidden states
            hidden_dict, labels, item_ids = extract_hidden_states(
                model, tokenizer, fo_items, device)"""

new_extract = """            # Extract hidden states
            try:
                hidden_dict, labels, item_ids = extract_hidden_states(
                    model, tokenizer, fo_items, device)
            except Exception as e:
                print(f"    ERROR extracting hidden states: {e}")
                lines.append(f"  Seed {seed}: FAILED - {e}")
                del model
                continue"""

if old_extract in content:
    content = content.replace(old_extract, new_extract)
    print("Added error handling for hidden state extraction")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("DONE")
