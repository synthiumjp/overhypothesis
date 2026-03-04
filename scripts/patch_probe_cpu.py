"""Patch: force CPU for linear probe to avoid DirectML issues"""
path = r'C:\overhypothesis\scripts\study1_linear_probe.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Force CPU device selection
old = """    # Device
    if DML_AVAILABLE:
        device = torch_directml.device()
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")"""

new = """    # Force CPU for probe -- only 80 items, speed irrelevant,
    # avoids DirectML tensor issues with hidden state extraction
    device = torch.device('cpu')
    print(f"Device: {device} (CPU forced for probe stability)")"""

if old in content:
    content = content.replace(old, new)
    print("Forced CPU device")
else:
    print("WARNING: device block not found")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("DONE")
