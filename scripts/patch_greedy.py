"""Quick patch: fix context extraction in study1_greedy_generation.py"""
import re

path = r'C:\overhypothesis\scripts\study1_greedy_generation.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old = """        context_text = item.get('context', '')
        if not context_text:
            # Reconstruct from target_text: everything up to the target token
            target_text = item.get('target_text', item.get('text', ''))
            target_token = item.get('target', '')
            if target_token and target_text:
                # Find position of target and take everything before it
                idx = target_text.rfind(target_token)
                if idx > 0:
                    context_text = target_text[:idx].rstrip()
                else:
                    context_text = target_text
            else:
                continue"""

new = """        context_text = item.get('context', '')
        if not context_text:
            # Extract context from frame field: "Another pell is a ___."
            frame = item.get('frame', '')
            if '___' in frame:
                context_text = frame.split('___')[0].rstrip()
            elif frame:
                context_text = frame
            else:
                continue"""

if old in content:
    content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("PATCHED successfully")
else:
    print("ERROR: old text not found - check script manually")
