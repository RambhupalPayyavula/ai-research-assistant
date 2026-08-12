#!/bin/bash
# Patches a known ragas 0.3.9/0.4.x bug: llms/base.py imports ChatVertexAI from a
# langchain_community path that no longer exists in modern langchain_community versions.
# This project never uses VertexAI — this patch makes the import optional instead of fatal.
python3 -c "
import importlib.util, os
spec = importlib.util.find_spec('ragas')
path = os.path.join(os.path.dirname(spec.origin), 'llms', 'base.py')
with open(path, 'r') as f:
    content = f.read()
old = 'from langchain_community.chat_models.vertexai import ChatVertexAI'
new = '''try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ImportError:
    ChatVertexAI = None'''
if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('Patched successfully.')
else:
    print('Already patched or file differs — no changes made.')
"
