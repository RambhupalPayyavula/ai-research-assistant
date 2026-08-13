python3 -c "
import re

path = None
import importlib.util
spec = importlib.util.find_spec('ragas')
import os
path = os.path.join(os.path.dirname(spec.origin), 'llms', 'base.py')
print('Patching:', path)

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
    print('Exact line not found — file may already be patched or differ. Manual check needed.')
"