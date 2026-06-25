# check_key.py
import os, sys
k = os.getenv("OPENAI_API_KEY")
print("Python:", sys.executable)
if not k:
    print("OPENAI_API_KEY not found in environment.")
else:
    print("OPENAI_API_KEY found. length =", len(k))
    print("masked:", k[:6] + "..." + k[-4:])
