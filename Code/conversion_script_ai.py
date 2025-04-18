import os
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-coder"

def query_ollama(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
    return response.json()["response"]

def convert_matlab_to_python(src_root, dst_root):
    for root, _, files in os.walk(src_root):
        for file in files:
            if file.endswith(".m"):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_root)
                dst_path = os.path.join(dst_root, rel_path).replace(".m", ".py")

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                with open(src_path, "r", encoding="utf-8") as f:
                    matlab_code = f.read()

                print(f"Converting {rel_path}...")

                prompt = f"""Convert the following MATLAB code to clean, idiomatic Python with comments for each line:

```matlab
{matlab_code}
```"""

                try:
                    python_code = query_ollama(prompt)
                    with open(dst_path, "w", encoding="utf-8") as f_out:
                        f_out.write(python_code)
                except Exception as e:
                    print(f"❌ Failed to convert {rel_path}: {e}")

# Example usage:
convert_matlab_to_python("APmodel", "PyConverted_APmodel")
