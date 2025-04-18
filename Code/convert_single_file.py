import requests

# === CONFIGURATION ===
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:latest"
INPUT_FILE = "APmodel/19742019/APmain.m"   # Replace with your actual file path
OUTPUT_FILE = "APmain_converted.py"

def convert_matlab_to_python(input_file, output_file):
    # Read the MATLAB code
    with open(input_file, 'r', encoding='utf-8') as f:
        matlab_code = f.read()


    # Enforced prompt with system context in the instruction
    prompt = (
        "Convert the following MATLAB code to Python. Just return well formatted python code. I should be able to run your output as a .py file"#Only write python code and use inline comments starting with #. I should be able to copy and paste your answer in a .py file and run it without errors.\n"
        f"{matlab_code}"
    )

    # Send the request to Ollama
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    print(f"[INFO] Sending code from: {input_file}")
    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()

    data = response.json()
    python_code = data.get("response", "")

    if not python_code.strip():
        print("[WARNING] Empty response from the model.")
        return

    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(python_code)
        print(f"[OK] Saved converted code to: {output_file}")

# === RUN IT ===
if __name__ == "__main__":
    convert_matlab_to_python(INPUT_FILE, OUTPUT_FILE)
