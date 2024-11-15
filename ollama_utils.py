import subprocess
import requests
import time
import json

def start_and_check_ollama():
    print("Starting Ollama server...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Check if the Ollama server is running
    print("Checking if Ollama server is running...")
    for _ in range(10):
        try:
            response = requests.get("http://localhost:11434")
            if response.status_code == 200:
                print("Ollama server is running!")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        print("Failed to start Ollama server")
        return False

    # Check if the Ollama model is already downloaded
    print("Checking if Ollama model is downloaded...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = json.loads(response.text)["models"]
        model_info = next((m for m in models if m["name"] == "hf.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Q4_K_L"), None)
        if model_info:
            print("Ollama model is already downloaded!")
            wait_time = 200
        else:
            print("Ollama model is not downloaded yet. Waiting for it to download...")
            wait_time = 600
    except requests.exceptions.RequestException:
        print("Failed to check if Ollama model is downloaded")
        return False

    # Start the Ollama model
    print("Starting Ollama model...")
    subprocess.run(["ollama", "run", "hf.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Q4_K_L",">", "/dev/null", "2>&1", "&"], check=True)

    # Check if the Ollama model is running
    print("Checking if Ollama model is running...")
    for _ in range(wait_time):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "hf.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Q4_K_L",
                    "prompt": "What is the importance of AI?",
                    "options": {"num_predict": 1},
                    "stream": False
                }
            )
            if response.status_code == 200:
                print("Ollama model is running!")
                return True
        except requests.exceptions.RequestException:
            time.sleep(3)
    else:
        print("Failed to start Ollama model")
        return False
