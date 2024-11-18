import subprocess
import requests
import time
import json
import shutil


def start_and_check_ollama(model_name):
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Check if the Ollama server is running
    for _ in range(10):
        try:
            response = requests.get("http://localhost:11434")
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        return False

    # Check if the Ollama model is already downloaded
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = json.loads(response.text)["models"]
        model_info = next((m for m in models if m["name"] == model_name), None)
        if model_info:
            wait_time = 200
        else:
            wait_time = 600
    except requests.exceptions.RequestException:
        return False

    # Start the Ollama model
    subprocess.Popen(f"ollama run {model_name} > /dev/null 2>&1 &",shell=True)

    # Check if the Ollama model is running
    for _ in range(wait_time):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "What is the importance of AI?",
                    "options": {"num_predict": 1},
                    "stream": False
                }
            )
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    else:
        return False
