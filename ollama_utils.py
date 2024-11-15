import subprocess
import requests
import time
import json
import shutil
import os

def install_ollama():
    """
    Check if Ollama is installed and install it if not.
    Returns True if Ollama is available (either pre-existing or newly installed).
    """
    print("Checking if Ollama is installed...")
    
    # Check if ollama executable exists in PATH
    if shutil.which('ollama'):
        print("Ollama is already installed!")
        return True
    
    print("Ollama not found. Installing Ollama...")
    try:
        # Download and execute the install script
        install_command = "curl -fsSL https://ollama.com/install.sh | sh"
        result = subprocess.run(install_command, shell=True, check=True, capture_output=True, text=True)
        
        # Verify installation was successful
        if shutil.which('ollama'):
            print("Ollama has been successfully installed!")
            return True
        else:
            print("Failed to verify Ollama installation.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Ollama. Error: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during installation: {e}")
        return False


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
        model_info = next((m for m in models if m["name"] == "hf.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Q3_K_L"), None)
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
    subprocess.Popen("ollama run hf.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Q3_K_L > /dev/null 2>&1 &",shell=True)

    # Check if the Ollama model is running
    print("Checking if Ollama model is running...")
    for _ in range(wait_time):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "hf.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Q3_K_L",
                    "prompt": "What is the importance of AI?",
                    "options": {"num_predict": 1},
                    "stream": False
                }
            )
            if response.status_code == 200:
                print("Ollama model is running!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    else:
        print("Failed to start Ollama model")
        return False
