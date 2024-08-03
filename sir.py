import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import zipfile
import os
from tkinter import Tk, filedialog
def fetch_prediction(text):
    try:
        response = requests.post(
            "https://hello-simpleai-chatgpt-detector-single.hf.space/run/predict_en",
            json={"data": [text]},
            timeout=60  # Timeout in seconds
        )
        response_json = response.json()
        if 'data' in response_json:
            data = response_json['data']
            if len(data) >= 2:
                label = data[0]
                confidence_score = float(data[1])
                return label, confidence_score
        return None, None
    except requests.RequestException as e:
        print(f"Request failed for '{text}': {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"JSON decode error for '{text}': {e}")
        return None, None
def get_multiple_predictions(texts):
    labels = []
    confidence_scores = []
    total_tasks = len(texts)
    completed_tasks = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_text = {executor.submit(fetch_prediction, text): text for text in texts}
        for future in as_completed(future_to_text):
            try:
                label, confidence = future.result()
                labels.append(label)
                confidence_scores.append(confidence)
            except Exception as e:
                print(f"Error retrieving result: {e}")
                labels.append(None)
                confidence_scores.append(None)
            completed_tasks += 1
            print(f"Progress: {completed_tasks}/{total_tasks} completed")

    return labels, confidence_scores

def process_csv(file_path):
    df = pd.read_csv(file_path)
    texts = list(df['body'])
    labels, confidence_scores = get_multiple_predictions(texts)
    df['label'] = labels
    df['confidence_score'] = confidence_scores
    df = df[df['confidence_score'].notna()]
    return df


def select_files():
    root = Tk()
    root.withdraw()  # Hiding the root window
    file_paths = filedialog.askopenfilenames(title="Select CSV Files", filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    return file_paths


def process_and_save_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    print(f"Processing {filename}...")
    df = process_csv(file_path)
    output_file_path = os.path.join(output_dir, f"processed_{filename}")
    df.to_csv(output_file_path, index=False)


def main():
    
    file_paths = select_files()
    
    if not file_paths:
        print("No files selected.")
        return


    output_dir = "processed_files"
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=len(file_paths)) as executor:
        futures = [executor.submit(process_and_save_file, file_path, output_dir) for file_path in file_paths]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")


    zip_filename = "processed_files.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)

    print(f"Processed files are zipped into {zip_filename}")

if __name__ == "__main__":
    main()
