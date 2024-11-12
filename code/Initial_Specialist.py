import json
import os
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm

def read_json(file_path):
    """Reads the content of a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json(data, file_path):
    """Writes data to a JSON file in the format {"rationale": data}."""
    formatted_data = {'rationale': data}
    with open(file_path, 'w') as file:
        json.dump(formatted_data, file, indent=4)

def load_image(image_folder, image_name):
    """Loads an image from a specific folder."""
    image_path = os.path.join(image_folder, image_name)
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        raise FileNotFoundError(f"Image not found: {image_path}")

def setup_gemini(gemini_api_key):
    """Set up the Gemini API with the given API key."""
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        raise Exception(f"An error occurred during Gemini setup: {e}")

def ask_gemini(question, choices, image):
    """Constructs and sends a request to the Gemini API. Chooses model based on whether an image is provided."""
    prompt = (f"As an agent/assistant of an experienced doctor, next steps are required. "
              f"Provide a reasonable rationale for the questions: {question} and image entered. "
              "Please proceed with a step-by-step analysis and provide a rationale.")
    
    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 2048
    }

    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }

    try:
        if image:
            model = genai.GenerativeModel(model_name='gemini-pro-vision')
            response = model.generate_content([prompt, image], generation_config=generation_config, safety_settings=safety_settings)
        else:
            model = genai.GenerativeModel(model_name='gemini-pro')
            response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        raise Exception(f"An error occurred during text generation: {e}")

def append_json(new_data, file_path):
    """Appends new data to an existing JSON file or creates a new file if it does not exist."""
    try:
        with open(file_path, 'r+') as file:
            file_data = json.load(file)
            file_data.update(new_data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
            file.truncate()
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            json.dump(new_data, file, indent=4)
    except json.JSONDecodeError:
        with open(file_path, 'w') as file:
            json.dump(new_data, file, indent=4)

def main():
    """Main function to execute the entire data processing workflow."""
    problems_file = 'path_to_problems_json'  # Path to your problems JSON file
    image_folder = 'path_to_image_folder'  # Your image folder path
    gemini_api_key = 'your_gemini_api_key'  # Replace with your Gemini API key
    results_file = 'path_to_results_json'  # Output JSON file

    problems = read_json(problems_file)
    setup_gemini(gemini_api_key)

    for key in tqdm(problems.keys(), desc="Processing questions"):
        value = problems[key]
        question = value['question']
        choices = value['choices']
        image_name = value.get('image')

        try:
            image = load_image(image_folder, image_name) if image_name else None
            rationale = ask_gemini(question, choices, image)
            append_json({key: rationale}, results_file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred during processing question {key}: {e}")

if __name__ == "__main__":
    main()
