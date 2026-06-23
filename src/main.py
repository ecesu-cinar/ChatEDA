import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
#from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent

QA_PATH = BASE_DIR / "data" / "private" / "eda_qa.json"
GIF_PATH = BASE_DIR / "data" / "private" / "gifs.json"
PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def format_examples(qa_data, limit = 40):
    examples = qa_data[:limit]

    text = ""
    for item in examples:
        text += f"Question: {item['question']}\n"
        text += f"Eda answer: {item['answer']}\n"
        if "gif" in item:
            text += f"GIF: {item['gif']}\n"
        text += "\n"

    return text

def format_gif(gifs):
    text = ""
    for gif in gifs:
        text += f"ID: {gif['id']}\n"
        text += f"Description: {gif['description']}\n"
        text += f"Vibes: {', '.join(gif['vibes'])}\n\n"

    return text

def clean_json_response(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text

def main():
    load_dotenv()

    #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    qa_data = load_json(QA_PATH)
    gifs = load_json(GIF_PATH)
    system_prompt = load_txt(PROMPT_PATH)

    examples_text = format_examples(qa_data)
    gifs_text = format_gif(gifs)

    print("I hope this works. Type 'exit' in order to quit \n")

    while True:
        user_input = input("User: ")

        if user_input.lower() == "exit":
            print("Wow okay then, baibai")
            break

        user_prompt = f"""

        Here are examples of how Eda talks:

        {examples_text}

        Available GIFs:

        {gifs_text}

        Now reply to this message as Eda:
        "{user_input}"

        Return ONLY valid JSON like this:
        {{
            "message": "text response",
            "gif": null
        }}

        If a GIF fits, use its ID instead of null.
        """
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        raw_response = response.choices[0].message.content
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system_prompt}\n\n{user_prompt}"
        )

        raw_response = response.text


        try:
            cleaned_response = clean_json_response(raw_response)
            parsed = json.loads(cleaned_response)
            print("\nChatEDA:", parsed["message"])

            if parsed["gif"]:
                print("GIF:", parsed["gif"])

        except json.JSONDecodeError:
            print("\n Awe the model did not return valid JSON:")
            print(raw_response)

        print()

if __name__ == "__main__":
    main()