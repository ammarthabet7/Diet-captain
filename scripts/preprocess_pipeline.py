import re
import json
import os

# --- 1. CONFIGURATION ---
IRRELEVANT_PATTERNS = [
    r"Page \d+",
    r"صفحة \d+",
    r"Diet & Cheat",
    r"DIET & CHEAT",
    r"Click here to go back",
    r"اضغط هنا للرجوع",
    r"www\.[a-zA-Z0-9-]+\.com",
    r"http[s]?://\S+",
    r"_{3,}",
    r"\*{3,}"
]

# --- 2. HELPER FUNCTIONS ---

def clean_arabic_text(text):
    if not isinstance(text, str):
        return text
    
    # Remove Tashkeel
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, "", text)
    
    # Remove Tatweel
    text = re.sub(r'ـ', '', text)
    
    # Normalize Alef
    text = re.sub(r"[أإآ]", "ا", text)
    
    return text

def clean_general_noise(text):
    if not isinstance(text, str):
        return text

    for pattern in IRRELEVANT_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Fix Mixed-Language Spacing
    text = re.sub(r'([a-zA-Z])([\u0600-\u06FF])', r'\1 \2', text)
    text = re.sub(r'([\u0600-\u06FF])([a-zA-Z])', r'\1 \2', text)

    # Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- MOVED THIS FUNCTION OUTSIDE ---
def clean_json_obj(obj):
    """
    Recursive function to clean a JSON object (dict or list).
    """
    if isinstance(obj, dict):
        return {k: clean_json_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_obj(i) for i in obj]
    elif isinstance(obj, str):
        val = clean_arabic_text(obj)
        val = clean_general_noise(val)
        return val
    else:
        return obj

def process_file(file_path):
    print(f"Processing {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found -> {file_path}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    output_path = None
    
    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Now this function is definitely defined!
        cleaned_data = clean_json_obj(data)
        
        output_path = file_path.replace('.json', '_clean.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
    elif ext == '.txt' or ext == '.md':
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        cleaned_content = clean_arabic_text(content)
        cleaned_content = clean_general_noise(cleaned_content)
        
        if ext == '.txt':
            output_path = file_path.replace('.txt', '_clean.txt')
        else:
            output_path = file_path.replace('.md', '_clean.md')
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
    
    else:
        print(f"Skipping {file_path}: Unknown extension {ext}")
        return

    if output_path:
        print(f"Saved to {output_path}")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure this matches your ACTUAL folder structure
    files_to_clean = [
        "food_exchange.json",
        "nutration-ebook.md",
        "Training-Guide-Diet-Cheat-1.md"
    ]
    
    # Run loop
    for file in files_to_clean:
        process_file(file)
