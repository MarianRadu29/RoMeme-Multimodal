# diacritics.py
#pip install transformers torch sentencepiece
#pip install protobuf tiktoken
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class DiacriticsRestorer:
    def __init__(self):
        print("Loading the diacritics restoration model (this will take a moment to download the first time)...")
        model_name = "iliemihai/mt5-base-romanian-diacritics"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model loaded successfully!")

    def restore_diacritics(self, text: str) -> str:
        """
        Takes a string without diacritics and returns the grammatically corrected string.
        """
        if not text or not isinstance(text, str):
            return ""

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        outputs = self.model.generate(**inputs, max_length=512)
        
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return corrected_text


#testing
if __name__ == "__main__":
    print("--- Starting Phase 4 Testing: Diacritics Restoration ---\n")
    
    restorer = DiacriticsRestorer()
    
    test_texts = [
        "pe langa plopii fara sot adesea am trecut",
        "romania este o tara cu o geografie foarte variata",
        "cand pisica nu-i acasa joaca soarecii pe masa"
    ]
    
    for raw_text in test_texts:
        result_text = restorer.restore_diacritics(raw_text)
        
        print(f"Normalized input:   {raw_text}")
        print(f"Diacritized output: {result_text}")
        print("-" * 60)