from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class DiacriticsRestorer:
    def __init__(
        self,
        model_name: str = "iliemihai/mt5-base-romanian-diacritics",
        max_input_tokens: int = 384,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_tokens = max_input_tokens
        print(f"[diacritics] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"[diacritics] Model loaded on: {self.device}")

    def _split_text_in_chunks(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        chunks = []
        current = []

        for line in lines:
            candidate = "\n".join(current + [line]) if current else line
            token_count = len(
                self.tokenizer(
                    candidate,
                    truncation=False,
                    add_special_tokens=True
                )["input_ids"]
            )

            if token_count <= self.max_input_tokens:
                current.append(line)
            else:
                if current:
                    chunks.append("\n".join(current))
                    current = [line]
                else:
                    words = line.split()
                    temp = []
                    for w in words:
                        cand = " ".join(temp + [w]) if temp else w
                        tc = len(
                            self.tokenizer(
                                cand,
                                truncation=False,
                                add_special_tokens=True
                            )["input_ids"]
                        )
                        if tc <= self.max_input_tokens:
                            temp.append(w)
                        else:
                            if temp:
                                chunks.append(" ".join(temp))
                            temp = [w]
                    if temp:
                        chunks.append(" ".join(temp))
                    current = []

        if current:
            chunks.append("\n".join(current))

        return chunks

    @torch.no_grad()
    def restore_diacritics(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        chunks = self._split_text_in_chunks(text)
        if not chunks:
            return ""

        restored_chunks = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

            restored = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            restored_chunks.append(restored)

        return "\n".join(restored_chunks).strip()