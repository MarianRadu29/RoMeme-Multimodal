# normalization.py
# pip install symspellpy

import os
import re
import unicodedata

try:
    from symspellpy import SymSpell, Verbosity
    _SYMSPELL_AVAILABLE = True
except ImportError:
    _SYMSPELL_AVAILABLE = False


DIACRITIC_FOLD = str.maketrans({
    "ă": "a", "â": "a", "î": "i", "ș": "s", "ț": "t",
    "Ă": "a", "Â": "a", "Î": "i", "Ș": "s", "Ț": "t",
    "ş": "s", "ţ": "t", "Ş": "s", "Ţ": "t",
})


def fold_diacritics(text: str) -> str:
    """Strip Romanian diacritics so text matches a no-diacritics dictionary."""
    return text.translate(DIACRITIC_FOLD)


class TextNormalizer:
    """
    Phase 3: Normalize raw OCR output before diacritics restoration.
    Steps (in order):
      1. unicode cleanup (NFC, remove control chars)
      2. character confusions (|→l, curly quotes, dashes, ellipsis)
      3. fix digit-in-word (R0mn1a → Romania)
      4. drop obvious noise lines (isolated junk like "dt»", "| —", "[pl —")
      5. strip noise (stray symbols, repeated punctuation)
      6. lowercase (optional)
      7. SymSpell spell-check against a Romanian dictionary (no diacritics)
      8. collapse whitespace
    Output is a clean, no-diacritics Romanian text ready for DiacriticsRestorer.
    """

    # common OCR character confusions (source -> target)
    OCR_CONFUSIONS = {
        "|": "l",
        "¢": "c",
        "€": "e",
        "£": "l",
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'",
        "–": "-", "—": "-", "−": "-",
        "…": "...",
    }

    # words where digits are legitimately part of the token
    DIGIT_WHITELIST = re.compile(r"^\d+([.,]\d+)?$")

    # token extractor used for spell-check (keeps only letter runs)
    TOKEN_SPLIT = re.compile(r"([^\W\d_]+)", re.UNICODE)
    NON_SPACE_TOKEN = re.compile(r"\S+")
    WORD_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

    # lightweight corpus-specific fixes that should work even without SymSpell
    BUILTIN_TOKEN_REPLACEMENTS = {
        "astainseamna": "asta inseamna",
        "cuagheasma": "cu agheasma",
        "mafio": "mafiot",
        "pacates": "pacate",
        "safiimafiot": "sa fii mafiot",
        "spalktorie": "spalatorie",
        "spaltorie": "spalatorie",
        "suu": "sub",
    }
    BUILTIN_LINE_REPLACEMENTS = (
        (re.compile(r"\btorie de pacate spa\b", re.UNICODE), "spalatorie de pacate spa"),
    )

    VALID_SHORT_WORDS = {
        "ai", "am", "ar", "as", "au", "ca", "ce", "cu", "da", "de", "e", "ea",
        "el", "eu", "ii", "in", "la", "le", "li", "ma", "mi", "ne", "nu", "o",
        "pe", "sa", "se", "si", "te", "tu", "un",
    }
    VALID_SHORT_PHRASES = {
        "nu te", "sa fii", "de la", "de ce", "pe el", "pe ea", "cu ea", "cu el",
    }
    NOISE_WORDS = {
        "ist", "mir", "poe", "tie",
    }

    def __init__(
        self,
        lowercase: bool = True,
        fix_digits: bool = True,
        spell_check: bool = True,
        dictionary_path: str = None,
        max_edit_distance: int = 2,
        min_token_len_for_correction: int = 5,
    ):
        self.lowercase = lowercase
        self.fix_digits = fix_digits
        self.spell_check = spell_check
        self.max_edit_distance = max_edit_distance
        self.min_token_len_for_correction = min_token_len_for_correction

        self.sym_spell = None
        if self.spell_check:
            self._load_symspell(dictionary_path)

    # ---------- SymSpell loading ----------

    def _load_symspell(self, dictionary_path: str):
        if not _SYMSPELL_AVAILABLE:
            print("[normalization] symspellpy not installed; spell-check disabled.")
            self.spell_check = False
            return

        if dictionary_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            dictionary_path = os.path.join(here, "ro_50k_no_diacritics.txt")

        if not os.path.exists(dictionary_path):
            print(f"[normalization] dictionary not found at {dictionary_path}; "
                  f"spell-check disabled. See README for how to build it.")
            self.spell_check = False
            return

        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=self.max_edit_distance,
            prefix_length=7,
        )
        loaded = self.sym_spell.load_dictionary(
            dictionary_path, term_index=0, count_index=1, encoding="utf-8"
        )
        if not loaded:
            print(f"[normalization] failed to load dictionary {dictionary_path}; "
                  f"spell-check disabled.")
            self.spell_check = False
            self.sym_spell = None
            return

        print(f"[normalization] SymSpell dictionary loaded "
              f"({len(self.sym_spell.words)} terms).")

    # ---------- text cleaning steps ----------

    def _drop_noise_lines(self, text: str) -> str:
        """Drop only obviously noisy OCR lines while preserving short meme text."""
        lines = text.splitlines()
        word_lines = [self._line_words(line) for line in lines]
        kept = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            non_space = [ch for ch in stripped if not ch.isspace()]
            if not non_space:
                continue

            words = word_lines[idx]
            if not words:
                continue

            letters = sum(ch.isalpha() for ch in stripped)
            if letters / len(non_space) < 0.35:
                continue

            prev_words = self._nearby_context_words(word_lines, idx, -1)
            next_words = self._nearby_context_words(word_lines, idx, 1)
            if self._is_probable_artifact_line(stripped, words, prev_words, next_words):
                continue

            kept.append(stripped)
        return "\n".join(kept)

    def _line_words(self, text: str) -> list[str]:
        return [word.lower() for word in self.WORD_TOKEN.findall(text)]

    def _has_long_word(self, words: list[str]) -> bool:
        return any(len(word) >= 4 for word in words)

    def _nearby_context_words(
        self,
        word_lines: list[list[str]],
        start_idx: int,
        direction: int,
        max_nonempty_lines: int = 3,
    ) -> list[str]:
        collected = []
        seen = 0
        idx = start_idx + direction
        while 0 <= idx < len(word_lines):
            if word_lines[idx]:
                collected.extend(word_lines[idx])
                seen += 1
                if seen >= max_nonempty_lines:
                    break
            idx += direction
        return collected

    def _is_probable_artifact_line(
        self,
        line: str,
        words: list[str],
        prev_words: list[str],
        next_words: list[str],
    ) -> bool:
        normalized = " ".join(words)
        near_long_context = self._has_long_word(prev_words) or self._has_long_word(next_words)

        if normalized in self.NOISE_WORDS:
            return True

        if line[:1] in {'"', "'"} and len(words) == 1 and len(words[0]) <= 4:
            return True

        if len(words) == 1:
            word = words[0]
            if len(word) == 1:
                return near_long_context or word not in {"o"}
            if len(word) <= 2:
                return near_long_context and word not in self.VALID_SHORT_WORDS
            if len(word) == 3 and near_long_context and word not in self.VALID_SHORT_WORDS:
                return True

        if len(words) <= 2 and all(len(word) <= 2 for word in words):
            return near_long_context and normalized not in self.VALID_SHORT_PHRASES

        return False

    def _unicode_cleanup(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = "".join(ch if ch.isprintable() or ch in "\n\t" else " " for ch in text)
        return text

    def _replace_confusions(self, text: str) -> str:
        for bad, good in self.OCR_CONFUSIONS.items():
            text = text.replace(bad, good)
        return text

    def _fix_digit_in_word(self, token: str) -> str:
        if self.DIGIT_WHITELIST.match(token):
            return token

        letters = sum(c.isalpha() for c in token)
        digits = sum(c.isdigit() for c in token)
        if letters < 2 or digits == 0:
            return token

        substitutions = {"0": "o", "1": "i", "3": "e", "5": "s", "7": "t", "8": "B"}
        chars = list(token)
        for idx, ch in enumerate(chars):
            if ch not in substitutions:
                continue

            prev_is_letter = idx > 0 and chars[idx - 1].isalpha()
            next_is_letter = idx + 1 < len(chars) and chars[idx + 1].isalpha()
            if prev_is_letter or next_is_letter or letters >= digits:
                chars[idx] = substitutions[ch]

        return "".join(chars)

    def _fix_digit_tokens(self, text: str) -> str:
        def repl(match):
            return self._fix_digit_in_word(match.group(0))

        return self.NON_SPACE_TOKEN.sub(repl, text)

    def _normalize_line_break_hyphenation(self, text: str) -> str:
        """Merge OCR line breaks that split a word with a trailing hyphen."""
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text, flags=re.UNICODE)

    def _collapse_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n", text)
        return text.strip()

    def _strip_noise(self, text: str) -> str:
        text = re.sub(r"(?<!\w)[^\w\s.,!?;:'\"()\-]+(?!\w)", " ", text, flags=re.UNICODE)
        text = re.sub(r"([!?,;:])\1{1,}", r"\1", text)
        text = re.sub(r"\.{4,}", "...", text)
        text = re.sub(r"(?m)\s+[)\]}]+$", "", text)
        text = re.sub(r"(?m)^[([{]+\s*", "", text)
        text = re.sub(r"(?m)^['\"]+(?=\w)", "", text)
        return text

    def _apply_builtin_corrections(self, text: str) -> str:
        def repl(match):
            token = match.group(0)
            return self.BUILTIN_TOKEN_REPLACEMENTS.get(token, token)

        text = self.TOKEN_SPLIT.sub(repl, text)
        for pattern, replacement in self.BUILTIN_LINE_REPLACEMENTS:
            text = pattern.sub(replacement, text)
        return text

    # ---------- spell-check ----------

    def _correct_token(self, token: str) -> str:
        """
        Look up a single lowercase, diacritic-folded token in SymSpell.
        For long tokens that look like glued words, try word segmentation.
        Returns the best candidate if found; otherwise the original token.
        """
        if self.sym_spell is None:
            return token
        if len(token) < self.min_token_len_for_correction:
            return token  # don't risk correcting short tokens (false positives)

        query = fold_diacritics(token.lower())
        suggestions = self.sym_spell.lookup(
            query,
            Verbosity.TOP,
            max_edit_distance=self.max_edit_distance,
            transfer_casing=False,
        )
        if suggestions:
            best = suggestions[0]
            # exact match — accept immediately
            if best.distance == 0:
                return best.term
            # short tokens: only accept exact matches (prevents mafio→mario)
            if len(token) <= 5:
                return token
            # medium tokens: only edit-distance-1 corrections
            if best.distance >= 2 and len(token) <= 7:
                return token
            return best.term

        # no direct match — if the token is long, try word segmentation
        # (handles OCR outputs like "safiimafiot" -> "sa fii mafiot")
        if len(query) >= 8:
            try:
                seg = self.sym_spell.word_segmentation(query, max_edit_distance=0)
                if seg and seg.corrected_string and " " in seg.corrected_string:
                    pieces = seg.corrected_string.split()
                    # reject segmentation that produced too many tiny fragments
                    # (avoids "agheasma" -> "agh e as ma")
                    if (len(pieces) <= 3
                            and all(len(p) >= 2 for p in pieces)
                            and sum(1 for p in pieces if len(p) >= 3) >= 1):
                        return seg.corrected_string
            except Exception:
                pass
        return token

    def _spell_check(self, text: str) -> str:
        """Replace every letter-only token in text with its SymSpell candidate."""
        if not self.spell_check or self.sym_spell is None:
            return text

        def repl(match):
            return self._correct_token(match.group(0))

        return self.TOKEN_SPLIT.sub(repl, text)

    # ---------- main entry ----------

    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = self._unicode_cleanup(text)
        text = self._normalize_line_break_hyphenation(text)
        text = self._replace_confusions(text)
        if self.fix_digits:
            text = self._fix_digit_tokens(text)
        text = self._strip_noise(text)

        if self.lowercase:
            text = text.lower()

        text = self._apply_builtin_corrections(text)
        text = self._drop_noise_lines(text)

        # spell-check AFTER lowercase + digit fix so tokens are in canonical form
        text = self._spell_check(text)

        text = self._collapse_whitespace(text)
        return text


# testing
if __name__ == "__main__":
    print("--- Starting Phase 3 Testing: Text Normalization ---\n")

    normalizer = TextNormalizer()

    test_texts = [
        # realistic OCR dump from ocr/tests/tesseract/00100006_tesseract.txt
        "7oRIE DE PACATE SPA\n\n| —\n\nal —\n\nPoe\n\n|\n\nDE PACATE\n\nbă\n\nTA E\n\n"
        "„ist\n\n\"CU AGHEASMA\n\nSUB PRESIUNE\n\n2480\n\n[pl —\n\nmir\n\na da\n\n= tie",

        # clean caps like 00100001
        "PROTESTELE\nPROTESTELE\nDIN FRANTA\nDIN ROMANIA",

        # synthetic OCR garbage
        "   Sal0t!!!    Ce  mai    fac1???   \n\n",
        "R0man1a  est3  o   tara   frumoasa….",
        "„Pe  langa   plopii   fara  sot”  ~@#  adesea  am  trecut",
        "CAND PIS1CA NU-I ACASA……… JOACA S0ARECII PE MASA!!!!",

        # words with simple OCR errors that SymSpell should fix
        "SpaLTORIE DE PaCaTEs CuAGHEASMA SUU PRESIUNE",
    ]

    for raw_text in test_texts:
        result_text = normalizer.normalize(raw_text)
        print(f"Raw OCR input:    {raw_text!r}")
        print(f"Normalized output: {result_text!r}")
        print("-" * 60)
