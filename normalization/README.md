# Normalization

`normalization.py` has the logic for normalization.  
`normalize_all.py` reads files from `../ocr/tests/...` and writes the normalized results in `./tests/...`.  
`test_normalization.py` has a some simple tests for the main cases.  
`ro_50k_no_diacritics.txt` is the SymSpell dictionary used here.  
`ro_50k.txt` is just the original version with diacritics.

Run tests with:

```
python normalization\test_normalization.py
```

Regenerate normalized outputs with:

```
python normalization\normalize_all.py
```

If SymSpell is missing, install it with:

```
pip install symspellpy
```
