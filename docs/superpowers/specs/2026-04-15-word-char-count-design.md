# Design: Word & Character Count Feature

**Date:** 2026-04-15  
**Project:** english-to-text  
**Status:** Approved

## Overview

Add word count and character count (with and without spaces) statistics directly below the recognized text output in the Gradio interface.

## Changes

### `transcribe_image` function

Return a tuple of 4 values instead of a single string:

1. Recognized text (`str`)
2. Word count (`int`) — `len(text.split())`
3. Character count with spaces (`int`) — `len(text)`
4. Character count without spaces (`int`) — `len(text.replace(" ", ""))`

Edge case: if image is `None`, return empty/zero values for all four outputs.

### Gradio Interface

Replace the single `outputs=gr.Textbox(...)` with a list of 4 components:

```python
outputs=[
    gr.Textbox(label="Распознанный текст", lines=10),
    gr.Textbox(label="Количество слов"),
    gr.Textbox(label="Символов (с пробелами)"),
    gr.Textbox(label="Символов (без пробелов)"),
]
```

## Scope

- Only `main.py` is modified — no new files.
- No external libraries needed — standard Python string methods only.
