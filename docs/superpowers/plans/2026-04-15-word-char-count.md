# Word & Character Count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add word count, character count (with spaces), and character count (without spaces) outputs below the recognized text in the Gradio interface.

**Architecture:** Modify `transcribe_image` to return a 4-tuple instead of a plain string, and update the Gradio `Interface` to accept 4 output components. All logic uses standard Python string methods — no new dependencies.

**Tech Stack:** Python 3, Gradio, existing `main.py`

---

### Task 1: Update `transcribe_image` to return statistics

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update the early-return branch (None image)**

In `main.py`, find this line (~line 29):
```python
return "⚠️ Пожалуйста, загрузите изображение."
```
Replace with:
```python
return "⚠️ Пожалуйста, загрузите изображение.", 0, 0, 0
```

- [ ] **Step 2: Add statistics calculation and update the return value**

Find the final `return generated_text` (~line 66) and replace with:
```python
words = len(generated_text.split())
chars_with = len(generated_text)
chars_without = len(generated_text.replace(" ", ""))
return generated_text, words, chars_with, chars_without
```

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: return word and char counts from transcribe_image"
```

---

### Task 2: Update Gradio interface to display 4 outputs

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace single output with list of 4 Textbox components**

Find this block (~lines 70-76):
```python
demo = gr.Interface(
    fn=transcribe_image,
    inputs=gr.Image(type="pil", label="🖼️ Загрузите изображение"),
    outputs=gr.Textbox(label="📝 Распознанный текст", lines=10),
    title="🌟 Распознать английский текст с изображения!",
    description="Для распознавания загрузите изображение с английским текстом"
)
```
Replace with:
```python
demo = gr.Interface(
    fn=transcribe_image,
    inputs=gr.Image(type="pil", label="🖼️ Загрузите изображение"),
    outputs=[
        gr.Textbox(label="📝 Распознанный текст", lines=10),
        gr.Textbox(label="🔢 Количество слов"),
        gr.Textbox(label="🔡 Символов (с пробелами)"),
        gr.Textbox(label="🔡 Символов (без пробелов)"),
    ],
    title="🌟 Распознать английский текст с изображения!",
    description="Для распознавания загрузите изображение с английским текстом"
)
```

- [ ] **Step 2: Run the app and verify manually**

```bash
python3 main.py
```

Ожидаемое поведение:
- Загрузи любое изображение с текстом
- Под полем «Распознанный текст» появляются три новых поля: количество слов, символов с пробелами, символов без пробелов
- Если изображение не загружено — все поля возвращают предупреждение и `0`

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add word and char count outputs to Gradio interface"
```
