import os
import tempfile

import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

_model = None
_processor = None


# Функция загрузки модели
def load_model():
    global _model, _processor
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Загрузка модели на устройство: {device}")

        _processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        _model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        _model = _model.to(device)
        _model.eval()
    return _processor, _model


def transcribe_image(image):
    if image is None:
        return (
            "⚠️ Пожалуйста, загрузите изображение.",
            0, 0, 0,
            gr.DownloadButton(visible=False),
        )

    processor, model = load_model()

    # задаем промт
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "Transcribe all the text you can recognize in the image."
                    "Output only the recognized text."
                )
            }
        ]
    }]

    # применяем промт и задаем входные данные
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_length:]
    generated_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    words = len(generated_text.split())
    chars_with = len(generated_text)
    chars_without = len(generated_text.replace(" ", ""))

    tmp_path = os.path.join(tempfile.gettempdir(), "recognized_text.txt")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(generated_text)

    return (
        generated_text,
        words,
        chars_with,
        chars_without,
        gr.DownloadButton(value=tmp_path, visible=True),
    )


ABOUT_MD = """
## 📚 О проекте

Это **учебный проект** — веб-приложение для распознавания английского текста
с изображений.

| | |
|---|---|
| **Программа обучения** | [Профессиональная переподготовка УрФУ](https://dpo.urfu.ru/programs/92) |
| **Модель распознавания** | [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) |
| **Команда** | См. [README](README.md) |
"""


# создаём интерфейс Gradio
theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="fuchsia",
    neutral_hue="slate",
).set(
    body_background_fill="linear-gradient(135deg, #f5f3ff 0%, #fdf4ff 100%)",
    button_primary_background_fill="linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #7c3aed 0%, #c026d3 100%)",
    button_primary_text_color="white",
    button_primary_border_color="*primary_500",
    button_primary_shadow="0 0 20px rgba(139, 92, 246, 0.35)",
    button_primary_shadow_hover="0 0 28px rgba(217, 70, 239, 0.5)",
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="*primary_50",
    button_secondary_border_color="*primary_300",
    button_secondary_text_color="*primary_700",
    block_border_color="*primary_200",
    block_shadow="0 2px 12px rgba(139, 92, 246, 0.08)",
    input_border_color="*primary_200",
    input_border_color_focus="*primary_500",
)

with gr.Blocks(title="Распознавание английского текста") as demo:
    gr.Markdown("# 🌟 Распознать английский текст с изображения")
    gr.Markdown("Загрузите изображение с английским текстом — модель распознает его.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="🖼️ Изображение")
            with gr.Row():
                clear_btn = gr.ClearButton(value="Очистить")
                submit_btn = gr.Button("Распознать", variant="primary")
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="📝 Распознанный текст", lines=10)
            download_btn = gr.DownloadButton(
                "💾 Скачать текст (.txt)",
                visible=False,
            )
            with gr.Row():
                words_output = gr.Textbox(label="🔢 Слов")
                chars_with_output = gr.Textbox(label="🔡 С пробелами")
                chars_without_output = gr.Textbox(label="🔡 Без пробелов")

    submit_btn.click(
        fn=transcribe_image,
        inputs=image_input,
        outputs=[text_output, words_output, chars_with_output, chars_without_output, download_btn],
    )
    clear_btn.add([image_input, text_output, words_output, chars_with_output, chars_without_output, download_btn])

    gr.Markdown(ABOUT_MD)

if __name__ == "__main__":
    demo.launch(share=True, theme=theme)