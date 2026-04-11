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
        return "⚠️ Пожалуйста, загрузите изображение."

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

    return generated_text


# создаём интерфейс Gradio
demo = gr.Interface(
    fn=transcribe_image,
    inputs=gr.Image(type="pil", label="🖼️ Загрузите изображение"),
    outputs=gr.Textbox(label="📝 Распознанный текст", lines=10),
    title="🌟 Распознать английский текст с изображения!",
    description="Для распознавания загрузите изображение с английским текстом"
)

if __name__ == "__main__":
    demo.launch(share=True)