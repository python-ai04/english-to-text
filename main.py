import io
import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


# Функция загрузки модели
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        _attn_implementation="eager",
    ).to(device)

    model.eval()
    return processor, model, device


# Функция загрузки изображения через Streamlit
def load_image():
    uploaded_file = st.file_uploader(
        "Выберите изображение для распознавания",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data, caption="Загруженное изображение", use_container_width=True)
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    return None

# Функция распознования
def transcribe_image(processor, model, device, image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Transcribe the text exactly. Do not paraphrase. Keep punctuation, numbers, and line breaks if possible."
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text.strip()


# Определение заголовков приложения
st.set_page_config(page_title="Распознать английский текст с изображения!", page_icon="🧠")
st.title("🧠 Распознать английский текст с изображения!")
st.write("Загрузите изображение и нажмите кнопку распознавания.")


processor, model, device = load_model()
img = load_image()

# Запуск распознования и вывод результата
if st.button("Распознать изображение", type="primary"):
    if img is None:
        st.warning("Сначала загрузите изображение.")
    else:
        with st.spinner("Распознавание текста..."):
            try:
                result = transcribe_image(processor, model, device, img)
                st.success("✅ Распознавание завершено!")
                st.markdown("**Результат:**")
                print(result)
                st.code(result, language="text")
            except Exception as e:
                st.error(f"Ошибка при распознавании: {e}")