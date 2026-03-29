import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def ocr_with_smolvlm(model_name, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        _attn_implementation="eager",
    ).to(device)

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Transcribe the text exactly. Do not paraphrase."}
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

text = ocr_with_smolvlm("HuggingFaceTB/SmolVLM-256M-Instruct", "Text-eng-old.jpg")
print(text)