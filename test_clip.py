from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
import torch

model_name = "patrickjohncyh/fashion-clip"

model = CLIPModel.from_pretrained(model_name)
tokenizer = CLIPTokenizer.from_pretrained(model_name)
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded successfully on {device}")