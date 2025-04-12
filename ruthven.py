from diffusers import StableDiffusionPipeline
import torch

import fitz #PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def generate_image_from_text(text, output_path):
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
    pipe = pipe.to(device)

    image = pipe(text).images[0]
    image.save(output_path)

if __name__ == "__main__":
    pdf_path = "C:\Users\mjoshi01\Downloads\complete_text_vampyre"
    output_image_path = "output_image.png"

    text = extract_text_from_pdf(pdf_path)

    prompt = "Generate an image of Lord Ruthven from this text: " + text

    generate_image_from_text(text, output_image_path)
