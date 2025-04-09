from flask import Flask, request, send_file
from diffusers import StableDiffusionXLPipeline
import torch
import uuid

app = Flask(__name__)

print("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "A futuristic city at sunset")
    image = pipe(prompt).images[0]
    filename = f"{uuid.uuid4()}.png"
    image.save(filename)
    return send_file(filename, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
