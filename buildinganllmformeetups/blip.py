import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
import io

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

img = sys.argv[1]
raw_image = Image.open(img).convert('RGB')

inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
