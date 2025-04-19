import os
import torch
import numpy as np
from PIL import Image
import cv2
from paddleocr import PaddleOCR
from transformers import AutoProcessor, AutoModelForQuestionAnswering

IMAGE_PATH = "./test_image_docvqa.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_ocr_from_image(image_path, ocr_engine):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = ocr_engine.ocr(image, cls=True)
    
    words, boxes = [], []
    if result[0] is not None:
        for line in result[0]:
            text = line[1][0]
            box = line[0]
            
            # Get bounding box coordinates
            x1, y1 = min(box[0][0], box[3][0]), min(box[0][1], box[1][1])
            x2, y2 = max(box[1][0], box[2][0]), max(box[2][1], box[3][1])
            
            words.append(text)
            boxes.append([x1, y1, x2, y2])
            
    return words, boxes, image

def normalize_boxes(boxes, width, height):
    normalized_boxes = []
    for x1, y1, x2, y2 in boxes:
        # Scale to 0-1000 range
        normalized_boxes.append([
            min(int(1000 * x1 / width), 1000),
            min(int(1000 * y1 / height), 1000),
            min(int(1000 * x2 / width), 1000),
            min(int(1000 * y2 / height), 1000)
        ])
    return normalized_boxes

def get_answer_text(start_idx, end_idx, tokens, processor):
    answer_tokens = tokens[start_idx:end_idx+1]
    answer_text = processor.tokenizer.convert_tokens_to_string(answer_tokens)
    # Clean up the answer
    return answer_text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()

def test_model(question, max_length=512):
    # Initialize OCR and model
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=DEVICE=='cuda')
    processor = AutoProcessor.from_pretrained("lakshya-rawat/document-qa-model")
    model = AutoModelForQuestionAnswering.from_pretrained("lakshya-rawat/document-qa-model")
    model.to(DEVICE)
    model.eval()
    
    # Process image and run OCR
    words, boxes, _ = extract_ocr_from_image(IMAGE_PATH, ocr_engine)
    pil_image = Image.open(IMAGE_PATH).convert("RGB")
    width, height = pil_image.size
    
    # Process inputs
    encoding = processor(
        images=pil_image,
        text=question,
        text_pair=words,
        boxes=normalize_boxes(boxes, width, height),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    
    # Move to device and get predictions
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Get answer span
    start_idx = torch.argmax(outputs.start_logits).cpu().item()
    end_idx = torch.argmax(outputs.end_logits).cpu().item()
    end_idx = max(start_idx, end_idx)  # Ensure end >= start
    
    # Extract answer text
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].cpu().squeeze().tolist())
    answer = get_answer_text(start_idx, end_idx, tokens, processor)
    
    return answer

if __name__ == "__main__":
    answer = test_model("what is the date mentioned in the document?")
    print(f"Answer: {answer}")
