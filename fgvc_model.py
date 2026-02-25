import io
import os
import re
import copy
import json
import torch
import base64
import random
import string
import asyncio
import dashscope
import numpy as np
from pathlib import Path
from http import HTTPStatus
from openai import AsyncOpenAI
from tools.fgvc_tools import *
from typing import List, Union
from transformers.image_utils import load_images
from json_repair import repair_json, loads
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    set_default_openai_client,
)
from transformers import AutoProcessor, AutoTokenizer, AutoModelForZeroShotObjectDetection
from fgvc_prompt import COCO_CLASSES, get_caption_prompt, get_infer_prompt, get_search_prompt, get_agent_result, get_correlate_prompt

client = AsyncOpenAI(
    base_url="PLEASE INPUT THE BASE_URL.",
    api_key="PLEASE INPUT THE API_KEY.",
)
set_tracing_disabled(disabled=True)

class QwenAPI:
    def __init__(self, model_name: str = "PLEASE INPUT THE MODEL_NAME."):
        self.model_name = model_name
    def image_to_base64(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
        except Exception as e:
            return None

    def __call__(self, image_paths: List[str], text: str) -> str:
        messages = [{'role': 'user', 'content': []}]
        for path in image_paths:
            base64_data = self.image_to_base64(path)
            if base64_data:
                messages[0]['content'].append({'image': base64_data})
        messages[0]['content'].append({'text': text})
        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model_name,
                messages=messages
            )
            if response.status_code == HTTPStatus.OK:
                content_list = response.output.choices[0]['message']['content']
                raw_answer = ""
                if isinstance(content_list, list):
                    for part in content_list:
                        if 'text' in part:
                            raw_answer += part['text']
                else:
                    raw_answer = str(content_list)
                
                raw_answer = raw_answer.strip()
                return raw_answer
            else:
                return "Null_API_Error"
        except Exception as e:
            return "Null_Exception"

class GroundingDino:
    def __init__(
        self,
        model_id: str = "PLEASE INPUT THE MODEL_ID.",
        device: str = "cuda",
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.default_classes = COCO_CLASSES

    def __call__(
        self,
        in_image: Image.Image,
        classes: Union[List[str], None] = None,
    ):
        if in_image.mode == 'RGBA':
            in_image = in_image.convert('RGB')
        width, height = in_image.size

        if classes is None:
            classes = self.default_classes
        text = ". ".join(classes)
        inputs = self.processor(images=in_image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold = self.box_threshold,
            text_threshold = self.text_threshold,
            target_sizes=[in_image.size[::-1]]
        )
        bboxes = results[0]['boxes'].cpu().numpy()
        labels = results[0]['labels']
        if len(bboxes) == 0:
            return [], [], []
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        sorted_idx = np.argsort(areas)[::-1]
        cropped_images = []
        for idx in sorted_idx:
            x0, y0, x1, y1 = bboxes[idx]
            x0 = int(np.clip(np.floor(x0), 0, width-1))
            y0 = int(np.clip(np.floor(y0), 0, height-1))
            x1 = int(np.clip(np.ceil(x1), 0, width))
            y1 = int(np.clip(np.ceil(y1), 0, height))
            crop = in_image.crop((x0, y0, x1, y1))
            cropped_images.append(crop)
        topk_labels = [labels[i] for i in sorted_idx]
        topk_bboxes = [bboxes[i] for i in sorted_idx]
        return topk_bboxes, topk_labels, cropped_images

class Qwen:
    def __init__(
        self,
        model_id: str = "PLEASE INPUT THE MODEL_ID.",
        device: str = "cuda:1",
        temperature: float = 0.2,
        max_new_tokens: int = 2048,
    ):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
    
    def __call__(
        self,
        image_paths: List[str],
        text: str,
    ):
        pil_images = [Image.open(path).convert("RGB") for path in image_paths]
        content = []
        for _ in pil_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        messages = [{
            "role": "user",
            "content": content
        }]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=pil_images, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                temperature = self.temperature,
                do_sample = True if self.temperature > 0 else False,
                max_new_tokens = self.max_new_tokens,
                streamer = None,
                use_cache = True
            )
            output_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            outputs = self.processor.batch_decode(
                output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return outputs[0]


class FGVCAgent():
    def __init__(
        self,
        ground_model: str = "PLEASE INPUT THE MODEL_ID.",
        ground_device: str = "cuda"
    ):
        self.grounder = GroundingDino(
            model_id = ground_model,
            device = ground_device,
        )
        self.vlm = None

    async def __call__(
        self,
        image: Union[str, Image.Image, np.ndarray],
        text: str,
        ground_classes: Union[List[str], None] = None
    ):
        if not os.access('temp', os.F_OK):
            os.makedirs('temp')
        for file in os.listdir('temp'):
            os.remove(os.path.join('temp', file))
        with open('temp/text.txt', 'w', encoding='utf-8') as wf:
            wf.write(text)
        if isinstance(image, str):
            in_image = Image.open(image)
        elif isinstance(image, Image.Image):
            in_image = image
        elif isinstance(image, np.ndarray):
            in_image = Image.fromarray(image.astype(np.uint8))
        else:
            raise Exception('Unsupported input image format.')

        bboxes, labels, out_image = self.grounder(in_image, classes=ground_classes)
        det_images, image_paths = [], []
        width, height = in_image.size
        if width < 100 or height < 100:
            save_path = 'temp/bbox_image_0.jpg'
            shutil.copy(image, save_path)
            img_to_use = in_image.convert('RGB') if in_image.mode == 'RGBA' else in_image
            det_images.append(img_to_use)
            labels = ['image']
            image_paths.append(save_path)
        else:
            for bid, bbox in enumerate(bboxes):
                crop_box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                det_image = in_image.crop(crop_box)
                save_path = 'temp/bbox_image_{}_{}.jpg'.format(bid, labels[bid])
                if det_image.mode == 'RGBA':
                    det_image = det_image.convert('RGB')
                det_image.save(save_path)
                det_images.append(det_image)
                image_paths.append(save_path)
            if len(det_images) == 0:
                save_path = 'temp/bbox_image_0.jpg'
                img_to_save = in_image.convert('RGB') if in_image.mode == 'RGBA' else in_image
                img_to_save.save(save_path)
                det_images.append(img_to_save)
                labels.append('image')
                image_paths.append(save_path)

        captions = []
        final_images = [image]
        agent = Agent(name="Fine-Grained Reasoning Assistant", 
                      tools=[search_similar_images, get_wikipedia_description, get_discriminative_region, get_super_resolution, get_image_info],
                      model=OpenAIChatCompletionsModel(model="PLEASE INPUT THE MODEL_NAME.", openai_client=client))
        set_qwen(self.vlm)
        for det_image, label, path in zip(det_images, labels, image_paths):
            final_images.append(path)
            prompt = get_search_prompt(path, text)
            agent_result = await Runner.run(agent, prompt, max_turns=50)
            repair_data = repair_json(agent_result.final_output)
            output_json = json.loads(repair_data)
            agent_str, paths, part_labels = get_agent_result(output_json, path)
            if len(det_images) == 1:
                inp = get_caption_prompt(label, text, agent_str, part_labels)
                paths.append(image)
                caption = self.vlm(paths, inp)
                return caption
            if len(det_images) >= 2:
                inp = get_infer_prompt(label, text, agent_str, part_labels)
                caption = self.vlm(paths, inp)
                print(caption)
                captions.append(caption)
        if len(captions) >= 2:
            caption_block = ""
            for idx, caption in enumerate(captions):
                caption_block += f"[Region {idx}]\n{caption.strip()}\n\n"
            final_prompt = get_correlate_prompt(caption_block, text)
            final_answer = self.vlm(final_images, final_prompt)
            with open('temp/final_answer.txt', 'w', encoding='utf-8') as wf:
                wf.write(final_answer)
            return final_answer

async def main():
    image_path = "PLEASE INPUT THE IMAGE_PATH."
    question = "PLEASE INPUT THE QUESTION."

    qwen_instance = Qwen(
        model_id="PLEASE INPUT THE MODEL_ID.",
        device="cuda:3"
    )
    agent = FGVCAgent(
        ground_model="PLEASE INPUT THE MODEL_ID.",
        ground_device="cuda"
    )
    agent.vlm = qwen_instance

    answer = await agent(image=image_path, text=question)

    print("-" * 50)
    print(f"Image: {image_path}")
    print(f"Question: {question}")
    print("\n=== Model Answer ===")
    print(answer)
    print("-" * 50)

if __name__ == '__main__':
    asyncio.run(main())