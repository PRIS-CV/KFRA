import os
import re
import json
import torch
import base64
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from openai import OpenAI
from PIL import Image as PILImage
from abc import ABC, abstractmethod
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


STOP_WORDS = {"is", "are", "find", "the", "segment", "all", "in", "image", 
              "how", "many", "there", "locate", "please"}
MAX_QUERY_WORDS = 2


class BaseVisionModel(ABC):
    @abstractmethod
    def process_single_image(self, image, instruction):
        pass
    
    @abstractmethod
    def process_batch(self, batch_images, batch_instructions):
        pass

class DetectionModel(ABC):
    """Interface for object detection tasks"""
    
    @abstractmethod
    def detect_objects(self, image, query):
        pass
    
    @abstractmethod
    def detect_objects_batch(self, images, queries):
        pass

class QAModel(ABC):
    @abstractmethod
    def answer_question(self, image, question):
        pass
    
    @abstractmethod
    def answer_questions_batch(self, images, questions):
        pass


class VisionReasonerModel(BaseVisionModel, DetectionModel, QAModel):
    def __init__(self, reasoning_model_path="/home/zhouzilu/FGVC-Agent/VisionReasoner-7B"):
        self.resize_size = 840
        self.reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoning_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:2",
        )
        self.reasoning_model.eval()
        self.processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")
        self.DETECTION_TEMPLATE = \
            "Please find \"{Question}\" with bboxs, points and labels." \
            "Compare the difference between object(s) and find the most closely matched object(s)." \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            "Output the bbox(es), point(s) and label(s) inside the interested object(s) in JSON format." \
            "For multiple distinct parts (e.g., head, wing), output **separate** bbox entries for each component, instead of merging them into a single region." \
            "i.e., <think> thinking process here </think>" \
            "<answer>{Answer}</answer>"
        self.QA_TEMPLATE = "{Question}"
        self.use_hybrid_mode = False
  
    def extract_bbox_points_think(self, output_text, x_factor, y_factor):
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
        pred_bboxes = []
        pred_points = []
        pred_labels = []
        pred_answer = None
        think_text = ""
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if not isinstance(data, list):
                    print(f"Error: Parsed JSON is not a list, but a {type(data)}.")
                    data = []
                pred_answer = []
                pred_bboxes = []
                pred_points = []
                pred_labels = []
                for item in data:
                    bbox_data = item.get('bbox_2d')
                    if not (bbox_data and isinstance(bbox_data, list) and len(bbox_data) == 4):
                        print(f"Warning: Skipping item due to missing or invalid 'bbox_2d': {item}")
                        continue
                    point_data = item.get('point_2d')
                    if point_data and isinstance(point_data, list) and len(point_data) == 2:
                        pred_points.append([
                            int(point_data[0] * x_factor + 0.5),
                            int(point_data[1] * y_factor + 0.5)
                        ])
                    else:
                        pred_points.append([0, 0])
                    pred_bboxes.append([
                        int(bbox_data[0] * x_factor + 0.5),
                        int(bbox_data[1] * y_factor + 0.5),
                        int(bbox_data[2] * x_factor + 0.5),
                        int(bbox_data[3] * y_factor + 0.5)
                    ])
                    pred_labels.append(item.get('label', 'unknown')) # 如果label不存在，默认为'unknown'
                    pred_answer.append(item)
            except Exception as e:
                print(f"Error processing JSON data: {e}")
        think_pattern = r'<think>([^<]+)</think>'
        think_match = re.search(think_pattern, output_text)
        if think_match:
            think_text = think_match.group(1)
        return pred_bboxes, pred_points, pred_labels, think_text, pred_answer
    
    def extract_qa_answer(self, output_text):
        think_pattern = r'<think>([^<]+)</think>'
        think_match = re.search(think_pattern, output_text)
        thinking = think_match.group(1) if think_match else ""
        clean_answer = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()
        return {
            "answer": clean_answer,
            "thinking": thinking,
            "full_response": output_text
        }
    
    def _generate_model_output(self, images, instructions, template, batch_mode=False):
        if not batch_mode:
            images = [images]
            instructions = [instructions]
        batch_messages = []
        scale_factors = []
        
        for image, instruction in zip(images, instructions):
            original_width, original_height = image.size
            x_factor, y_factor = original_width/self.resize_size, original_height/self.resize_size
            scale_factors.append((x_factor, y_factor))
            resized_image = image.resize((self.resize_size, self.resize_size), PILImage.BILINEAR)
            if "{Question}" in template:
                formatted_text = template.format(
                    Question=instruction.lower().strip(".\"?!"),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110], \"label\": \"head\"}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}, \"label\": \"tail\"]"
                )
            else:
                formatted_text = template
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": resized_image
                    },
                    {   
                        "type": "text",
                        "text": formatted_text
                    }
                ]
            }]
            batch_messages.append(message)
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:2")
        generated_ids = self.reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        if not batch_mode:
            return output_texts[0], scale_factors[0]
        return output_texts, scale_factors
    
    def process_single_image(self, image, instruction, return_task_type=False):
        task_type = self.task_router.route_task(instruction)
        if task_type == "detection":
            result = self.detect_objects(image, instruction)
        else:
            result = self.answer_question(image, instruction)
        
        if return_task_type:
            return result, task_type
        else:
            return result
    
    def process_batch(self, batch_images, batch_instructions):
        results = []
        for image, instruction in zip(batch_images, batch_instructions):
            result = self.process_single_image(image, instruction)
            results.append(result)
        return results
    
    def detect_objects(self, image, query):
        try:
            output_text, (x_factor, y_factor) = self._generate_model_output(
                image,
                query,
                self.DETECTION_TEMPLATE
            )
            bboxes, points, labels, thinking, pred_answer = self.extract_bbox_points_think(
                output_text, 
                x_factor, 
                y_factor
            )
            scores = [1.0] * len(bboxes)
            return {
                "bboxes": bboxes,
                "points": points,
                "labels": labels,
                "scores": scores,
                "thinking": thinking,
                "full_response": output_text,
                "pred_answer": pred_answer
            }
        except Exception as e:
            print(f"Error in detection: {e}")
            return {
                "bboxes": [],
                "points": [],
                "labels": [],
                "scores": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            }
    
    def detect_objects_batch(self, images, queries):
        try:
            output_texts, scale_factors = self._generate_model_output(
                images,
                queries,
                self.DETECTION_TEMPLATE,
                batch_mode=True
            )
            results = []
            for output_text, (x_factor, y_factor) in zip(output_texts, scale_factors):
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
                scores = [1.0] * len(bboxes)
                results.append({
                    "bboxes": bboxes,
                    "points": points,
                    "scores": scores,
                    "thinking": thinking,
                    "full_response": output_text,
                    "pred_answer": pred_answer
                })
            return results
        except Exception as e:
            print(f"Error in batch detection: {e}")
            return [{
                "bboxes": [],
                "points": [],
                "scores": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            } for _ in range(len(images))]

    def answer_question(self, image, question):
        try:
            output_text, _ = self._generate_model_output(
                image,
                question,
                self.QA_TEMPLATE
            )
            
            result = self.extract_qa_answer(output_text)
            return result
        except Exception as e:
            print(f"Error in QA: {e}")
            return {
                "answer": "",
                "thinking": "",
                "full_response": ""
            }
    
    def answer_questions_batch(self, images, questions):
        try:
            output_texts, _ = self._generate_model_output(
                images,
                questions,
                self.QA_TEMPLATE,
                batch_mode=True
            )
            
            results = []
            for output_text in output_texts:
                result = self.extract_qa_answer(output_text)
                results.append(result)
            return results
        except Exception as e:
            print(f"Error in batch QA: {e}")
            return [{
                "answer": "",
                "thinking": "",
                "full_response": ""
            } for _ in range(len(images))]
        
def visualize_results_enhanced(image, result, task_type, output_path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    if 'bboxes' in result and result['bboxes']:
        for bbox in result['bboxes']:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
    
    if 'points' in result and result['points']:
        for point in result['points']:
            plt.plot(point[0], point[1], 'go', markersize=8)  # Green point
    
    plt.title('Image with Bounding Boxes')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(image, alpha=0.6)
    if task_type == 'detection':
        if 'bboxes' in result and result['bboxes']:
            for bbox in result['bboxes']:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 fill=True, edgecolor='red', facecolor='red', alpha=0.3)
                plt.gca().add_patch(rect)
    task_title = {
        'detection': 'Detection Overlay',
        'segmentation': 'Segmentation Mask',
        'counting': 'Counting Results',
        'qa': 'Visual QA'
    }
    plt.title(task_title.get(task_type, 'Results Overlay'))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()