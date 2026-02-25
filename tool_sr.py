import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import glob
import torch
import argparse
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image
from torchvision import transforms
from tools.ram.models.ram_lora import ram
from tools.ose_diffusion import OSEDiff_test
from tools.ram import inference_ram as inference
from tools.osediff.wavelet_color_fix import adain_color_fix, wavelet_color_fix

sys.path.append(os.getcwd())
tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_validation_prompt(args, image, image_name, model, device='cuda:0'):
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    base_caption = captions[0].lower().strip()
    caption_set = set([x.strip() for x in base_caption.split(",")])
    filename = os.path.splitext(os.path.basename(image_name))[0].lower()
    last_token = filename.split("_")[-1]
    if last_token and last_token not in caption_set:
        full_caption = f"{base_caption}, {last_token}"
    else:
        full_caption = base_caption
    user_prompt = ", ".join(args.prompt) if isinstance(args.prompt, list) else args.prompt
    validation_prompt = f"{full_caption}, {user_prompt},"

    return validation_prompt, lq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', nargs='+')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--seed', default=42)
    parser.add_argument("--upscale", default=4)
    parser.add_argument("--process_size", default=512)
    parser.add_argument("--align_method", default='adain')
    parser.add_argument("--mixed_precision", default="fp16")
    parser.add_argument('--ram_ft_path', default='PLEASE INPUT THE MODEL_PATH.')
    parser.add_argument("--latent_tiled_size", default=96) 
    parser.add_argument("--latent_tiled_overlap", default=32)
    parser.add_argument("--osediff_path", default='PLEASE INPUT THE MODEL_PATH.')
    parser.add_argument("--vae_decoder_tiled_size", default=224) 
    parser.add_argument("--vae_encoder_tiled_size", default=1024) 
    parser.add_argument("--merge_and_unload_lora", default=False)
    parser.add_argument('--ram_path', default='PLEASE INPUT THE MODEL_PATH.')
    parser.add_argument('--pretrained_model_name_or_path', default='PLEASE INPUT THE MODEL_PATH.')
    args = parser.parse_args()

    model = OSEDiff_test(args)
    image_names = args.input_image

    DAPE = ram(pretrained=args.ram_path, pretrained_condition=args.ram_ft_path, image_size=384, vit='swin_l')
    DAPE.eval()
    device = torch.device("cuda:0") 
    DAPE.to(device)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    DAPE = DAPE.to(dtype=weight_dtype)
    sr_image_paths = []
    for image_name in image_names:
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False
        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)
        base, ext = os.path.splitext(bname)

        validation_prompt, lq = get_validation_prompt(args, input_image, image_name, DAPE)
        with torch.no_grad():
            lq = lq * 2 - 1
            output_image = model(lq, prompt=validation_prompt)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)
            if resize_flag:
                output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        save_dir = os.path.dirname(image_name)
        save_name = f"{base}_SR{ext}"
        save_path = os.path.join(save_dir, save_name)
        output_pil.save(save_path)
        sr_image_paths.append(save_path)
    print(sr_image_paths)
