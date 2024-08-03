import os

import cv2
import json
import yaml
import time
import pytz
import datetime
import argparse
import shutil
import base64
import torch
import numpy as np
import requests
from litellm import completion
import litellm
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.nap import sleep
from litellm.caching import Cache
litellm.cache = Cache(type="disk")

openai.api_key = os.getenv("OPENAI_API_KEY")
litellm.set_verbose = True

from paddleocr import draw_ocr
from PIL import ImageChops, Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor

from modules.latex2png import tex2pil, zhtext2pil
from modules.extract_pdf import load_pdf_fitz
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from modules.self_modify import ModifiedPaddleOCR
from modules.post_process import get_croped_image, latex_rm_whitespace

from PIL import Image
from io import BytesIO
import base64

LATEX_STR_VALIDATION_PROMPT = "You are a LaTeX expert. Your job is to correct malformed latex. Only output the correct latex, even if the original is correct. Don't add any comments, just output the required latex string. Your response will directly be passed to the renderer, so if you add anything else you will fail.  Do not provide the open/close latex tags like \\(\\) \\[\\] or $,$$. Give the correct latex, nothing else."
LATEX_STR_VALIDATION_W_IMG_PROMPT = f"You are a LaTeX expert. Your job is to validate and correct (potentially malformed) latex, given an image of what the rendered latex SHOULD look like.You will be provided an image, and a latex string. Your job is to check whether the string will match the provided image when rendered, and if it doesn't, provide a corrected latex string. Only output the correct latex, even if the original is correct. Don't add any comments, just output the required latex string. Your response will directly be passed to the renderer, so if you add anything else you will fail. Do not provide the open/close latex tags like \\(\\) \\[\\] or $,$$. Give the correct latex, nothing else."
def mfd_model_init(weight):
    mfd_model = YOLO(weight)
    return mfd_model
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
def latex_to_image(latex_str, dpi=300):
    """
    Render a LaTeX string to a PNG image and return as a PIL Image object.

    Args:
        latex_str (str): The LaTeX string to render.
        dpi (int): Dots per inch (resolution) of the output image.

    Returns:
        PIL.Image.Image: The rendered image as a PIL Image object.
    """
    # Create a figure and axis with no frame or axis
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    ax.text(0.5, 0.5, f"${latex_str}$", fontsize=20, ha='center', va='center')
    ax.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Load the image from the buffer
    buf.seek(0)
    image = Image.open(buf)
    
    return image


def mfr_model_init(weight_dir, device='cpu'):
    args = argparse.Namespace(cfg_path="modules/UniMERNet/configs/demo.yaml", options=None)
    cfg = Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(device)
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    return model, vis_processor

def layout_model_init(weight):
    model = Layoutlmv3_Predictor(weight)
    return model


def pil_image_to_base64(image, format="PNG"):
    """
    Encode a PIL Image to a base64 string.

    Args:
        image (PIL.Image.Image): The PIL image to encode.
        format (str): The format to use for the image (e.g., "PNG", "JPEG").

    Returns:
        str: The base64 encoded string of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
def validate_and_correct_latex(latex, image):
    # First, try to render the LaTeX
    if isinstance(image, str):
        original_image = Image.open(image)
    elif isinstance(image, Image.Image):
        original_image = image
    encoded_image = pil_image_to_base64(original_image)
    try:
        rendered_image = latex_to_image(latex)
        
        # Compare the rendered image with the original image
        
        # Resize the rendered image to match the original image size
        rendered_image = rendered_image.resize(original_image.size)
        
        # Calculate the difference between the images
        diff = ImageChops.difference(original_image, rendered_image)
        
        # If the difference is significant, consider the LaTeX as potentially invalid
        if diff.getbbox() is not None and (diff.getbbox()[2] - diff.getbbox()[0]) * (diff.getbbox()[3] - diff.getbbox()[1]) > 0.1 * original_image.size[0] * original_image.size[1]:
            raise ValueError("Rendered image differs significantly from the original")
        
        return latex  # If no exception and images are similar, LaTeX is likely valid
    except:
        # If LaTeX is invalid or renders incorrectly, try to correct it using Mathpix API
        try:
            raise ValueError("LaTeX is invalid or renders incorrectly")
            response = requests.post(
                "https://api.mathpix.com/v3/text",
                json={"src": latex, "formats": ["latex_simplified"]},
                headers={
                    "app_id": "YOUR_MATHPIX_APP_ID",
                    "app_key": "YOUR_MATHPIX_APP_KEY",
                }
            )
            corrected_latex = response.json()["latex_simplified"]
            return corrected_latex
        except:
            # If Mathpix fails, use GPT-4 as a fallback
            try:                
                messages = [
                    {"role": "system", "content": LATEX_STR_VALIDATION_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": latex},
                    ]}
                ]
                
                response = completion(
                    model="gpt-4o-mini",
                    messages=messages
                )
                
                corrected_latex = response.choices[0].message.content
                print ("GPT-4o-mini response: ", corrected_latex)
                try:
                    latex_to_image(corrected_latex)
                    return corrected_latex
                except:
                    raise ValueError("GPT-4o-mini response is invalid")
            except:
                # If gpt-4o-mini fails, use gpt-4o with image
                try:
                    messages = [
                        {"role": "system", "content": LATEX_STR_VALIDATION_W_IMG_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": latex},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]}
                    ]
                    
                    response = completion(
                        model="gpt-4o",
                        messages=messages
                    )
                    
                    corrected_latex = response.choices[0].message.content
                    try:
                        latex_to_image(corrected_latex)
                        return corrected_latex
                    except:
                        raise ValueError("GPT-4o response is invalid")

                except:
                    # If all else fails, return the original LaTeX
                    return latex


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str)
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    print(args)
    
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(tz)
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Started!')
    
    ## ======== model init ========##
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    img_size = model_configs['model_args']['img_size']
    conf_thres = model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device = model_configs['model_args']['device']
    dpi = model_configs['model_args']['pdf_dpi']
    mfd_model = mfd_model_init(model_configs['model_args']['mfd_weight'])
    mfr_model, mfr_vis_processors = mfr_model_init(model_configs['model_args']['mfr_weight'], device=device)
    mfr_transform = transforms.Compose([mfr_vis_processors, ])
    layout_model = layout_model_init(model_configs['model_args']['layout_weight'])
    ocr_model = ModifiedPaddleOCR(show_log=True)
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Model init done!')
    ## ======== model init ========##
    
    start = time.time()
    if os.path.isdir(args.pdf):
        all_pdfs = [os.path.join(args.pdf, name) for name in os.listdir(args.pdf)]
    else:
        all_pdfs = [args.pdf]
    print("total files:", len(all_pdfs))
    for idx, single_pdf in enumerate(all_pdfs):
        try:
            img_list = load_pdf_fitz(single_pdf, dpi=dpi)
        except:
            img_list = None
            print("unexpected pdf file:", single_pdf)
        if img_list is None:
            continue
        print("pdf index:", idx, "pages:", len(img_list))
        # layout detection and formula detection
        doc_layout_result = []
        latex_filling_list = []
        mf_image_list = []
        for idx, image in enumerate(img_list):
            img_H, img_W = image.shape[0], image.shape[1]
            layout_res = layout_model(image, ignore_catids=[])
            mfd_res = mfd_model.predict(image, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0]
            for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    'category_id': 13 + int(cla.item()),
                    'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    'score': round(float(conf.item()), 2),
                    'latex': '',
                }
                layout_res['layout_dets'].append(new_item)
                latex_filling_list.append(new_item)
                bbox_img = get_croped_image(Image.fromarray(image), [xmin, ymin, xmax, ymax])
                mf_image_list.append(bbox_img)
                
            layout_res['page_info'] = dict(
                page_no = idx,
                height = img_H,
                width = img_W
            )
            doc_layout_result.append(layout_res)
            
        # Formula recognition, collect all formula images in whole pdf file, then batch infer them.
        a = time.time()  
        dataset = MathDataset(mf_image_list, transform=mfr_transform)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=32)
        mfr_res = []
        for imgs in dataloader:
            imgs = imgs.to(device)
            output = mfr_model.generate({'image': imgs})
            mfr_res.extend(output['pred_str'])
        for res, latex, img in zip(latex_filling_list, mfr_res, mf_image_list):
            latex = latex_rm_whitespace(latex)
            res['latex'] = validate_and_correct_latex(latex, img)
        b = time.time()
        print("formula nums:", len(mf_image_list), "mfr time:", round(b-a, 2))
            
        # ocr
        for idx, image in enumerate(img_list):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            single_page_res = doc_layout_result[idx]['layout_dets']
            single_page_mfdetrec_res = []
            for res in single_page_res:
                if int(res['category_id']) in [13, 14]:
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    single_page_mfdetrec_res.append({
                        "bbox": [xmin, ymin, xmax, ymax],
                    })
            for res in single_page_res:
                if int(res['category_id']) in [0, 1, 2, 4, 6, 7]:  #categories that need to do ocr
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    crop_box = [xmin, ymin, xmax, ymax]
                    cropped_img = Image.new('RGB', pil_img.size, 'white')
                    cropped_img.paste(pil_img.crop(crop_box), crop_box)
                    cropped_img = cv2.cvtColor(np.asarray(cropped_img), cv2.COLOR_RGB2BGR)
                    ocr_res = ocr_model.ocr(cropped_img, mfd_res=single_page_mfdetrec_res)[0]
                    if ocr_res:
                        for box_ocr_res in ocr_res:
                            p1, p2, p3, p4 = box_ocr_res[0]
                            text, score = box_ocr_res[1]
                            doc_layout_result[idx]['layout_dets'].append({
                                'category_id': 15,
                                'poly': p1 + p2 + p3 + p4,
                                'score': round(score, 2),
                                'text': text,
                            })

        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(single_pdf)[0:-4]
        with open(os.path.join(output_dir, f'{basename}.json'), 'w') as f:
            json.dump(doc_layout_result, f)
        
        if args.vis:
            color_palette = [
                (255,64,255),(255,255,0),(0,255,255),(255,215,135),(215,0,95),(100,0,48),(0,175,0),(95,0,95),(175,95,0),(95,95,0),
                (95,95,255),(95,175,135),(215,95,0),(0,0,255),(0,255,0),(255,0,0),(0,95,215),(0,0,0),(0,0,0),(0,0,0)
            ]
            id2names = ["title", "plain_text", "abandon", "figure", "figure_caption", "table", "table_caption", "table_footnote", 
                        "isolate_formula", "formula_caption", " ", " ", " ", "inline_formula", "isolated_formula", "ocr_text"]
            vis_pdf_result = []
            for idx, image in enumerate(img_list):
                single_page_res = doc_layout_result[idx]['layout_dets']
                vis_img = Image.new('RGB', Image.fromarray(image).size, 'white') if args.render else Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                draw = ImageDraw.Draw(vis_img)
                for res in single_page_res:
                    label = int(res['category_id'])
                    if label > 15:     # categories that do not need visualize
                        continue
                    label_name = id2names[label]
                    x_min, y_min = int(res['poly'][0]), int(res['poly'][1])
                    x_max, y_max = int(res['poly'][4]), int(res['poly'][5])
                    if args.render and label in [13, 14, 15]:
                        try:
                            if label in [13, 14]:  # render formula
                                window_img = tex2pil(res['latex'])[0]
                            else:
                                if True:           # render chinese
                                    window_img = zhtext2pil(res['text'])
                                else:              # render english
                                    window_img = tex2pil([res['text']], tex_type="text")[0]
                            ratio = min((x_max - x_min) / window_img.width, (y_max - y_min) / window_img.height) - 0.05
                            window_img = window_img.resize((int(window_img.width * ratio), int(window_img.height * ratio)))
                            vis_img.paste(window_img, (int(x_min + (x_max-x_min-window_img.width) / 2), int(y_min + (y_max-y_min-window_img.height) / 2)))
                        except Exception as e:
                            print(f"got exception on {text}, error info: {e}")
                    draw.rectangle([x_min, y_min, x_max, y_max], fill=None, outline=color_palette[label], width=1)
                    fontText = ImageFont.truetype("assets/fonts/simhei.ttf", 15, encoding="utf-8")
                    draw.text((x_min, y_min), label_name, color_palette[label], font=fontText)
                
                width, height = vis_img.size
                width, height = int(0.75*width), int(0.75*height)
                vis_img = vis_img.resize((width, height))
                vis_pdf_result.append(vis_img)
            
            first_page = vis_pdf_result.pop(0)
            first_page.save(os.path.join(output_dir, f'{basename}.pdf'), 'PDF', resolution=100, save_all=True, append_images=vis_pdf_result)
            try:
                shutil.rmtree('./temp')
            except:
                pass
            
    now = datetime.datetime.now(tz)
    end = time.time()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Finished! time cost:', int(end-start), 's')
