import io
import re
import os
import ast
import math
import shutil
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from typing import List
from bs4 import BeautifulSoup
from newspaper import Article
from serpapi import GoogleSearch
from agents import function_tool
from fgvc_prompt import get_judge_prompt
from tools.vision_reasoner import VisionReasonerModel
from newspaper import Article, ArticleException, Config

_qwen_instance = None

def set_qwen(instance):
    global _qwen_instance
    _qwen_instance = instance

@function_tool
def search_similar_images(image_path: str):
    """
    A tool function that performs reverse image search using Google Lens (via SerpAPI)
    and leverages a vision-language model (Qwen) to infer a shortlist of the most plausible fine-grained categories for the main image.

    Args:
    image_path (str): The local path to the input image.

    Returns:
    str: The Qwen model's output, including a ranked list of fine-grained category candidates.

    How it works:
    1. The input image is uploaded to imgbb to generate a public URL, which is used to perform a reverse image search via SerpAPI's Google Lens engine.
    2. For each of the top visually similar images found online (up to 20), the function downloads the thumbnail, retrieves the page title, and extracts a short summary from the linked article if available (via newspaper3k).
    3. All images (main image first, followed by search results) are saved locally, and an information string is constructed in the format: "{filename}: {title}；{description}；". The main image always uses "null" for title and description.
    4. This information string and the ordered list of image paths are passed to a prompt generator (`get_judge_prompt`) to build a structured prompt.
    5. Both the images and the prompt are sent to the Qwen vision-language model, which is instructed to:
    - Focus on identifying the main image (always the first image in the list).
    - Compare it with retrieved visually similar images and accompanying descriptions.
    - Output a ranked shortlist (typically 1–3) of the most likely fine-grained categories, with reasoning and relevant search context.
    6. The model’s output is returned as the final result.

    Notes:
    - The imgbb and SerpAPI API keys must be valid and active.
    - The text field may be "null" if article parsing fails or no meaningful content is found.
    - The model only includes relevant and informative search results;    uninformative ones are excluded based on the prompt.
    - This function is suitable for fine-grained category candidate generation using multimodal reasoning and web context.

    Example use case:
    Given an image of a bird, use this tool to retrieve similar bird images online, extract relevant descriptions, and let the Qwen model suggest the most likely species candidates for further verification.
    """
    if _qwen_instance is None:
        error_msg = "[FATAL ERROR] Qwen instance is not set. Please call set_qwen(instance) first."
        print(error_msg)
        return f"Error: {error_msg}"
    IMGBB_API_KEY = "PLEASE INPUT THE API_KEY."
    SERPAPI_API_KEY = "PLEASE INPUT THE API_KEY."

    def upload_image_to_imgbb(image: Image.Image, timeout=20):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        res = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": IMGBB_API_KEY},
            files={"image": buffer},
            timeout=timeout
        )
        res.raise_for_status()
        return res.json()["data"]["url"]

    def extract_basic_info_single(match: dict):
        title, link, thumbnail = match.get("title"), match.get("link"), match.get("thumbnail")
        text = None 
        if link and isinstance(link, str):
            try:
                config = Config()
                config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
                config.request_timeout = 8
                config.fetch_images = False
                article = Article(link, config=config)
                article.download()
                article.parse()
                if article.text:
                    text = article.text.strip().replace("\n", " ")[:1500] 
            except Exception:
                pass

        return {"title": title, "link": link, "thumbnail": thumbnail, "text": text}

    def reverse_image_search_serpapi(image_url: str):
        params = {
            "engine": "google_lens", "url": image_url,
            "api_key": SERPAPI_API_KEY, "type": "visual_matches",
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        visual_matches = results.get("visual_matches", [])[:20]
        processed_matches = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_match = {executor.submit(extract_basic_info_single, m): m for m in visual_matches}
            for future in as_completed(future_to_match):
                match = future_to_match[future]
                try:
                    processed_matches.append(future.result(timeout=10))
                except (TimeoutError, Exception):
                    processed_matches.append({"title": match.get("title"), "link": match.get("link"), "thumbnail": match.get("thumbnail"), "text": None})
        return processed_matches

    def download_single_image(args):
        url, save_path, timeout = args
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            with open(save_path, "wb") as f: f.write(resp.content)
            return True
        except Exception:
            return False

    def download_and_build_info(results, main_image_path, save_dir="./search_imgs"):
        if os.path.exists(save_dir): shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        lines = [f"{os.path.basename(main_image_path)}: null；null；"]
        img_paths = [main_image_path]
        task_data, download_tasks = [], []
        for i, item in enumerate(results):
            url = item.get('thumbnail')
            if not url or not isinstance(url, str): continue
            ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
            if len(ext) > 5 or len(ext) < 3: ext = ".jpg"
            filename = f"search_{i+1}{ext}"
            path = os.path.join(save_dir, filename)
            task_data.append({"item": item, "filename": filename, "path": path})
            download_tasks.append((url, path, 8))
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_idx = {executor.submit(download_single_image, task): i for i, task in enumerate(download_tasks)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    if future.result(timeout=5):
                        data = task_data[idx]
                        title = (data["item"].get("title") or "null").strip().replace("\n", " ")
                        text = (data["item"].get("text") or "null").strip()
                        lines.append(f"{data['filename']}: {title}；{text}；")
                        img_paths.append(data['path'])
                except (TimeoutError, Exception):
                    pass         
        return img_paths, "\n".join(lines)

    print(f"[INFO] Called search_similar_images with image_path: '{image_path}'")
    try:
        image = Image.open(image_path).convert("RGB")
        uploaded_url = upload_image_to_imgbb(image)
        print(f"[INFO] Image uploaded to: {uploaded_url}")
        results = reverse_image_search_serpapi(uploaded_url)
        print(f"[INFO] Found {len(results)} visual matches, now processing them...")
        image_paths, info_str = download_and_build_info(results, image_path)
        print(f"[INFO] Prepared {len(image_paths)} images and info string for the model.")
        if len(image_paths) <= 1:
            print("[WARN] No valid similar images could be retrieved. The result may be less accurate.")
        inp = get_judge_prompt(info_str)
        caption = _qwen_instance(image_paths, inp)
        print(caption)
        return caption
    except Exception as e:
        print(f"[ERROR] search_similar_images failed: {e}")
        return f"Error executing search_similar_images: {str(e)}"
    

@function_tool
def get_wikipedia_description(query: str):
    """
    Fetches biological taxonomy and a descriptive summary for a given query from Wikipedia and Wikidata.

    This function searches for a given term on Wikipedia, retrieves its corresponding Wikidata entity,
    and traverses the taxonomic hierarchy (e.g., species, genus, family) to build a complete
    biological classification. It also parses and extracts the main descriptive text from the
    Wikipedia page, saving it to a local file.

    Args:
        query (str): The name of the biological entity to look up (e.g., "Lion", "Panthera leo").

    Returns:
        dict: A dictionary containing two keys:
              - 'taxonomy' (dict): A dictionary mapping taxonomic ranks (e.g., 'kingdom', 'phylum',
                'class', 'order', 'family', 'genus', 'species') to their respective names.
                Values may be None if not found.
              - 'description' (str): The local file path to the saved text file containing the
                summary from Wikipedia. Returns None if no description could be extracted.

    How it works:
    1.  It searches the English Wikipedia API to find the most relevant page for the given query.
    2.  From that page, it retrieves the corresponding Wikidata item ID.
    3.  Using the Wikidata API, it recursively follows the 'parent taxon' property (P171) to
        climb the biological classification tree.
    4.  At each level, it identifies the taxon's rank (P105) and name, populating the taxonomy dictionary.
    5.  It then parses the Wikipedia page's HTML to extract a clean text description, automatically
        stopping before common concluding sections like 'References' or 'See also'.
    6.  The extracted description is saved to a text file in a local 'temp' directory.
    7.  The final result is a dictionary containing the structured taxonomy and the path to the
        description file.

    Example use case:
        To get information about the domestic cat, you would call:
        get_wikipedia_description("Felis catus")
    """
    def _get_wikidata_entity(qid: str, session: requests.Session):
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return None
        except requests.exceptions.JSONDecodeError:
            return None
    print(f"[INFO] Called get_biological_info with query: '{query}'")
    taxonomy = {
        "kingdom": None, "phylum": None, "class": None, "order": None,
        "family": None, "genus": None, "species": None
    }
    description = None
    headers = {
        'User-Agent': 'MyCoolBot/1.0 (https://example.org/my-bot; myemail@example.com)'
    }
    session = requests.Session()
    session.headers.update(headers)
    api_url = "https://en.wikipedia.org/w/api.php"
    try:
        search_params = {
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "formatversion": 2
        }
        res = session.get(api_url, params=search_params, timeout=10)
        res.raise_for_status()
        search_results = res.json().get("query", {}).get("search", [])
        if not search_results:
            print(f"[INFO] No Wikipedia page found for query: '{query}'")
            return {"taxonomy": taxonomy, "description": None}
        page_title = search_results[0]["title"]
        print(f"[INFO] Found matching page: '{page_title}'")
        page_info_params = {
            "action": "query", "prop": "pageprops", "titles": page_title,
            "format": "json", "formatversion": 2
        }
        res = session.get(api_url, params=page_info_params, timeout=10)
        res.raise_for_status()
        pages = res.json().get("query", {}).get("pages", [])
        wikidata_id = None
        if pages and "pageprops" in pages[0]:
            wikidata_id = pages[0].get("pageprops", {}).get("wikibase_item")
        if wikidata_id:
            print(f"[INFO] Found Wikidata ID: {wikidata_id}")
            rank_map = {
                "Q36732": "kingdom", "Q38348": "phylum", "Q37517": "class",
                "Q36602": "order", "Q35409": "family", "Q34740": "genus", "Q7432": "species"
            }
            current_qid = wikidata_id
            visited = set()
            for i in range(10):
                if not current_qid or current_qid in visited:
                    break
                print(f"[DEBUG] Traversing taxon: {current_qid} (iteration {i+1})")
                visited.add(current_qid)
                data = _get_wikidata_entity(current_qid, session)
                if not data:
                    break
                entity = data.get("entities", {}).get(current_qid, {})
                claims = entity.get("claims", {})
                labels = entity.get("labels", {})
                
                rank_claims = claims.get("P105", [])
                if rank_claims:
                    rank_id = rank_claims[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                    name = labels.get("en", {}).get("value")
                    
                    if name and rank_id in rank_map:
                        taxonomic_level = rank_map[rank_id]
                        if not taxonomy[taxonomic_level]:
                             taxonomy[taxonomic_level] = name
                             print(f"[DEBUG] Found {taxonomic_level}: {name}")
                parent_claims = claims.get("P171", [])
                if parent_claims:
                    parent_value = parent_claims[0].get("mainsnak", {}).get("datavalue", {}).get("value", {})
                    current_qid = parent_value.get("id") if isinstance(parent_value, dict) else None
                else:
                    current_qid = None
        else:
            print(f"[INFO] Page '{page_title}' has no Wikidata item.")

        parse_params = {
            "action": "parse", "page": page_title, "prop": "text",
            "format": "json", "formatversion": 2
        }
        res = session.get(api_url, params=parse_params, timeout=15)
        res.raise_for_status()
        html = res.json().get("parse", {}).get("text", "")
        if html:
            soup = BeautifulSoup(html, "html.parser")
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            if content_div:
                text_blocks = []
                stop_sections = {'see also', 'references', 'external links', 'further reading', 'notes', 'sources'}
                for tag in content_div.children:
                    if not hasattr(tag, 'name'):
                        continue
                    if tag.name in ['h2', 'h3']:
                        section_text = tag.get_text(separator=' ', strip=True).lower().replace("[edit]", "")
                        if any(stop_word in section_text for stop_word in stop_sections):
                            break
                    if tag.name == 'p':
                        text = tag.get_text(separator=' ', strip=True)
                        text = re.sub(r'\s+([.,;?!:)\]}])', r'\1', text)
                        if text:
                            text_blocks.append(text)
                if text_blocks:
                    description = "\n\n".join(text_blocks)
                    print(f"[INFO] Successfully extracted description.")
    except requests.exceptions.RequestException as e:
        print(f"[FATAL ERROR] A network request failed during execution: {e}")
    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {e}")
    if description:
        try:
            output_dir = "temp"
            sanitized_query = re.sub(r'[\\/*?:"<>|]', "", query).replace(' ', '_')
            file_path = os.path.join(output_dir, f"{sanitized_query}.txt")
            os.makedirs(output_dir, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(description)
            print(f"[INFO] Description for '{query}' saved to: {file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save description to file: {e}")
    print(file_path)
    return {"taxonomy": taxonomy, "description": file_path}


@function_tool
def get_discriminative_region(image_path: str, query: str):
    """
    A tool function that identifies and extracts fine-grained semantic regions within an image based on a natural 
    language query (e.g., "head", "wing", "car headlight", "airplane tail") and returns cropped region images 
    along with their bounding boxes and semantic labels.

    Args:
        image_path (str): Path to the input image to be analyzed.
        query (str): A textual prompt describing the regions of interest 
                     (e.g., "bird's head and wing", "car headlights", "airplane wings").

    Returns:
        List[Dict]: A list of dictionaries, each containing:
            - 'path': Path to the saved cropped image region.
            - 'bbox': The [x1, y1, x2, y2] coordinates of the region in the original image.
            - 'label': The predicted semantic label of the region (e.g., "head", "wing", "wheel", "tail").

    Tool Description:
        This tool enables an agent to localize and isolate discriminative visual components from complex scenes 
        involving biological entities (e.g., birds, insects) as well as artificial structures (e.g., vehicles, aircraft). 
        It uses a vision-language reasoning model to identify relevant regions that match the input query and 
        produces a structured output for downstream use.

    Structure:
    - Image loading:
        Loads and preprocesses the image from the given path for analysis.
    - Region reasoning and detection:
        Applies `VisionReasonerModel.detect_objects()` to generate bounding boxes, key points, 
        and semantic labels aligned with the query prompt.
    - Cropping and saving:
        Each predicted bounding box is used to crop the original image, 
        and each region is saved with a structured filename pattern.
    - Output formatting:
        Collects region metadata including image path, bounding box coordinates, and labels.

    Notes:
    - This tool is domain-agnostic and works across natural and artificial object categories.
    - Supports identifying both anatomical parts (e.g., bird head, insect leg) and structural components 
      (e.g., car tire, airplane wing).
    - Cropped images are saved to: `temp/<base_name>_<idx>_region.jpg`.
    - Labels are inferred directly from the model’s semantic understanding.

    Example use cases:
    - For an image of a bird: detect and crop the "head" and "wing" regions.
    - For a car image: extract "front bumper" or "rear lights".
    - For an airplane: locate the "wing" and "tail section".
    """
    print(f"[INFO] Called get_discriminative_region with image_path: '{image_path}', query: '{query}'")
    model = VisionReasonerModel(reasoning_model_path="PLEASE INPUT THE MODEL_PATH.")
    image = Image.open(image_path).convert("RGB")
    result = model.detect_objects(image, query)
    bboxes = result.get("bboxes", [])
    labels = result.get("labels", [])

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    paths = []
    bboxes_list = []
    labels_list = []
    for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2) 
        crop_box = (x1, y1, x2, y2)
        cropped = image.crop(crop_box)
        w, h = cropped.size
        if w <= 0 or h <= 0:
            print(f"[WARN] Invalid crop size {w}x{h} for bbox {bbox}, skip.")
            continue
        if w < 28 or h < 28:
            scale = math.ceil(28 / min(w, h))
            new_w, new_h = max(28, w * scale), max(28, h * scale)
            cropped = cropped.resize((new_w, new_h), Image.BICUBIC)
            print(f"[INFO] Small crop detected ({w}x{h}), scaled by x{scale} -> ({new_w}x{new_h})")
        safe_label = str(label).replace("/", "_")
        save_path = f"temp/{base_name}_{idx}_{safe_label}.jpg"
        cropped.save(save_path)
        paths.append(save_path)
        bboxes_list.append([x1, y1, x2, y2])
        labels_list.append(label)

    return paths, bboxes_list, labels_list


@function_tool
def get_super_resolution(image_paths: List[str]):
    """
    A tool function that enhances the resolution and visual quality of low-resolution or degraded images 
    using a diffusion-based super-resolution model. It returns file paths to the corresponding enhanced images.

    Args:
        image_paths (List[str]): A list of local image file paths that require super-resolution processing.

    Returns:
        List[str]: A list of file paths pointing to the saved super-resolved images.

    Tool Description:
        This tool improves the visual quality of low-resolution or blurry images using a diffusion-based 
        super-resolution pipeline. It is typically invoked under two circumstances:

        1. **After Region Cropping**: When the agent identifies and crops specific semantic parts (e.g., head, tail, wing) 
           of an object (like a bird), the resulting cropped patches may suffer from reduced resolution or detail loss. 
           This tool is used to restore clarity in such fine-grained regions.

        2. **For Low-Quality Inputs**: If the original input image itself is of poor resolution or quality 
           (e.g., compressed, distant, or small object in the frame), this tool is also applied directly 
           to enhance the entire image before any reasoning or analysis.

    Structure:
    - Input handling:
        Accepts one or more image paths and calls an external `test_osediff.py` script within the `OSEDiff` 
        conda environment using `subprocess`.

    - Super-resolution processing:
        The script uses a diffusion-based model to upscale and refine the images, applying optional alignment 
        techniques such as AdaIN or wavelet-based color correction.

    - Output retrieval:
        Parses the enhanced image paths returned via standard output using `ast.literal_eval`.

    - Error handling:
        Returns an empty list if execution fails or if the output is malformed. Logs are printed for debugging.

    Notes:
    - Output images are saved with a `_SR` suffix in the same directory as the input images.
    - This tool helps ensure both global and local image clarity, enabling more accurate visual understanding 
      and downstream tasks such as fine-grained classification, captioning, or retrieval.

    Example use cases:
    - Enhance cropped regions like a bird’s head or tail before feature extraction.
    - Improve the clarity of a blurry input bird image captured at low resolution.
    - Use as preprocessing before sending the image for reverse search, description generation, or discrimination reasoning.
    """
    print(f"[INFO] Called get_super_resolution with image_path: '{image_paths}'")
    command = [
        "conda", "run", "-n", "OSEDiff", "python",
        "tool_sr.py",
        "--input_image", *image_paths,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        sr_image_paths = ast.literal_eval(result.stdout.strip())
    except Exception as e:
        sr_image_paths = []
    return sr_image_paths


@function_tool
def get_image_info(image_paths: List[str]):
    """
    A tool function that receives a list of image paths and returns their resolution information.

    Args:
        image_paths (List[str]): A list of image file paths to be analyzed.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
            - 'path': The original image path.
            - 'width': Width of the image in pixels.
            - 'height': Height of the image in pixels.

    Tool Description:
        This utility tool allows the agent to inspect the resolution of one or more images.
        It is especially useful for determining whether an image meets the resolution requirement
        for downstream processing such as fine-grained recognition or super-resolution.

    Notes:
    - If any image file cannot be opened, it will raise an exception.
    - Useful for LLM agents that need to check whether image resolution is below a threshold.

    Example use case:
    - Given a list of image paths, return:
        [{"path": "bird1.jpg", "width": 180, "height": 150},
         {"path": "car2.jpg", "width": 512, "height": 512}]
    """
    print(f"[INFO] Called get_image_info with image_path: '{image_paths}'")
    output = []
    for image_path in image_paths:
        img = Image.open(image_path)
        width, height = img.size
        output.append({
            "path": image_path,
            "width": width,
            "height": height
        })
    return output