import os
import random
import concurrent.futures
from typing import List
from urllib.parse import urlencode

import requests
from loguru import logger
from tqdm import tqdm
from PIL import Image
from io import BytesIO

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, VideoConcatMode
from app.utils import utils

requested_count = 0


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\nPlease set it in the config.toml file: {config.config_file}\n\n"
            f"{utils.to_json(config.app)}"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global requested_count
    requested_count += 1
    return api_keys[requested_count % len(api_keys)]


def search_images_pexels(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    orientation = aspect.name
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 20, "orientation": orientation}
    query_url = f"https://api.pexels.com/v1/search?{urlencode(params)}"
    logger.info(f"searching images: {query_url}, with proxies: {config.proxy}")

    try:
        r = requests.get(
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=False,
            timeout=(30, 60),
        )
        response = r.json()
        image_items = []
        if "photos" not in response:
            logger.error(f"search images failed: {response}")
            return image_items
        photos = response["photos"]
        # loop through each image in the result
        for photo in photos:
            item = MaterialInfo()
            item.provider = "pexels"
            # Use the largest image size available
            item.url = photo["src"]["original"]
            item.duration = 0  # Not applicable for images
            image_items.append(item)
        return image_items
    except Exception as e:
        logger.error(f"search images failed: {str(e)}")

    return []


def search_images_pixabay(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    orientation = "vertical" if aspect == VideoAspect.portrait else "horizontal"
    if aspect == VideoAspect.square:
        orientation = "horizontal"  # Pixabay doesn't have square orientation

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "image_type": "photo",
        "orientation": orientation,
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/?{urlencode(params)}"
    logger.info(f"searching images: {query_url}, with proxies: {config.proxy}")

    try:
        r = requests.get(
            query_url, proxies=config.proxy, verify=False, timeout=(30, 60)
        )
        response = r.json()
        image_items = []
        if "hits" not in response:
            logger.error(f"search images failed: {response}")
            return image_items
        photos = response["hits"]
        # loop through each image in the result
        for photo in photos:
            item = MaterialInfo()
            item.provider = "pixabay"
            item.url = photo["largeImageURL"]  # Use large image size
            item.duration = 0  # Not applicable for images
            image_items.append(item)
        return image_items
    except Exception as e:
        logger.error(f"search images failed: {str(e)}")

    return []


def save_image(image_url: str, save_dir: str = "") -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_images")
    
    # Directory is now created in the download_images function
    
    url_without_query = image_url.split("?")[0]
    url_hash = utils.md5(url_without_query)
    image_id = f"img-{url_hash}"
    image_path = f"{save_dir}/{image_id}.jpg"

    # if image already exists, return the path
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        logger.info(f"image already exists: {image_path}")
        return image_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # if image does not exist, download it
    try:
        response = requests.get(
            image_url,
            headers=headers,
            proxies=config.proxy,
            verify=False,
            timeout=(30, 60),
        )
        
        # Process the image to ensure it's valid and properly formatted
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Convert to RGB format
        img.save(image_path, format='JPEG', quality=95)
        
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            return image_path
    except Exception as e:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass
        logger.warning(f"invalid image file: {image_path} => {str(e)}")
    return ""


def download_images(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_contact_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    target_duration: float = 0.0,
    max_workers: int = 4,
) -> List[str]:
    valid_image_items = []
    valid_image_urls = []
    search_images = search_images_pexels
    if source == "pixabay":
        search_images = search_images_pixabay

    # First, gather all image items from search terms
    for search_term in search_terms:
        image_items = search_images(
            search_term=search_term,
            video_aspect=video_aspect,
        )
        logger.info(f"found {len(image_items)} images for '{search_term}'")

        for item in image_items:
            if item.url not in valid_image_urls:
                valid_image_items.append(item)
                valid_image_urls.append(item.url)

    logger.info(f"found total images: {len(valid_image_items)}")

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""
        
    # Create the download directory before starting downloads to avoid race conditions
    save_dir = material_directory or utils.storage_dir("cache_images")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Download directory prepared: {save_dir}")

    if video_contact_mode.value == VideoConcatMode.random.value:
        random.shuffle(valid_image_items)

    # Define the number of images to download - we'll use one image approximately every 3-4 seconds
    # with a minimum of 5 images and a maximum of 20 images
    duration_to_use = target_duration if target_duration > 0 else audio_duration
    num_images = max(5, min(20, int(duration_to_use / 4)))
    required_images = valid_image_items[:num_images]
    
    logger.info(f"Downloading {len(required_images)} images in parallel with {max_workers} workers")
    
    # Use ThreadPoolExecutor for parallel downloads
    image_paths = []
    valid_images_for_download = required_images
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a map of futures to image items for tracking
        future_to_item = {
            executor.submit(save_image, item.url, material_directory): item 
            for item in valid_images_for_download
        }
        
        # Process completed downloads as they finish
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(valid_images_for_download), desc="Downloading Images"):
            try:
                saved_image = future.result()
                if saved_image:
                    image_paths.append(saved_image)
            except Exception as e:
                item = future_to_item[future]
                logger.error(f"Failed to download image from URL {item.url}: {e}")
    
    logger.success(f"downloaded {len(image_paths)} images")
    return image_paths


if __name__ == "__main__":
    download_images(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )
