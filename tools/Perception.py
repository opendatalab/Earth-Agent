import httpx
import random
import argparse

from pathlib import Path
from fastmcp import FastMCP


mcp = FastMCP('Perception')
parser = argparse.ArgumentParser()
parser.add_argument('--temp_dir', default='tmp/tmp', type=str)
parser.add_argument('--sam2_endpoint', type=str, action='append')
parser.add_argument('--remotesam_endpoint', type=str, action='append')
parser.add_argument('--instructsam_endpoint', type=str, action='append')
parser.add_argument('--remoteclip_endpoint', type=str, action='append')
parser.add_argument('--striprcnn_endpoint', type=str, action='append')
parser.add_argument('--sm3det_endpoint', type=str, action='append')
parser.add_argument('--changeos_endpoint', type=str, action='append')
parser.add_argument('--mscn_endpoint', type=str, action='append')
parser.add_argument('--port', default=20003, type=int)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 默认端口配置
DEFAULT_ENDPOINTS = {
    'sam2': 'http://0.0.0.0:16000',
    'remotesam': 'http://0.0.0.0:16001',
    'instructsam': 'http://0.0.0.0:16002',
    'remoteclip': 'http://0.0.0.0:16003',
    'striprcnn': 'http://0.0.0.0:16004',
    'sm3det': 'http://0.0.0.0:16005',
    'changeos': 'http://0.0.0.0:16006',
    'mscn': 'http://0.0.0.0:16007',
}

def parse_endpoints(endpoint_str, default_key=None):
    """Parse endpoint argument that can be string or list"""
    if endpoint_str is None:
        # 使用默认值
        if default_key and default_key in DEFAULT_ENDPOINTS:
            return [DEFAULT_ENDPOINTS[default_key]]
        return []
    if isinstance(endpoint_str, str):
        return [endpoint_str]
    elif endpoint_str:
        return endpoint_str
    return []

sam2_endpoints = parse_endpoints(args.sam2_endpoint, 'sam2')
remotesam_endpoints = parse_endpoints(args.remotesam_endpoint, 'remotesam')
instructsam_endpoints = parse_endpoints(args.instructsam_endpoint, 'instructsam')
remoteclip_endpoints = parse_endpoints(args.remoteclip_endpoint, 'remoteclip')
striprcnn_endpoints = parse_endpoints(args.striprcnn_endpoint, 'striprcnn')
sm3det_endpoints = parse_endpoints(args.sm3det_endpoint, 'sm3det')
changeos_endpoints = parse_endpoints(args.changeos_endpoint, 'changeos')
mscn_endpoints = parse_endpoints(args.mscn_endpoint, 'mscn')


@mcp.tool()
async def SAM2(input_image_path: str, bbox: list = None, output_path: str = None) -> dict:
    """
    Use SAM2 to segment the input image and return the bounding box.

    Parameters:
        input_image_path (str): Path to the input image.
        bbox (list): Bounding box of the segmented object.
        output_path (str): Path to save the segmented image.

    Returns:
        str: Path to the segmented image.
    """
    if not sam2_endpoints:
        return {"status": "error", "message": "SAM2 endpoint not configured"}

    endpoint = random.choice(sam2_endpoints)

    payload = {"image_path": input_image_path}
    if bbox is not None:
        payload["bbox"] = bbox
    if output_path is not None:
        payload["output_path"] = output_path

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{endpoint}/sam2/predict",
            json=payload,
        )
        resp.raise_for_status()

    return resp.json()


@mcp.tool()
async def RemoteSAM(input_image_path: str, text_prompt: str) -> dict:
    """
    RemoteSAM is a remote sensing visual grounding model. Given an input image and a text prompt
    describing a region of interest (e.g., "the football field located on the westernmost side"),
    it outputs the corresponding bounding box coordinates.

    Parameters:
        input_image_path (str): Path to the input image.
        text_prompt (str): Natural language description of the target object/region.

    Returns:
        list[int]: Bounding box [x_min, y_min, x_max, y_max]
    """
    if not remotesam_endpoints:
        return {"status": "error", "message": "RemoteSAM endpoint not configured"}

    endpoint = random.choice(remotesam_endpoints)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{endpoint}/remotesam/predict", 
            json={"image_path": input_image_path, "query_text": text_prompt},
        )
        resp.raise_for_status()

    return resp.json()


@mcp.tool()
async def InstructSAM(input_image_path: str, text_prompt: str) -> dict:
    """
    InstructSAM is an instruction-guided counting model for remote sensing images.
    Given an input image and a natural language prompt specifying the target object
    (e.g., "storage tank", "football field"), it detects and counts the number of
    instances matching the description.

    Parameters:
        input_image_path (str): Path to the input image.
        text_prompt (str): Natural language description of the object to count.

    Returns:
        int: The number of objects in the image that match the text prompt.
    """
    if not instructsam_endpoints:
        return {"status": "error", "message": "InstructSAM endpoint not configured"}

    endpoint = random.choice(instructsam_endpoints)

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{endpoint}/instructsam/predict",
            json={"image_path": input_image_path, "query_text": text_prompt},
        )
        resp.raise_for_status()

    return resp.json()


@mcp.tool()
async def RemoteCLIP(input_image_path: str) -> dict:
    """
    RemoteCLIP is a scene and land-use image classifier, specialized for categories such as
    Airport, Beach, Bridge, Commercial, Desert, Farmland, FootballField, Forest, Industrial,
    Meadow, Mountain, Park, Parking, Pond, Port, RailwayStation, Residential, River, and Viaduct.

    Parameters:
        input_image_path (str): Path to the input image.

    Returns:
        np.ndarray: [model_name, image_path, predicted_class, confidence, top-5 predictions]
    """
    if not remoteclip_endpoints:
        return {"status": "error", "message": "RemoteCLIP endpoint not configured"}

    endpoint = random.choice(remoteclip_endpoints)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{endpoint}/remoteclip/predict",
            json={"image_path": input_image_path},
        )
        resp.raise_for_status()

    return resp.json()


@mcp.tool()
async def Strip_R_CNN(input_image_path: str, text_prompt: str) -> dict:
    """
    Strip_R_CNN is a remote sensing object detection model with a strong focus on
    maritime and ship-related targets. Compared to SM3Det, it is particularly
    specialized in detecting and localizing different types of ships and naval vessels.

    Parameters:
        input_image_path (str): Path to the input image.
        text_prompt (str): Natural language description of the ship type to detect.

    Returns:
        list[list[float]]: A list of bounding boxes, each represented as
          [x_min, y_min, x_max, y_max].
    """
    if not striprcnn_endpoints:
        return {"status": "error", "message": "StripRCNN endpoint not configured"}

    endpoint = random.choice(striprcnn_endpoints)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{endpoint}/striprcnn/predict",
            json={"image_path": input_image_path, "text_prompt": text_prompt},
        )
        resp.raise_for_status()

    return resp.json()


@mcp.tool()
async def SM3Det(input_image_path: str, text_prompt: str) -> dict:
    """
    SM3Det is a remote sensing object detection model.
    Given an input image and a natural language prompt specifying the target object
    (e.g., "plane", "ship", "storage tank"), it detects all instances of that object
    and returns their bounding boxes.

    Parameters:
        input_image_path (str): Path to the input image.
        text_prompt (str): Natural language description of the object to detect.

    Returns:
        list[list[float]]: A list of bounding boxes, each represented as
          [x_min, y_min, x_max, y_max].
    """
    if not sm3det_endpoints:
        return {"status": "error", "message": "SM3Det endpoint not configured"}

    endpoint = random.choice(sm3det_endpoints)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{endpoint}/sm3det/predict",
            json={"image_path": input_image_path, "text_prompt": text_prompt},
        )
        resp.raise_for_status()

    return resp.json()


@mcp.tool()
async def ChangeOS(pre_image_path: str, post_image_path: str, output_path: str = None) -> dict:
    """
    Use ChangeOS to detect the change between two images and return the change mask.
    Can also be used to segment building by providing same image path in pre_image_path and post_image_path.

    Parameters:
        pre_image_path (str): Path to the pre-image.
        post_image_path (str): Path to the post-image.
        output_path (str): Path to the output change mask.

    Returns:
        str: Path to the segmented image.
    """
    if not changeos_endpoints:
        return {"status": "error", "message": "ChangeOS endpoint not configured"}

    endpoint = random.choice(changeos_endpoints)

    payload = {"pre_image_path": pre_image_path, "post_image_path": post_image_path}
    if output_path:
        payload["output_path"] = output_path

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{endpoint}/changeos/predict",
            json=payload,
        )
        resp.raise_for_status()

    return resp.json()


@mcp.tool()
async def MSCN(input_image_path: str) -> dict:
    """
    MSCN is a scene and land-use image classifier, effective for categories such as
    Airport, BareLand, BaseballField, Beach, Bridge, Center, Church, Commercial,
    DenseResidential, Desert, Farmland, Forest, Industrial, Meadow, MediumResidential,
    Mountain, Park, Parking, Playground, Pond, Port, RailwayStation, Resort, River,
    School, SparseResidential, Square, Stadium, StorageTanks, and Viaduct.

    Parameters:
        input_image_path (str): Path to the input image.

    Returns:
        np.ndarray: [model_name, image_path, predicted_class, confidence, top-5 predictions]
    """
    if not mscn_endpoints:
        return {"status": "error", "message": "MSCN endpoint not configured"}

    endpoint = random.choice(mscn_endpoints)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{endpoint}/mscn/predict",
            json={"image_path": input_image_path},
        )
        resp.raise_for_status()

    return resp.json()




@mcp.tool(description="""
Perform threshold-based segmentation on a single-band raster image.

The function reads a raster image from the specified path, converts it to a binary mask
by applying a fixed threshold, and writes the resulting binary image to a new file.
Pixel values greater than the threshold are set to 255 (white), and values less than or
equal to the threshold are set to 0 (black).

Parameters:
    input_image_path (str): Path to the input raster image file (e.g., TIFF, PNG, JPG).
    threshold (float or int): Pixel intensity threshold used to generate the binary mask.
    output_path (str): Relative output path (under TEMP_DIR) where the result will be saved,
                        e.g., "question17/threshold_segmentation_2022-01-16.tif".

Returns:
    str: Message indicating the file path where the result is saved.
""")
def threshold_segmentation(input_image_path, threshold, output_path):
    '''
    Perform threshold-based segmentation on a single-band raster image.

    The function reads a raster image from the specified path, converts it to a binary mask
    by applying a fixed threshold, and writes the resulting binary image to a new file.
    Pixel values greater than the threshold are set to 255 (white), and values less than or
    equal to the threshold are set to 0 (black).

    Parameters:
        input_image_path (str): Path to the input raster image file (e.g., TIFF, PNG, JPG).
        threshold (float or int): Pixel intensity threshold used to generate the binary mask.
        output_path (str): Relative output path (under TEMP_DIR) where the result will be saved,
                           e.g., "question17/threshold_segmentation_2022-01-16.tif".

    Returns:
        str: Message indicating the file path where the result is saved.
    '''
    try:
        import os
        import rasterio
        import numpy as np

        with rasterio.open(input_image_path) as src:
            image = src.read(1)
            meta = src.meta.copy()

        binary_image = (image > threshold).astype(np.uint8) * 255

        meta.update(dtype=rasterio.uint8, count=1)
        os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
        with rasterio.open(TEMP_DIR / output_path, 'w', **meta) as dst:
            dst.write(binary_image, 1)

        return f'Result save at {TEMP_DIR / output_path}'
    except Exception as e:

        return f'Error in threshold_segmentation: {str(e)}'


@mcp.tool(description="""
Expands bounding boxes by a given radius and returns the expanded bounding boxes.

Parameters:
    bboxes (list[list[float]]): List of bounding boxes, each represented as [x1, y1, x2, y2].
    radius (float): Expansion radius in the same unit as the GSD.
    gsd (float): Ground Sampling Distance in the same unit as the radius.

Returns:
    list[list[float]]: List of expanded bounding boxes, each represented as [x1, y1, x2, y2].
""")
def bbox_expansion(bboxes: list[list[float]], radius: float, gsd: float):
    """
    Expands bounding boxes by a given radius and returns the expanded bounding boxes.

    Parameters:
        bboxes (list[list[float]]): List of bounding boxes, each represented as [x1, y1, x2, y2].
        radius (float): Expansion radius in the same unit as the GSD.
        gsd (float): Ground Sampling Distance in the same unit as the radius.

    Returns:
        list[list[float]]: List of expanded bounding boxes, each represented as [x1, y1, x2, y2].
    """
    try:
        expanded_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1 = x1 - radius / gsd
            y1 = y1 - radius / gsd
            x2 = x2 + radius / gsd
            y2 = y2 + radius / gsd
            expanded_bboxes.append([x1, y1, x2, y2])

        return expanded_bboxes
    except Exception as e:

        return f'Error in bbox_expansion: {str(e)}'



@mcp.tool(description="""
    Description:
        Count the number of pixels in an image whose values are greater than 
        the specified threshold.

    Parameters:
        file_path (str):
            Path to the input image (GeoTIFF or raster format).
        threshold (float):
            Threshold value for hotspot detection.

    Returns:
        count (int):
            Number of pixels with values greater than the threshold.

    Example:
        >>> count_above_threshold("sample_image.tif", 100)
        2456
    """)
def count_above_threshold(file_path: str, threshold: float):
    """
    Description:
        Count the number of pixels in an image whose values are greater than 
        the specified threshold.

    Parameters:
        file_path (str):
            Path to the input image (GeoTIFF or raster format).
        threshold (float):
            Threshold value for hotspot detection.

    Returns:
        count (int):
            Number of pixels with values greater than the threshold.

    Example:
        >>> count_above_threshold("sample_image.tif", 100)
        2456
    """
    try:
        import numpy as np
        import rasterio
        with rasterio.open(file_path) as src:
            x = src.read(1)
        x = np.asarray(x)
        # Count elements greater than threshold
        count = np.sum(x > threshold)
    
        return int(count)
    except Exception as e:

        return f'Error in count_above_threshold: {str(e)}'



@mcp.tool(description=
    """
    Description:
        Read a binary image, apply erosion and skeletonization, 
        then count the number of external contours in the skeletonized image.

    Parameters:
        image_path (str):
            Path to the input binary (black and white) image.

    Returns:
        count (int):
            Number of external contours detected after skeletonization.

    Example:
        >>> count_connected_components("binary_mask.png")
        12
    """)
def count_connected_components(image_path):
    """
    Description:
        Read a binary image, apply erosion and skeletonization, 
        then count the number of external contours in the skeletonized image.

    Parameters:
        image_path (str):
            Path to the input binary (black and white) image.

    Returns:
        count (int):
            Number of external contours detected after skeletonization.

    Example:
        >>> count_connected_components("binary_mask.png")
        12
    """
    try:
        import cv2
        import numpy as np
        from skimage.morphology import skeletonize
        # Read image as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        # Binarize the image
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Apply erosion
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)

        # Skeletonize
        skeleton = skeletonize(eroded > 0)  # Convert to boolean for skimage
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return len(contours)
    except Exception as e:

        return f'Error in count_skeleton_contours: {str(e)}'



@mcp.tool(description=
    """
    Description:
        Convert bounding boxes from [x_min, y_min, x_max, y_max] format
        to centroid coordinates (x, y).

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, each defined as [x_min, y_min, x_max, y_max].

    Returns:
        centroids (list[tuple[float, float]]):
            A list of centroid coordinates, each in (x, y) format.

    Example:
        >>> bboxes2centroids([[0, 0, 10, 20], [5, 5, 15, 15]])
        [(5.0, 10.0), (10.0, 10.0)]
    """)
def bboxes2centroids(bboxes):
    """
    Description:
        Convert bounding boxes from [x_min, y_min, x_max, y_max] format
        to centroid coordinates (x, y).

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, each defined as [x_min, y_min, x_max, y_max].

    Returns:
        centroids (list[tuple[float, float]]):
            A list of centroid coordinates, each in (x, y) format.

    Example:
        >>> bboxes2centroids([[0, 0, 10, 20], [5, 5, 15, 15]])
        [(5.0, 10.0), (10.0, 10.0)]
    """
    try:
        return [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in bboxes]
    except Exception as e:

        return f'Error in bboxes2centroids: {str(e)}'



@mcp.tool(description=
    """
    Description:
        Compute pairwise distances between centroids and return both the closest 
        and farthest pairs with their indices and distances.

    Parameters:
        centroids (list[tuple[float, float]] or np.ndarray):
            A list or NumPy array of centroid coordinates in (x, y) format.

    Returns:
        result (dict):
            A dictionary containing:
              - 'min': (index1, index2, distance)
                  Indices of the closest centroid pair and their distance.
              - 'max': (index1, index2, distance)
                  Indices of the farthest centroid pair and their distance.

    Example:
        >>> centroids = [(0, 0), (3, 4), (10, 0)]
        >>> centroid_distance_extremes(centroids)
        {'min': (0, 1, 5.0), 'max': (1, 2, 7.211102550927978)}
    """)
def centroid_distance_extremes(centroids):
    """
    Description:
        Compute pairwise distances between centroids and return both the closest 
        and farthest pairs with their indices and distances.

    Parameters:
        centroids (list[tuple[float, float]] or np.ndarray):
            A list or NumPy array of centroid coordinates in (x, y) format.

    Returns:
        result (dict):
            A dictionary containing:
              - 'min': (index1, index2, distance)
                  Indices of the closest centroid pair and their distance.
              - 'max': (index1, index2, distance)
                  Indices of the farthest centroid pair and their distance.

    Example:
        >>> centroids = [(0, 0), (3, 4), (10, 0)]
        >>> centroid_distance_extremes(centroids)
        {'min': (0, 1, 5.0), 'max': (1, 2, 7.211102550927978)}
    """
    try:
        import numpy as np
        points = np.array(centroids)
        diff = points[:, None, :] - points[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        np.fill_diagonal(dist_matrix, np.inf)
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        min_dist = dist_matrix[min_idx]

        np.fill_diagonal(dist_matrix, -np.inf)
        max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        max_dist = dist_matrix[max_idx]

        return {
            "min": (int(min_idx[0]), int(min_idx[1]), float(min_dist)),
            "max": (int(max_idx[0]), int(max_idx[1]), float(max_dist))
        }
    except Exception as e:

        return f'Error in centroid_distance_extremes: {str(e)}'



@mcp.tool(description="""
    Description:
        Calculate the total area of a list of bounding boxes in [x, y, w, h] format.

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, where each box is defined as [x, y, w, h].
            - x, y → top-left corner coordinates
            - w, h → width and height of the box
        gsd (float, optional):
            Ground sample distance (meters per pixel). 
            - If provided, the result is in square meters (m²).
            - If None, the result is in square pixels (pixel²). Default = None.

    Returns:
        total_area (float):
            The total area of all bounding boxes, in m² if gsd is provided, otherwise in pixel².

    Example:
        >>> calculate_bbox_area([[0, 0, 10, 20], [5, 5, 15, 10]])
        350.0
        >>> calculate_bbox_area([[0, 0, 10, 20]], gsd=0.5)
        50.0
    """)
def calculate_bbox_area(bboxes, gsd=None):
    """
    Description:
        Calculate the total area of a list of bounding boxes in [x, y, w, h] format.

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, where each box is defined as [x, y, w, h].
            - x, y → top-left corner coordinates
            - w, h → width and height of the box
        gsd (float, optional):
            Ground sample distance (meters per pixel). 
            - If provided, the result is in square meters (m²).
            - If None, the result is in square pixels (pixel²). Default = None.

    Returns:
        total_area (float):
            The total area of all bounding boxes, in m² if gsd is provided, otherwise in pixel².

    Example:
        >>> calculate_bbox_area([[0, 0, 10, 20], [5, 5, 15, 10]])
        350.0
        >>> calculate_bbox_area([[0, 0, 10, 20]], gsd=0.5)
        50.0
    """
    try:
        total_area = 0.0
        for bbox in bboxes:
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox format: {bbox}. Expected [x, y, w, h].")
            _, _, w, h = bbox
            area = w * h
            total_area += area

        if gsd is not None:
            total_area *= gsd * gsd
    
        return total_area
    except Exception as e:

        return f'Error in calculate_bbox_area: {str(e)}'



if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=args.port)