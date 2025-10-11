import argparse

from pathlib import Path
from fastmcp import FastMCP

from utils import read_image, read_image_uint8


mcp = FastMCP()
parser = argparse.ArgumentParser()
parser.add_argument('--temp_dir', type=str)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


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
    expanded_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 = x1 - radius / gsd
        y1 = y1 - radius / gsd
        x2 = x2 + radius / gsd
        y2 = y2 + radius / gsd
        expanded_bboxes.append([x1, y1, x2, y2])

    return expanded_bboxes



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
    import numpy as np
    import rasterio
    with rasterio.open(file_path) as src:
        x = src.read(1)
    x = np.asarray(x)
    # Count elements greater than threshold
    count = np.sum(x > threshold)
    
    return int(count)



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
        >>> count_skeleton_contours("binary_mask.png")
        12
    """)
def count_skeleton_contours(image_path):
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
        >>> count_skeleton_contours("binary_mask.png")
        12
    """
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
    return [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in bboxes]



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
   
def get_model_output(model_name: str, input_image_path: str, **args):
    import pandas as pd

    results = pd.read_csv('../model_results.csv')
    result = None
    try:
        # classification
        if model_name in ['MSCN', 'RemoteCLIP']:
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
        # detection
        elif model_name == 'Strip-R-CNN':
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
        # visual grounding
        elif model_name == 'RemoteSAM':
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
            result = result[args['text_prompt']]
        # counting
        elif model_name == 'InstructSAM':
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
            result = result[args['text_prompt']]
        # segmentation
        elif model_name == 'SAM2':
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
            result = result[args['bbox']]
    except:
        pass
    
    if result is None:
        return 'Failed to call model'
    else:
        return result



@mcp.tool(description="""
MSCN is a scene and land-use image classifier, effective for categories such as 
Airport, BareLand, BaseballField, Beach, Bridge, Center, Church, Commercial, 
DenseResidential, Desert, Farmland, Forest, Industrial, Meadow, MediumResidential, 
Mountain, Park, Parking, Playground, Pond, Port, RailwayStation, Resort, River, 
School, SparseResidential, Square, Stadium, StorageTanks, and Viaduct.

Parameters:
- input_image_path (str): Path to the input image.

Returns:
- np.ndarray: [model_name, image_path, predicted_class, confidence, top-5 predictions]

Example:
array([
  'MSCN',
  'benchmark/data/question189/J.jpg',
  'Resort',
  0.7052103281021118,
  [
    ('Resort', 0.7052103281021118),
    ('StorageTanks', 0.11459718644618988),
    ('Desert', 0.019159140065312386),
    ('Meadow', 0.013844668865203857),
    ('Beach', 0.013844599016010761)
  ]
], dtype=object)
""")
def MSCN(input_image_path):
    return get_model_output('MSCN', input_image_path)


@mcp.tool(description="""
RemoteCLIP is a scene and land-use image classifier, specialized for categories such as 
Airport, Beach, Bridge, Commercial, Desert, Farmland, FootballField, Forest, Industrial, 
Meadow, Mountain, Park, Parking, Pond, Port, RailwayStation, Residential, River, and Viaduct.

Parameters:
- input_image_path (str): Path to the input image.

Returns:
- np.ndarray: [model_name, image_path, predicted_class, confidence, top-5 predictions]

Example:
array([
  'RemoteCLIP',
  'benchmark/data/question189/J.jpg',
  'Resort',
  0.7052103281021118,
  [
    ('Resort', 0.7052103281021118),
    ('StorageTanks', 0.11459718644618988),
    ('Desert', 0.019159140065312386),
    ('Meadow', 0.013844668865203857),
    ('Beach', 0.013844599016010761)
  ]
], dtype=object)
""")
def RemoteCLIP(input_image_path):
    return get_model_output('RemoteCLIP', input_image_path)



@mcp.tool(description="""
Strip_R_CNN is a remote sensing object detection model with a strong focus on 
maritime and ship-related targets. Compared to SM3Det, it is particularly 
specialized in detecting and localizing different types of ships and naval vessels.

This model is highly effective at detecting the following categories:
- L3 ship
- L3 warcraft
- L3 merchant ship
- L3 aircraft carrier
- Arleigh Burke
- ContainerA
- Ticonderoga
- Perry
- Tarawa
- WhidbeyIsland
- CommanderA
- Austen
- Nimitz
- Sanantonio
- Container
- Car carrierB
- Enterprise
- Car carrierA
- Medical

Parameters:
- input_image_path (str): Path to the input image.
- text_prompt (str): Natural language description of the ship type to detect.

Returns:
- list[list[float]]: A list of bounding boxes, each represented as 
  [x_min, y_min, x_max, y_max].

Example:
Input:
  input_image_path = "benchmark/data/questionXXX/ship_example.png"
  text_prompt = "L3 aircraft carrier"

Output:
  [
    [120.5, 340.7, 480.2, 600.9],
    [700.3, 220.1, 950.6, 400.4]
  ]
""")
def Strip_R_CNN(input_image_path, text_prompt):
    return get_model_output('Strip-R-CNN', input_image_path, text_prompt=text_prompt)


@mcp.tool(description="""
SM3Det is a remote sensing object detection model. 
Given an input image and a natural language prompt specifying the target object 
(e.g., "plane", "ship", "storage tank"), it detects all instances of that object 
and returns their bounding boxes.

This model is particularly strong at detecting and localizing the following categories:
- plane
- ship
- storage tank
- baseball diamond
- tennis court
- basketball court
- ground track field
- harbor
- bridge
- large vehicle
- small vehicle
- helicopter
- roundabout
- soccer ball field
- swimming pool

Parameters:
- input_image_path (str): Path to the input image.
- text_prompt (str): Natural language description of the object to detect.

Returns:
- list[list[float]]: A list of bounding boxes, each represented as 
  [x_min, y_min, x_max, y_max].

Example:
Input:
  input_image_path = "benchmark/data/question235/P0173.png"
  text_prompt = "plane"

Output:
  [
    [491.08, 532.47, 562.03, 598.47],
    [548.89, 563.04, 636.54, 643.85],
    [57.80, 335.57, 191.65, 446.29],
    [401.37, 474.06, 509.87, 573.09],
    [344.69, 146.72, 464.14, 249.53],
    [736.10, 503.04, 809.28, 568.13],
    [680.84, 448.89, 760.03, 512.00],
    [588.72, 312.11, 666.23, 378.02],
    [537.49, 258.38, 610.34, 313.70]
  ]
""")
def SM3Det(input_image_path, text_prompt):
    return get_model_output('SM3Det', input_image_path, text_prompt=text_prompt)

@mcp.tool(description="""
RemoteSAM is a remote sensing visual grounding model. Given an input image and a text prompt 
describing a region of interest (e.g., "the football field located on the westernmost side"), 
it outputs the corresponding bounding box coordinates.

Parameters:
- input_image_path (str): Path to the input image.
- text_prompt (str): Natural language description of the target object/region.

Returns:
- list[int]: Bounding box [x_min, y_min, x_max, y_max]

Example:
Input:
  input_image_path = "benchmark/data/question226/478549_4934011_2048_32610_sport_soccer.jpg"
  text_prompt = "the football field located on the westernmost side"

Output:
  [0, 264, 127, 342]
""")
def RemoteSAM(input_image_path, text_prompt):
    return get_model_output('RemoteSAM', input_image_path, text_prompt=text_prompt)


@mcp.tool(description="""
InstructSAM is an instruction-guided counting model for remote sensing images. 
Given an input image and a natural language prompt specifying the target object 
(e.g., "storage tank", "football field"), it detects and counts the number of 
instances matching the description.

Parameters:
- input_image_path (str): Path to the input image.
- text_prompt (str): Natural language description of the object to count.

Returns:
- int: The number of objects in the image that match the text prompt.

Example:
Input:
  input_image_path = "benchmark/data/question231/B.jpg"
  text_prompt = "storage tank"

Output:
  8
""")
def InstructSAM(input_image_path, text_prompt):
    return get_model_output('InstructSAM', input_image_path, text_prompt=text_prompt)


@mcp.tool(description="""
    Use SAM2 to segment the input image and return the bounding box.

    Parameters:
        input_image_path (str): Path to the input image.
        bbox (list): Bounding box of the segmented object.

    Returns:
        str: Path to the segmented image.
""")
def SAM2(input_image_path, bbox, output_path):
    return get_model_output('SAM2', input_image_path, bbox=bbox, output_path=output_path)


@mcp.tool(description="""
    Use ChangeOS to detect the change between two images and return the change mask.
    Can also be used to segment building by providing same image path in pre_image_path and post_image_path.

    Parameters:
        pre_image_path (str): Path to the pre-image.
        post_image_path (str): Path to the post-image.
        output_path (str): Path to the output change mask.

    Returns:
        str: Path to the segmented image.
""")
def ChangeOS(pre_image_path: str, post_image_path: str, output_path: str):
    if pre_image_path == post_image_path:
        return get_model_output('ChangeOS_Building_Extraction', pre_image_path, output_path=output_path)
    else:
        return get_model_output('ChangeOS', pre_image_path, post_image_path=post_image_path, output_path=output_path)

if __name__ == "__main__":
    mcp.run()
