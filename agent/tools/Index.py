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


def calculate_ndvi(input_nir_path, input_red_path, output_path):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        input_red_path (str): Path to the Red band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/ndvi_2022-01-16.tif"

    Returns:
        str: Path to the saved NDVI file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the NIR and Red band raster files
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)
        nir_profile = nir_src.profile  # Get the metadata profile

    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    nir_band = np.array(nir_band, dtype=np.float32)
    red_band = np.array(red_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = nir_band + red_band + 1e-6

    # Calculate NDVI using the formula: NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir_band - red_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    ndvi_profile = nir_profile.copy()
    ndvi_profile.update(
        dtype=rasterio.float32,  # NDVI values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the NDVI result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **ndvi_profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)
    
    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate NDVI from multiple pairs of NIR/Red raster files and save results.

Parameters:
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    input_red_paths (list[str]): Paths to Red band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/ndvi_2022-01-16.tif") for each pair.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_ndvi`.
""")
def calculate_batch_ndvi(
    input_nir_paths: list[str],
    input_red_paths: list[str],
    output_paths: list[str]
) -> list[str]:
    """
    Batch-calculate NDVI from multiple pairs of NIR/Red rasters.

    Parameters:
        input_nir_paths (list[str]): Paths to NIR band rasters.
        input_red_paths (list[str]): Paths to Red band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/ndvi_2022-01-16.tif").

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """

    return [
        calculate_ndvi(nir_path, red_path, out_path)
        for nir_path, red_path, out_path in zip(input_nir_paths, input_red_paths, output_paths)
    ]


def calculate_ndwi(input_nir_path, input_swir_path, output_path):
    """
    Calculate the Normalized Difference Water Index (NDWI) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        input_swir_path (str): Path to the Short-Wave Infrared (SWIR) band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/ndwi_2022-01-16.tif"

    Returns:
        str: Path to the saved NDWI file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the NIR and SWIR band raster files
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)
        nir_profile = nir_src.profile  # Get the metadata profile

    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    nir_band = np.array(nir_band, dtype=np.float32)
    swir_band = np.array(swir_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = nir_band + swir_band + 1e-6

    # Calculate NDWI using the formula: NDWI = (NIR - SWIR) / (NIR + SWIR)
    ndwi = (nir_band - swir_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    ndwi_profile = nir_profile.copy()
    ndwi_profile.update(
        dtype=rasterio.float32,  # NDWI values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the NDWI result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **ndwi_profile) as dst:
        dst.write(ndwi.astype(rasterio.float32), 1)  # Write the NDWI band
    
    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate NDWI from multiple pairs of NIR/SWIR raster files and save results.

Parameters:
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    input_swir_paths (list[str]): Paths to Short-Wave Infrared (SWIR) band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/ndwi_2022-01-16.tif") for each pair.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_ndwi`.
""")
def calculate_batch_ndwi(
    input_nir_paths: list[str],
    input_swir_paths: list[str],
    output_paths: list[str]
) -> list[str]:
    """
    Batch-calculate NDWI from multiple pairs of NIR/SWIR rasters.

    Parameters:
        input_nir_paths (list[str]): Paths to NIR band rasters.
        input_swir_paths (list[str]): Paths to SWIR band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/ndwi_2022-01-16.tif").

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """

    return [
        calculate_ndwi(nir_path, swir_path, out_path)
        for nir_path, swir_path, out_path in zip(input_nir_paths, input_swir_paths, output_paths)
    ]


def calculate_ndbi(input_swir_path, input_nir_path, output_path):
    """
    Calculate the Normalized Difference Built-Up Index (NDBI) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_swir_path (str): Path to the Short-Wave Infrared (SWIR) band raster file.
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/ndbi_2022-01-16.tif"

    Returns:
        str: Path to the saved NDBI file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the SWIR and NIR band raster files
    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)  # Read the first band (assuming single-band rasters)
        swir_profile = swir_src.profile  # Get the metadata profile

    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    swir_band = np.array(swir_band, dtype=np.float32)
    nir_band = np.array(nir_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = swir_band + nir_band + 1e-6

    # Calculate NDBI using the formula: NDBI = (SWIR - NIR) / (SWIR + NIR)
    ndbi = (swir_band - nir_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    ndbi_profile = swir_profile.copy()
    ndbi_profile.update(
        dtype=rasterio.float32,  # NDBI values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the NDBI result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **ndbi_profile) as dst:
        dst.write(ndbi.astype(rasterio.float32), 1)  # Write the NDBI band

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate NDBI from multiple pairs of SWIR/NIR raster files and save results.

Parameters:
    input_swir_paths (list[str]): Paths to Short-Wave Infrared (SWIR) band raster files.
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/ndbi_2022-01-16.tif") for each pair.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_ndbi`.
""")
def calculate_batch_ndbi(
    input_swir_paths: list[str],
    input_nir_paths: list[str],
    output_paths: list[str]
) -> list[str]:
    """
    Batch-calculate NDBI from multiple pairs of SWIR/NIR rasters.

    Parameters:
        input_swir_paths (list[str]): Paths to SWIR band rasters.
        input_nir_paths (list[str]): Paths to NIR band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/ndbi_2022-01-16.tif").

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """

    return [
        calculate_ndbi(swir_path, nir_path, out_path)
        for swir_path, nir_path, out_path in zip(input_swir_paths, input_nir_paths, output_paths)
    ]


def  calculate_evi(input_nir_path, input_red_path, input_blue_path, output_path, G: float = 2.5, C1: float = 6, C2: float = 7.5, L: float = 1): 
    """
    Calculate the Enhanced Vegetation Index (EVI) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        input_red_path (str): Path to the Red band raster file.
        input_blue_path (str): Path to the Blue band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/evi_2022-01-16.tif"
        G (float, optional): Gain factor. Defaults to 2.5.
        C1 (float, optional): Coefficient 1. Defaults to 6.
        C2 (float, optional): Coefficient 2. Defaults to 7.5.
        L (float, optional): Adjustment factor. Defaults to 1.

    """
    import os
    import rasterio
    import numpy as np

    # Open the NIR, Red, and Blue band raster files
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)
        nir_profile = nir_src.profile  # Get the metadata profile

    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)  # Read the first band (assuming single-band rasters)

    with rasterio.open(input_blue_path) as blue_src:
        blue_band = blue_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    nir_band = np.array(nir_band, dtype=np.float32)
    red_band = np.array(red_band, dtype=np.float32)
    blue_band = np.array(blue_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = nir_band + C1 * red_band - C2 * blue_band + L + 1e-6

    # Calculate EVI using the formula
    evi = G * (nir_band - red_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    evi_profile = nir_profile.copy()
    evi_profile.update(
        dtype=rasterio.float32,  # EVI values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the EVI result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **evi_profile) as dst:
        dst.write(evi.astype(rasterio.float32), 1)  # Write the EVI band

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate EVI from multiple sets of NIR/Red/Blue raster files and save results.

Parameters:
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    input_red_paths (list[str]): Paths to Red band raster files.
    input_blue_paths (list[str]): Paths to Blue band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/evi_2022-01-16.tif") for each set.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_evi`.
""")
def calculate_batch_evi(
    input_nir_paths: list[str],
    input_red_paths: list[str],
    input_blue_paths: list[str],
    output_paths: list[str],
    G: float = 2.5,
    C1: float = 6,
    C2: float = 7.5,
    L: float = 1
) -> list[str]:
    """
    Batch-calculate EVI from multiple sets of NIR/Red/Blue rasters.

    Parameters:
        input_nir_paths (list[str]): Paths to NIR band rasters.
        input_red_paths (list[str]): Paths to Red band rasters.
        input_blue_paths (list[str]): Paths to Blue band rasters.
        output_paths (list[str]): Relative output paths.
        G, C1, C2, L (float, optional): EVI coefficients.

    Returns:
        list[str]: Result messages from each `calculate_evi` call.
    """
    results: list[str] = []

    for nir_path, red_path, blue_path, out_path in zip(
        input_nir_paths, input_red_paths, input_blue_paths, output_paths
    ):
        res = calculate_evi(nir_path, red_path, blue_path, out_path, G=G, C1=C1, C2=C2, L=L)
        results.append(res)

    return results


def calculate_nbr(input_nir_path, input_swir_path, output_path):
    """
    Calculate the Normalized Burn Ratio (NBR) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        input_swir_path (str): Path to the Short-Wave Infrared (SWIR) band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/nbr_2022-01-16.tif"

    Returns:
        str: Path to the saved NBR file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the NIR and SWIR band raster files
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)
        nir_profile = nir_src.profile  # Get the metadata profile

    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    nir_band = np.array(nir_band, dtype=np.float32)
    swir_band = np.array(swir_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = nir_band + swir_band + 1e-6

    # Calculate NBR using the formula: NBR = (NIR - SWIR) / (NIR + SWIR)
    nbr = (nir_band - swir_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    nbr_profile = nir_profile.copy()
    nbr_profile.update(
        dtype=rasterio.float32,  # NBR values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the NBR result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **nbr_profile) as dst:
        dst.write(nbr.astype(rasterio.float32), 1)  # Write the NBR band

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate NBR from multiple pairs of NIR/SWIR raster files and save results.

Parameters:
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    input_swir_paths (list[str]): Paths to Short-Wave Infrared (SWIR) band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/nbr_2022-01-16.tif") for each pair.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_nbr`.
""")
def calculate_batch_nbr(
    input_nir_paths: list[str],
    input_swir_paths: list[str],
    output_paths: list[str]
) -> list[str]:
    """
    Batch-calculate NBR from multiple pairs of NIR/SWIR rasters.

    Parameters:
        input_nir_paths (list[str]): Paths to NIR band rasters.
        input_swir_paths (list[str]): Paths to SWIR band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/nbr_2022-01-16.tif").

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """
    results: list[str] = []

    for nir_path, swir_path, out_path in zip(input_nir_paths, input_swir_paths, output_paths):
        res = calculate_nbr(nir_path, swir_path, out_path)
        results.append(res)

    return results


def calculate_fvc(input_nir_path, input_red_path, output_path, ndvi_min=0.1, ndvi_max=0.9):
    """
    Calculate the Fractional Vegetation Cover (FVC) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        input_red_path (str): Path to the Red band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/fvc_2022-01-16.tif"
        ndvi_min (float): Minimum NDVI value for non-vegetated areas (default: 0.1).
        ndvi_max (float): Maximum NDVI value for fully vegetated areas (default: 0.9).

    Returns:
        str: Path to the saved FVC raster file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the NIR and Red band raster files
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)
        nir_profile = nir_src.profile  # Get the metadata profile

    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    nir_band = np.array(nir_band, dtype=np.float32)
    red_band = np.array(red_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = nir_band + red_band + 1e-6

    # Calculate NDVI using the formula: NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir_band - red_band) / denominator

    # Calculate FVC using the formula: FVC = ((NDVI - NDVI_min) / (NDVI_max - NDVI_min)) * 100%
    fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) * 100

    # Clip FVC values to the range [0, 100]
    fvc = np.clip(fvc, 0, 100)

    # Update the profile for the output raster (e.g., data type and NoData value)
    fvc_profile = nir_profile.copy()
    fvc_profile.update(
        dtype=rasterio.float32,  # FVC values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the FVC result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **fvc_profile) as dst:
        dst.write(fvc.astype(rasterio.float32), 1)  # Write the FVC band

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate FVC from multiple pairs of NIR/Red raster files and save results.

Parameters:
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    input_red_paths (list[str]): Paths to Red band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/fvc_2022-01-16.tif") for each pair.
    ndvi_min (float, optional): Minimum NDVI value for non-vegetated areas (default: 0.1).
    ndvi_max (float, optional): Maximum NDVI value for fully vegetated areas (default: 0.9).

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_fvc`.
""")
def calculate_batch_fvc(
    input_nir_paths: list[str],
    input_red_paths: list[str],
    output_paths: list[str],
    ndvi_min: float = 0.1,
    ndvi_max: float = 0.9
) -> list[str]:
    """
    Batch-calculate FVC from multiple pairs of NIR/Red rasters.

    Parameters:
        input_nir_paths (list[str]): Paths to NIR band rasters.
        input_red_paths (list[str]): Paths to Red band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/fvc_2022-01-16.tif").
        ndvi_min (float, optional): Minimum NDVI value for non-vegetated areas. Defaults to 0.1.
        ndvi_max (float, optional): Maximum NDVI value for fully vegetated areas. Defaults to 0.9.

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """
    results: list[str] = []

    for nir_path, red_path, out_path in zip(input_nir_paths, input_red_paths, output_paths):
        res = calculate_fvc(nir_path, red_path, out_path, ndvi_min=ndvi_min, ndvi_max=ndvi_max)
        results.append(res)

    return results


def calculate_wri(input_green_path, input_red_path, input_nir_path, input_swir_path, output_path):
    """
    Calculate the Water Ratio Index (WRI) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_green_path (str): Path to the Green band raster file.
        input_red_path (str): Path to the Red band raster file.
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        input_swir_path (str): Path to the Short-Wave Infrared (SWIR) band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/wri_2022-01-16.tif"

    Returns:
        str: Path to the saved WRI raster file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the Green, Red, NIR, and SWIR band raster files
    with rasterio.open(input_green_path) as green_src:
        green_band = green_src.read(1)  # Read the first band (assuming single-band rasters)
        green_profile = green_src.profile  # Get the metadata profile

    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)  # Read the first band (assuming single-band rasters)

    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)

    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    green_band = np.array(green_band, dtype=np.float32)
    red_band = np.array(red_band, dtype=np.float32)
    nir_band = np.array(nir_band, dtype=np.float32)
    swir_band = np.array(swir_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = nir_band + swir_band + 1e-6

    # Calculate WRI using the formula: WRI = (Green + Red) / (NIR + SWIR)
    wri = (green_band + red_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    wri_profile = green_profile.copy()
    wri_profile.update(
        dtype=rasterio.float32,  # WRI values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the WRI result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **wri_profile) as dst:
        dst.write(wri.astype(rasterio.float32), 1)  # Write the WRI band

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate WRI from multiple sets of Green/Red/NIR/SWIR raster files and save results.

Parameters:
    input_green_paths (list[str]): Paths to Green band raster files.
    input_red_paths (list[str]): Paths to Red band raster files.
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    input_swir_paths (list[str]): Paths to Short-Wave Infrared (SWIR) band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/wri_2022-01-16.tif") for each set.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_wri`.
""")
def calculate_batch_wri(
    input_green_paths: list[str],
    input_red_paths: list[str],
    input_nir_paths: list[str],
    input_swir_paths: list[str],
    output_paths: list[str]
) -> list[str]:
    """
    Batch-calculate WRI from multiple sets of Green/Red/NIR/SWIR rasters.

    Parameters:
        input_green_paths (list[str]): Paths to Green band rasters.
        input_red_paths (list[str]): Paths to Red band rasters.
        input_nir_paths (list[str]): Paths to NIR band rasters.
        input_swir_paths (list[str]): Paths to SWIR band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/wri_2022-01-16.tif").

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """
    results: list[str] = []

    for green_path, red_path, nir_path, swir_path, out_path in zip(
        input_green_paths, input_red_paths, input_nir_paths, input_swir_paths, output_paths
    ):
        res = calculate_wri(green_path, red_path, nir_path, swir_path, out_path)
        results.append(res)

    return results


def calculate_ndti(input_red_path, input_green_path, output_path):
    """
    Calculate the Normalized Difference Turbidity Index (NDTI) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_red_path (str): Path to the Red band raster file.
        input_green_path (str): Path to the Green band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/ndti_2022-01-16.tif"

    Returns:
        str: Path to the saved NDTI file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the Red and Green band raster files
    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)  # Read the first band (assuming single-band rasters)
        red_profile = red_src.profile  # Get the metadata profile

    with rasterio.open(input_green_path) as green_src:
        green_band = green_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    red_band = np.array(red_band, dtype=np.float32)
    green_band = np.array(green_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = red_band + green_band + 1e-6

    # Calculate NDTI using the formula: NDTI = (Red - Green) / (Red + Green)
    ndti = (red_band - green_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    ndti_profile = red_profile.copy()
    ndti_profile.update(
        dtype=rasterio.float32,  # NDTI values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the NDTI result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **ndti_profile) as dst:
        dst.write(ndti.astype(rasterio.float32), 1)  # Write the NDTI band

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate NDTI from multiple pairs of Red/Green raster files and save results.

Parameters:
    input_red_paths (list[str]): Paths to Red band raster files.
    input_green_paths (list[str]): Paths to Green band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/ndti_2022-01-16.tif") for each pair.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_ndti`.
""")
def calculate_batch_ndti(
    input_red_paths: list[str],
    input_green_paths: list[str],
    output_paths: list[str]
) -> list[str]:
    """
    Batch-calculate NDTI from multiple pairs of Red/Green rasters.

    Parameters:
        input_red_paths (list[str]): Paths to Red band rasters.
        input_green_paths (list[str]): Paths to Green band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/ndti_2022-01-16.tif").

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """
    results: list[str] = []

    for red_path, green_path, out_path in zip(input_red_paths, input_green_paths, output_paths):
        res = calculate_ndti(red_path, green_path, out_path)
        results.append(res)

    return results


def calculate_frp(input_frp_path, output_path, fire_threshold=0):
    """
    Calculate Fire Radiative Power (FRP) statistics from input raster files
    and save the result to a specified output path.

    Parameters:
        input_frp_path (str): Path to the FRP raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/frp_2022-01-16.tif"
        fire_threshold (float): Minimum FRP value to be considered as fire (default: 0).
    
    Returns:
        str: Path to the saved fire mask file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the FRP raster file
    with rasterio.open(input_frp_path) as frp_src:
        frp_band = frp_src.read(1)  # Read the first band (assuming single-band rasters)
        frp_profile = frp_src.profile  # Get the metadata profile

    # Ensure the input band data is in numpy array format
    frp_band = np.array(frp_band, dtype=np.float32)

    # Create fire mask (pixels where FRP > threshold)
    fire_mask = (frp_band > fire_threshold).astype(np.uint8)
    
    # Create output raster with fire mask
    output_data = fire_mask * 255  # Convert to 0-255 range for visualization

    # Update the profile for the output raster
    fire_profile = frp_profile.copy()
    fire_profile.update(
        dtype=rasterio.uint8,  # Fire mask values are uint8
        nodata=0,  # Set NoData value to 0
    )

    # Save the fire mask result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **fire_profile) as dst:
        dst.write(output_data, 1)  # Write the fire mask

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate Fire Radiative Power (FRP) masks from multiple raster files and save results.

Parameters:
    input_frp_paths (list[str]): Paths to FRP raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/frp_2022-01-16.tif") for each file.
    fire_threshold (float, optional): Minimum FRP value to be considered as fire (default: 0).

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_frp`.
""")
def calculate_batch_frp(
    input_frp_paths: list[str],
    output_paths: list[str],
    fire_threshold: float = 0
) -> list[str]:
    """
    Batch-calculate FRP fire masks from multiple raster files.

    Parameters:
        input_frp_paths (list[str]): Paths to FRP raster files.
        output_paths (list[str]): Relative output paths (e.g., "question17/frp_2022-01-16.tif").
        fire_threshold (float, optional): Minimum FRP value to be considered as fire. Defaults to 0.

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """
    results: list[str] = []

    for frp_path, out_path in zip(input_frp_paths, output_paths):
        res = calculate_frp(frp_path, out_path, fire_threshold=fire_threshold)
        results.append(res)

    return results


def calculate_ndsi(input_green_path: str, input_swir_path: str, output_path: str) -> str:
    """
    Calculate the Normalized Difference Snow Index (NDSI) from input raster files
    and save the result to a specified output path.

    NDSI = (Green - SWIR) / (Green + SWIR)

    Parameters:
        input_green_path (str): Path to the Green band raster file.
        input_swir_path (str): Path to the SWIR band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/ndsi_2022-01-16.tif"

    Returns:
        str: Path to the output NDSI raster file.
    """
    import os
    import rasterio
    import numpy as np
    from scipy.ndimage import zoom

    # Open the Green and SWIR band raster files
    with rasterio.open(input_green_path) as green_src:
        green_band = green_src.read(1)  # Read the first band (assuming single-band rasters)
        green_profile = green_src.profile  # Get the metadata profile

    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)  # Read the first band (assuming single-band rasters)
        swir_profile = swir_src.profile

    # Ensure the input band data is in numpy array format
    green_band = np.array(green_band, dtype=np.float32)
    swir_band = np.array(swir_band, dtype=np.float32)
    
    # Apply scaling factor for MODIS surface reflectance data
    # MODIS surface reflectance is typically stored as integer values that need to be scaled by 0.0001
    scale_factor = 0.0001
    green_band = green_band * scale_factor
    swir_band = swir_band * scale_factor
    
    # Handle size mismatch by resampling to the smaller size
    if green_band.shape != swir_band.shape:
        # Determine target size (smaller of the two)
        target_height = min(green_band.shape[0], swir_band.shape[0])
        target_width = min(green_band.shape[1], swir_band.shape[1])
        target_shape = (target_height, target_width)
        
        # Resample green band if needed
        if green_band.shape != target_shape:
            zoom_factors = (target_height / green_band.shape[0], target_width / green_band.shape[1])
            green_band = zoom(green_band, zoom_factors, order=1)
        # Resample SWIR band if needed
        if swir_band.shape != target_shape:
            zoom_factors = (target_height / swir_band.shape[0], target_width / swir_band.shape[1])
            swir_band = zoom(swir_band, zoom_factors, order=1)

    # Handle invalid values and outliers
    # Set very large or very small values to NaN (outside reasonable reflectance range)
    green_band = np.where((green_band < 0) | (green_band > 1), np.nan, green_band)
    swir_band = np.where((swir_band < 0) | (swir_band > 1), np.nan, swir_band)
    
    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = green_band + swir_band + 1e-6
    
    # Set denominator to NaN where both bands are invalid
    denominator = np.where((np.isnan(green_band)) | (np.isnan(swir_band)), np.nan, denominator)

    # Calculate NDSI using the formula: NDSI = (Green - SWIR) / (Green + SWIR)
    ndsi = (green_band - swir_band) / denominator
    
    # Ensure NDSI values are within reasonable range [-1, 1]
    ndsi = np.clip(ndsi, -1, 1)

    # Update the profile for the output raster (e.g., data type and NoData value)
    # Use the profile from the smaller image
    output_profile = swir_profile.copy() if swir_band.shape[0] <= green_band.shape[0] else green_profile.copy()
    output_profile.update(
        dtype=rasterio.float32,  # NDSI values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw',  # Optional: compress the output file
        height=ndsi.shape[0],
        width=ndsi.shape[1]
    )

    # Save the NDSI result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **output_profile) as dst:
        dst.write(ndsi.astype(rasterio.float32), 1)  # Write the NDSI band

    return f'Result save at {TEMP_DIR / output_path}'


@mcp.tool(description="""
Calculate NDSI for multiple pairs of Green and SWIR band images.

Parameters:
    green_file_list (list[str]): List of paths to Green band raster files.
    swir_file_list (list[str]): List of paths to SWIR band raster files.
    output_path_list (list[str]): relative path for the output raster file, e.g. ["question17/ndsi_2022-01-16.tif", "question17/ndsi_2022-01-16.tif"]

Returns:
    list[str]: List of paths to the output NDSI raster files.
""")
def calculate_batch_ndsi(green_file_list: list[str], swir_file_list: list[str], output_path_list: list[str]) -> list[str]:
    """
    Calculate NDSI for multiple pairs of Green and SWIR band images.

    Parameters:
        green_file_list (list[str]): List of paths to Green band raster files.
        swir_file_list (list[str]): List of paths to SWIR band raster files.
        output_path_list (list[str]): relative path for the output raster file, e.g. ["question17/ndsi_2022-01-16.tif", "question17/ndsi_2022-01-16.tif"]

    Returns:
        list[str]: List of paths to the output NDSI raster files.
    """
    if len(green_file_list) != len(swir_file_list):
        raise ValueError("Number of Green and SWIR files must be equal")
    
    results = []
    for i, (green_path, swir_path) in enumerate(zip(green_file_list, swir_file_list)):
        result = calculate_ndsi(green_path, swir_path, output_path_list[i])
        results.append(result)
    
    return results



@mcp.tool(description="""
Calculate the percentage of extreme snow and ice loss areas from a binary map.

Parameters:
    binary_map_path (str):
        Path to the binary raster image where pixels with value 1.0 
        represent extreme snow/ice loss areas.

Returns:
    float:
        The percentage of extreme snow/ice loss pixels relative to all valid pixels 
        (range: 0.0–1.0).

Example:
    >>> calc_extreme_snow_loss_percentage_from_binary_map("snow_loss_binary.tif")
    0.27
""")
def calc_extreme_snow_loss_percentage_from_binary_map(binary_map_path: str) -> float:
    """
    Calculate the percentage of extreme snow and ice loss areas from a binary map.

    Parameters:
        binary_map_path (str):
            Path to the binary raster image where pixels with value 1.0 
            represent extreme snow/ice loss areas.

    Returns:
        float:
            The percentage of extreme snow/ice loss pixels relative to all valid pixels 
            (range: 0.0–1.0).

    Example:
        >>> calc_extreme_snow_loss_percentage_from_binary_map("snow_loss_binary.tif")
        0.27
    """
    import rasterio
    import numpy as np

    img = read_image(binary_map_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    
    # Remove NaN values for calculation
    valid_pixels = flat[~np.isnan(flat)]
    
    if len(valid_pixels) == 0:
        return 0.0
    
    # Calculate extreme snow loss percentage (binary map where 1.0 indicates extreme loss)
    extreme_loss_pixels = valid_pixels[valid_pixels == 1.0]
    extreme_loss_percentage = len(extreme_loss_pixels) / len(valid_pixels)
    
    return float(extreme_loss_percentage)


@mcp.tool(description='''
Compute TVDI (Temperature Vegetation Dryness Index) using NDVI and LST from local raster files.

Parameters:
    ndvi_path (str): Path to local NDVI GeoTIFF (e.g., MODIS NDVI scaled by 0.0001).
    lst_path (str): Path to local LST GeoTIFF (e.g., MODIS LST scaled by 0.02).
    output_path (str): relative path for the output raster file, e.g. "question17/tvdi_2022-01-16.tif"

Returns:
    str: Path to the exported TVDI GeoTIFF.
''')
def compute_tvdi(
    ndvi_path: str,
    lst_path: str,
    output_path: str
) -> str:
    """
    Description:
        Compute the Temperature Vegetation Dryness Index (TVDI) based on NDVI and LST raster data.
        TVDI quantifies soil moisture conditions by analyzing the relationship between NDVI and LST
        through a trapezoidal space approach. The function fits linear regressions for LST maxima
        and minima per NDVI bin and normalizes per-pixel LST values accordingly.

    Parameters:
        ndvi_path (str): Path to the NDVI GeoTIFF file (e.g., MODIS NDVI scaled by 0.0001).
        lst_path (str): Path to the LST GeoTIFF file (e.g., MODIS LST scaled by 0.02).
        output_path (str): Relative path to save the computed TVDI raster
                           (e.g., "question17/tvdi_2022-01-16.tif").

    Return:
        str: Path to the saved TVDI GeoTIFF file.

    Example:
        >>> compute_tvdi(
        ...     ndvi_path="data/ndvi_2022-01-16.tif",
        ...     lst_path="data/lst_2022-01-16.tif",
        ...     output_path="question17/tvdi_2022-01-16.tif"
        ... )
        'Result saved at question17/tvdi_2022-01-16.tif'
    """
    import os
    import rasterio
    import numpy as np
    from scipy.stats import linregress

    # Read NDVI and LST
    with rasterio.open(ndvi_path) as src_ndvi:
        ndvi = src_ndvi.read(1).astype(np.float32) * 0.0001
        profile = src_ndvi.profile
    with rasterio.open(lst_path) as src_lst:
        lst = src_lst.read(1).astype(np.float32) * 0.02

    # Create mask for valid data
    valid_mask = (ndvi >= 0) & (ndvi <= 1) & (lst > 0)

    # No valid data
    if not np.any(valid_mask):
        print(f"Warning: No valid data points in {output_path}")
        tvdi = np.full_like(ndvi, np.nan, dtype=np.float32)
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
        with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
            dst.write(tvdi, 1)
        return f'Result saved at {TEMP_DIR / output_path}'

    ndvi_valid = ndvi[valid_mask]
    lst_valid = lst[valid_mask]

    # Not enough valid pixels
    if len(ndvi_valid) < 100:
        print(f"Warning: Too few valid data points ({len(ndvi_valid)}) in {output_path}")
        tvdi = np.full_like(ndvi, np.nan, dtype=np.float32)
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
        with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
            dst.write(tvdi, 1)
        return f'Result saved at {TEMP_DIR / output_path}'

    # Bin NDVI values
    n_bins = 100
    bins = np.linspace(ndvi_valid.min(), ndvi_valid.max(), n_bins + 1)
    ndvi_bin_centers, lst_max_vals, lst_min_vals = [], [], []

    for i in range(n_bins):
        bin_mask = (ndvi_valid >= bins[i]) & (ndvi_valid < bins[i + 1])
        if np.any(bin_mask):
            ndvi_bin_centers.append((bins[i] + bins[i + 1]) / 2)
            lst_max_vals.append(np.max(lst_valid[bin_mask]))
            lst_min_vals.append(np.min(lst_valid[bin_mask]))

    # Not enough bins for regression
    if len(ndvi_bin_centers) < 2:
        print(f"Warning: Not enough data bins for regression in {output_path}")
        tvdi = np.full_like(ndvi, np.nan, dtype=np.float32)
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
        with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
            dst.write(tvdi, 1)
        return f'Result saved at {TEMP_DIR / output_path}'

    ndvi_bin_centers = np.array(ndvi_bin_centers)
    lst_max_vals = np.array(lst_max_vals)
    lst_min_vals = np.array(lst_min_vals)

    # Linear regression
    slope_max, intercept_max, _, _, _ = linregress(ndvi_bin_centers, lst_max_vals)
    slope_min, intercept_min, _, _, _ = linregress(ndvi_bin_centers, lst_min_vals)

    lst_max = ndvi * slope_max + intercept_max
    lst_min = ndvi * slope_min + intercept_min

    # TVDI calculation
    denominator = lst_max - lst_min
    denominator[denominator == 0] = 1e-6
    tvdi = (lst - lst_min) / denominator
    tvdi = np.clip(tvdi, 0, 1).astype(np.float32)
    tvdi[~valid_mask] = np.nan

    # Save result
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(tvdi, 1)

    return f'Result saved at {TEMP_DIR / output_path}'


if __name__ == "__main__":
    mcp.run() 
