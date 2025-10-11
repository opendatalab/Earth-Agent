import os
import numpy as np
from osgeo import gdal
import rasterio
from pathlib import Path
from scipy.stats import linregress
from typing import List
import pandas as pd

def get_filelist(dir_path: str):
    """
    Returns a list of .tif files in the specified directory.

    Parameters:
        dir_path (str): Path to the directory.

    Returns:
        list: List of .tif file names in the directory.
    """
    files = os.listdir(dir_path)
    # Filter for .tif files and exclude .tif.enp files
    tif_files = [f for f in files if f.endswith('.tif') and not f.endswith('.tif.enp')]
    return sorted(tif_files)
    
def read_image(file_path: str) -> np.ndarray:
    ds = gdal.Open(file_path)
    if ds is None:
        raise RuntimeError(f"Failed to open {file_path}")
    
    bands = ds.RasterCount
    if bands == 1:
        img = ds.GetRasterBand(1).ReadAsArray()
    else:
        img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
        img = np.transpose(img, (1, 2, 0))

    ds = None
    return img

def calc_single_image_mean(file_path: str, uint8: bool = False) -> float:
    """
    Compute mean value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        mean (float): Mean pixel value
    """
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return np.nanmean(flat)

def calc_batch_image_mean(file_list: list[str], uint8: bool = False) -> list[float]:
    """
    Compute mean value of an batch of images.

    Parameters:
        file_list (list(str)): Paths to input images.

    Returns:
        mean (list(float)): Mean pixel value
    """

    return [calc_single_image_mean(file_path, uint8) for file_path in file_list]

def calculate_tif_average(file_list: list[str], output_filename: str = "avg_result.tif", uint8: bool = False) -> str:
    """
    Calculate average of multiple tif files and save result to same directory.

    Parameters:
        file_list (list[str]): List of tif file paths.
        output_filename (str): Output filename, default "avg_result.tif".
        uint8 (bool): Convert to uint8 format, default False.

    Returns:
        output_path (str): Full path of output file.
    """
    # Output directory unified to data/question1
    output_dir = r"F:/benchmark/supported_data/question1"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Read first file to get basic info
    ds = gdal.Open(file_list[0])
    bands = ds.RasterCount
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    
    # Read first image
    if bands == 1:
        first_img = ds.GetRasterBand(1).ReadAsArray()
    else:
        first_img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
        first_img = np.transpose(first_img, (1, 2, 0))
    ds = None
    
    # Initialize accumulator and counter
    sum_img = np.zeros_like(first_img, dtype=np.float64)
    count_img = np.zeros_like(first_img, dtype=np.float64)
    
    # Add first image
    valid_mask = ~np.isnan(first_img)
    sum_img[valid_mask] += first_img[valid_mask]
    count_img[valid_mask] += 1
    
    # Read and accumulate remaining images
    for file_path in file_list[1:]:
        ds = gdal.Open(file_path)
        if bands == 1:
            img = ds.GetRasterBand(1).ReadAsArray()
        else:
            img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
            img = np.transpose(img, (1, 2, 0))
        ds = None
        
        valid_mask = ~np.isnan(img)
        sum_img[valid_mask] += img[valid_mask]
        count_img[valid_mask] += 1
    
    # Calculate average (avoiding division by zero)
    count_img[count_img == 0] = 1  # Avoid division by zero
    avg_img = sum_img / count_img
    
    # Set pixels with no valid data to NaN
    avg_img[count_img == 0] = np.nan
    
    # Convert to uint8 if needed
    if uint8:
        if len(avg_img.shape) == 2:
            valid_mask = ~np.isnan(avg_img)
            if np.any(valid_mask):
                min_val = np.nanmin(avg_img)
                max_val = np.nanmax(avg_img)
                avg_img = (avg_img - min_val) / (max_val - min_val) * 255
            avg_img = avg_img.astype(np.uint8)
        else:
            for band in range(avg_img.shape[2]):
                band_data = avg_img[:, :, band]
                valid_mask = ~np.isnan(band_data)
                if np.any(valid_mask):
                    min_val = np.nanmin(band_data)
                    max_val = np.nanmax(band_data)
                    band_data = (band_data - min_val) / (max_val - min_val) * 255
                avg_img[:, :, band] = band_data.astype(np.uint8)
    
    # Save result
    driver = gdal.GetDriverByName('GTiff')
    data_type = gdal.GDT_Byte if uint8 else gdal.GDT_Float32
    
    if len(avg_img.shape) == 2:
        # Single band
        out_ds = driver.Create(output_path, cols, rows, 1, data_type)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        out_ds.GetRasterBand(1).WriteArray(avg_img)
    else:
        # Multi band
        out_ds = driver.Create(output_path, cols, rows, bands, data_type)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        for i in range(bands):
            out_ds.GetRasterBand(i + 1).WriteArray(avg_img[:, :, i])
    
    out_ds = None
    return output_path

def calculate_annual_mean(tvdi_files: list[str]) -> tuple[float, int]:
    """
    Calculate annual mean TVDI by first calculating mean of each image,
    then taking the mean of all image means.
    
    Parameters:
        tvdi_files (list[str]): List of TVDI file paths for a year
        
    Returns:
        tuple[float, int]: Annual mean TVDI and number of valid images used
    """
    image_means = []
    
    for file_path in tvdi_files:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            if not np.all(np.isnan(data)):  
                mean_value = float(np.nanmean(data))  
                if not np.isnan(mean_value):  
                    image_means.append(mean_value)
    
    if not image_means: 
        return np.nan, 0
        
    return float(np.mean(image_means)), len(image_means)

def difference(a: float, b: float) -> float:
    """
    Calculate the difference between two numbers.

    Parameters:
        a (float): First number.
        b (float): Second number.

    Returns:
        diff (float): Absolute difference between the two numbers.
    """
    return abs(a - b)

def compute_tvdi(
    ndvi_path: str,
    lst_path: str,
    save_name: str = "tvdi_output.tif"
) -> str:
    import rasterio
    import numpy as np
    from pathlib import Path
    from scipy.stats import linregress

    out_dir = Path(r"F:/benchmark/supported_data/question1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / save_name

    # Read NDVI and LST data
    with rasterio.open(ndvi_path) as src_ndvi:
        ndvi = src_ndvi.read(1).astype(np.float32) * 0.0001
        profile = src_ndvi.profile

    with rasterio.open(lst_path) as src_lst:
        lst = src_lst.read(1).astype(np.float32) * 0.02

    # Validity mask
    valid_mask = (ndvi >= 0) & (ndvi <= 1) & (lst > 0)
    if not np.any(valid_mask):
        print(f"Warning: {save_name} has no valid data points")
        tvdi = np.full_like(ndvi, np.nan, dtype=np.float32)
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(tvdi, 1)
        return str(out_file)

    ndvi_valid = ndvi[valid_mask]
    lst_valid = lst[valid_mask]
    if len(ndvi_valid) < 100:
        print(f"Warning: {save_name} has too few valid data points ({len(ndvi_valid)} points)")
        tvdi = np.full_like(ndvi, np.nan, dtype=np.float32)
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(tvdi, 1)
        return str(out_file)

    n_bins = 100
    bins = np.linspace(ndvi_valid.min(), ndvi_valid.max(), n_bins + 1)
    ndvi_bin_centers = []
    lst_max_vals = []
    lst_min_vals = []
    for i in range(n_bins):
        bin_mask = (ndvi_valid >= bins[i]) & (ndvi_valid < bins[i + 1])
        if np.any(bin_mask):
            ndvi_bin_centers.append((bins[i] + bins[i + 1]) / 2)
            lst_max_vals.append(np.max(lst_valid[bin_mask]))
            lst_min_vals.append(np.min(lst_valid[bin_mask]))
    if len(ndvi_bin_centers) < 2:
        print(f"Warning: {save_name} does not have enough data for regression analysis")
        tvdi = np.full_like(ndvi, np.nan, dtype=np.float32)
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(tvdi, 1)
        return str(out_file)
    # Regression analysis
    slope_min, intercept_min, _, _, _ = linregress(ndvi_bin_centers, lst_min_vals)
    slope_max, intercept_max, _, _, _ = linregress(ndvi_bin_centers, lst_max_vals)
    lst_min_fit = slope_min * ndvi + intercept_min
    lst_max_fit = slope_max * ndvi + intercept_max
    tvdi = (lst - lst_min_fit) / (lst_max_fit - lst_min_fit)
    tvdi[~valid_mask] = np.nan
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(tvdi, 1)
    return str(out_file)

if __name__ == "__main__":
    # 1. Get LST and NDVI file lists
    data_dir = r"F:\benchmark\data\question1"
    if not os.path.exists(data_dir):
        raise RuntimeError(f"Data directory does not exist: {data_dir}")
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.tif') and not f.endswith('.tif.enp')]
    lst_files = sorted([f for f in all_files if f.endswith('_LST.tif')])
    ndvi_files = sorted([f for f in all_files if f.endswith('_NDVI.tif')])
    print("Step 1 - File statistics:")
    print(f"Number of LST files: {len(lst_files)}")
    print(f"Number of NDVI files: {len(ndvi_files)}")
    if len(lst_files) == 0 or len(ndvi_files) == 0:
        raise RuntimeError("No data files found")
    print(f"Example LST file: {lst_files[0]}")
    print(f"Example NDVI file: {ndvi_files[0]}")
    print("-" * 50)

    # 2. Automatically pair LST and NDVI files (by date)
    def extract_date(filename, key):
        # e.g. Xinjiang_2022-12-19_LST.tif
        parts = filename.split('_')
        for p in parts:
            if '-' in p:
                return p
        return None

    lst_dict = {extract_date(f, 'LST'): f for f in lst_files}
    ndvi_dict = {extract_date(f, 'NDVI'): f for f in ndvi_files}
    common_dates = sorted(set(lst_dict.keys()) & set(ndvi_dict.keys()))

    print(f"Number of matched dates: {len(common_dates)}")
    if len(common_dates) == 0:
        raise RuntimeError("No matched LST and NDVI files")

    # 3. Organize by year
    years = sorted(set([d[:4] for d in common_dates]))
    annual_means = {}

    for year in years:
        print(f"\nProcessing year {year}:")
        year_dates = [d for d in common_dates if d.startswith(year)]
        print(f"Number of matched dates in this year: {len(year_dates)}")
        year_tvdi_files = []
        for date_str in year_dates:
            lst_file = lst_dict[date_str]
            ndvi_file = ndvi_dict[date_str]
            lst_path = os.path.join(data_dir, lst_file)
            ndvi_path = os.path.join(data_dir, ndvi_file)
            if not os.path.exists(lst_path) or not os.path.exists(ndvi_path):
                print(f"Warning: File not found - LST: {lst_path}, NDVI: {ndvi_path}")
                continue
            print(f"Calculating TVDI: {date_str} ...")
            try:
                tvdi_output = compute_tvdi(
                    ndvi_path=ndvi_path,
                    lst_path=lst_path,
                    save_name=f"tvdi_{date_str}.tif"
                )
                year_tvdi_files.append(tvdi_output)
            except Exception as e:
                print(f"Error calculating TVDI: {e}")
                continue
        if not year_tvdi_files:
            print(f"Warning: No TVDI files generated for year {year}")
            continue
        # 4. Calculate annual average TVDI
        print(f"Calculating annual average TVDI: {year} ...")
        try:
            annual_avg_file = calculate_tif_average(
                year_tvdi_files,
                output_filename=f"tvdi_annual_avg_{year}.tif"
            )
            annual_mean, valid_count = calculate_annual_mean(year_tvdi_files)
            if not np.isnan(annual_mean):
                annual_means[year] = annual_mean
                print(f"Year {year} average TVDI: {annual_mean:.4f} (based on {valid_count} valid time points)")
            else:
                print(f"Warning: Year {year} has no valid TVDI data")
        except Exception as e:
            print(f"Error calculating annual average: {e}")
            continue

    if not annual_means:
        raise RuntimeError("No annual average TVDI values generated")

    # 5. Calculate multi-year trend
    print("\nStep 5 - Calculating multi-year trend:")
    valid_years = [int(y) for y in annual_means.keys()]
    valid_means = [annual_means[y] for y in annual_means.keys()]
    if len(valid_years) < 2:
        print("Warning: Not enough valid data for trend analysis")
    else:
        slope, intercept, r_value, p_value, std_err = linregress(valid_years, valid_means)
        print("\nTrend analysis result:")
        print(f"Slope: {slope:.6f}")
        print(f"Intercept: {intercept:.6f}")
        print(f"R squared: {r_value*r_value:.4f}")
        print(f"P value: {p_value:.4f}")
        print("\nTrend interpretation:")
        if p_value < 0.05:
            if slope > 0:
                print("Significant increasing trend, indicating increasing dryness in the study area")
            else:
                print("Significant decreasing trend, indicating decreasing dryness in the study area")
        else:
            print("No significant trend observed")
    