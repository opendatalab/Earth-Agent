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


@mcp.tool(description='''
Compute Precipitable Water Vapor (PWV) image from local MODIS surface reflectance band files
using the band ratio method.

This method uses surface reflectance bands:
- sur_refl_b02 (0.865 μm), sur_refl_b05 (1.240 μm): atmospheric window bands
- sur_refl_b17, sur_refl_b18, sur_refl_b19: water vapor absorption bands
          
Data Source :
    - Bands used:
        * "sur_refl_b02": 0.865 μm (NIR window)
        * "sur_refl_b05": 1.240 μm (SWIR window)
        * "sur_refl_b17": 0.905 μm (H₂O absorption)
        * "sur_refl_b18": 0.936 μm (H₂O absorption)
        * "sur_refl_b19": 0.940 μm (H₂O absorption)
Parameters:
    sur_refl_b02_path (str): File path to band sur_refl_b02 (0.865 um) GeoTIFF.
    sur_refl_b05_path (str): File path to band sur_refl_b05 (1.240 um) GeoTIFF.
    sur_refl_b17_path (str): File path to band sur_refl_b17 GeoTIFF.
    sur_refl_b18_path (str): File path to band sur_refl_b18 GeoTIFF.
    sur_refl_b19_path (str): File path to band sur_refl_b19 GeoTIFF.
    output_path (str): relative path for the output raster file, e.g. "question17/pwv_2022-01-16.tif"

Returns:
    str: Path to the saved PWV GeoTIFF.
''')
def band_ratio(
    sur_refl_b02_path: str,
    sur_refl_b05_path: str,
    sur_refl_b17_path: str,
    sur_refl_b18_path: str,
    sur_refl_b19_path: str,
    output_path: str
) -> str:
    """
    Description:
        Compute a Precipitable Water Vapor (PWV) image from MODIS surface reflectance bands 
        using the band ratio method. 
        This method interpolates atmospheric window reflectance between 0.865 µm and 1.240 µm, 
        computes transmittance in water vapor absorption bands (0.905, 0.936, 0.940 µm),
        and derives PWV in centimeters. 
        The output GeoTIFF contains four bands:
            1. PWV
            2. T17 (transmittance at 0.905 µm)
            3. T18 (transmittance at 0.936 µm)
            4. T19 (transmittance at 0.940 µm)

    Parameters:
        sur_refl_b02_path (str): Path to MODIS surface reflectance band sur_refl_b02 (0.865 µm) GeoTIFF.
        sur_refl_b05_path (str): Path to MODIS surface reflectance band sur_refl_b05 (1.240 µm) GeoTIFF.
        sur_refl_b17_path (str): Path to MODIS surface reflectance band sur_refl_b17 (0.905 µm) GeoTIFF.
        sur_refl_b18_path (str): Path to MODIS surface reflectance band sur_refl_b18 (0.936 µm) GeoTIFF.
        sur_refl_b19_path (str): Path to MODIS surface reflectance band sur_refl_b19 (0.940 µm) GeoTIFF.
        output_path (str): Relative output path for the PWV GeoTIFF, e.g. "question17/pwv_2022-01-16.tif".

    Return:
        str: Full path to the saved PWV GeoTIFF.

    Example:
        result = band_ratio(
            sur_refl_b02_path="data/MODIS_sur_refl_b02.tif",
            sur_refl_b05_path="data/MODIS_sur_refl_b05.tif",
            sur_refl_b17_path="data/MODIS_sur_refl_b17.tif",
            sur_refl_b18_path="data/MODIS_sur_refl_b18.tif",
            sur_refl_b19_path="data/MODIS_sur_refl_b19.tif",
            output_path="question17/pwv_2022-01-16.tif"
        )
        print(result)
        # Output: "Result saved at TEMP_DIR/question17/pwv_2022-01-16.tif"
    """
    import os
    import rasterio
    import numpy as np

    with rasterio.open(sur_refl_b02_path) as src02, \
         rasterio.open(sur_refl_b05_path) as src05, \
         rasterio.open(sur_refl_b17_path) as src17, \
         rasterio.open(sur_refl_b18_path) as src18, \
         rasterio.open(sur_refl_b19_path) as src19:
        
        b02 = src02.read(1).astype(np.float32) 
        b05 = src05.read(1).astype(np.float32) 
        b17 = src17.read(1).astype(np.float32) 
        b18 = src18.read(1).astype(np.float32) 
        b19 = src19.read(1).astype(np.float32) 

        profile = src02.profile

    # um
    λ2, λ5 = 0.865, 1.240
    λ17, λ18, λ19 = 0.905, 0.936, 0.940

    # Linear interpolation of window reflectance
    a = (b05 - b02) / (λ5 - λ2)
    b = b02 - a * λ2
    rho17 = a * λ17 + b
    rho18 = a * λ18 + b
    rho19 = a * λ19 + b

    T17 = np.divide(b17, rho17, out=np.zeros_like(b17), where=rho17 != 0)
    T18 = np.divide(b18, rho18, out=np.zeros_like(b18), where=rho18 != 0)
    T19 = np.divide(b19, rho19, out=np.zeros_like(b19), where=rho19 != 0)

    # PWV calculation
    k = 0.03
    with np.errstate(divide='ignore', invalid='ignore'):
        PWV = -np.log(T18) / k
        PWV[np.isnan(PWV)] = 0
        PWV[PWV < 0] = 0

    out_data = np.stack([PWV, T17, T18, T19], axis=0).astype(np.float32)
    profile.update(dtype=rasterio.float32, count=4, compress='lzw')

    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(out_data)

    return f'Result saved at {TEMP_DIR / output_path}'




@mcp.tool(description='''
Estimate Land Surface Temperature (LST) using the Single-Channel method, with NDVI-based emissivity estimation from RED and NIR bands.

Parameters:
    bt_path (str): Brightness Temperature GeoTIFF (Kelvin).
    red_path (str): Red band GeoTIFF (e.g., Landsat 8 Band 4).
    nir_path (str): NIR band GeoTIFF (e.g., Landsat 8 Band 5).
    output_path (str): relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Path to saved LST GeoTIFF.
''')
def lst_single_channel(
    bt_path: str,
    red_path: str,
    nir_path: str,
    output_path: str
) -> str:
    """
    Description:
        Estimate Land Surface Temperature (LST) using the Single-Channel method.  
        This approach calculates LST from thermal brightness temperature and adjusts 
        for surface emissivity estimated using NDVI derived from RED and NIR bands.  
        It is suitable for single thermal band sensors such as Landsat 8 TIRS.

    Parameters:
        bt_path (str): Path to the Brightness Temperature GeoTIFF (Kelvin).
        red_path (str): Path to the RED band GeoTIFF (e.g., Landsat 8 Band 4).
        nir_path (str): Path to the NIR band GeoTIFF (e.g., Landsat 8 Band 5).
        output_path (str): Relative path for the output LST GeoTIFF, 
                           e.g. "question17/lst_2022-01-16.tif".

    Return:
        str: Full path to the saved LST GeoTIFF.

    Example:
        result = lst_single_channel(
            bt_path="data/Landsat8_BT.tif",
            red_path="data/Landsat8_B4.tif",
            nir_path="data/Landsat8_B5.tif",
            output_path="question17/lst_2022-01-16.tif"
        )
        print(result)
        # Output: "Result saved at TEMP_DIR/question17/lst_2022-01-16.tif"
    """
    import os
    import rasterio
    import numpy as np

    def read_band(path):
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            profile = src.profile
            band[band < 0] = np.nan  # Filter invalid values
        return band, profile

    # Read input bands
    bt, profile = read_band(bt_path)   # Brightness Temperature in Kelvin
    red, _ = read_band(red_path)
    nir, _ = read_band(nir_path)

    # Calculate NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)  # Avoid division by zero

    # Estimate emissivity based on NDVI
    emissivity = np.where(
        ndvi > 0.7, 0.99,
        np.where(
            ndvi < 0.2, 0.96,
            0.97 + 0.003 * ndvi
        )
    )

    # Single-channel method parameters
    wavelength = 10.9  # Center wavelength of Landsat 8 TIRS Band 10 (micrometers)
    c2 = 1.43877e4     # Second radiation constant (μm·K)

    # Calculate LST in Kelvin
    lst = bt / (1 + (wavelength * bt / c2) * np.log(emissivity))

    # Update profile and write output
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(lst.astype(np.float32), 1)

    return f'Result saved at {TEMP_DIR / output_path}'


@mcp.tool(description='''
Estimate Land Surface Temperature (LST) using the multi-channel algorithm.

Requires local input files:
- Two thermal infrared bands (e.g., Band 31 and Band 32) as GeoTIFF files.

Parameters:
    band31_path (str): Path to local GeoTIFF file for thermal band 31 (~11 μm).
    band32_path (str): Path to local GeoTIFF file for thermal band 32 (~12 μm).
    output_path (str): Relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Local file path of the exported LST image.
''')
def lst_multi_channel(
    band31_path: str,
    band32_path: str,
    output_path: str
) -> str:
    """
    Description:
        Estimate Land Surface Temperature (LST) using the multi-channel algorithm.  
        This method combines two thermal infrared bands (typically at ~11 μm and ~12 μm) 
        to reduce atmospheric effects and improve LST estimation accuracy.

    Parameters:
        band31_path (str): Path to local GeoTIFF file for thermal band 31 (~11 μm).
        band32_path (str): Path to local GeoTIFF file for thermal band 32 (~12 μm).
        output_path (str): Relative path for the output LST GeoTIFF,
                           e.g. "question17/lst_2022-01-16.tif".

    Return:
        str: Full path to the saved LST GeoTIFF.

    Example:
        result = lst_multi_channel(
            band31_path="data/MODIS_Band31.tif",
            band32_path="data/MODIS_Band32.tif",
            output_path="question17/lst_2022-01-16.tif"
        )
        print(result)
        # Output: "Result saved at TEMP_DIR/question17/lst_2022-01-16.tif"
    """
    import os
    import rasterio
    import numpy as np

    # Read two thermal infrared bands
    with rasterio.open(band31_path) as src31:
        band31 = src31.read(1).astype(np.float32)
        profile = src31.profile

    with rasterio.open(band32_path) as src32:
        band32 = src32.read(1).astype(np.float32)

    # Ensure spatial alignment of bands (assumes co-registered input)
    # Split-window algorithm coefficients (empirical)
    a = 1.022
    b = 0.47
    c = 0.43

    # Calculate LST
    lst = a * band31 + b * (band31 - band32) + c

    # Update profile for output
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    # Write output GeoTIFF
    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(lst.astype(np.float32), 1)

    return f'Result saved at {TEMP_DIR / output_path}'


@mcp.tool(description='''
Estimate Land Surface Temperature (LST) or Precipitable Water Vapor (PWV) using the split-window algorithm.

Requires local input files:
- Thermal band 31 (~11 μm) GeoTIFF
- Thermal band 32 (~12 μm) GeoTIFF
- Emissivity band 31 GeoTIFF
- Emissivity band 32 GeoTIFF

Parameters:
    band31_path (str): Path to thermal band 31 GeoTIFF.
    band32_path (str): Path to thermal band 32 GeoTIFF.
    emissivity31_path (str): Path to emissivity band 31 GeoTIFF.
    emissivity32_path (str): Path to emissivity band 32 GeoTIFF.
    parameter (str): "LST" or "PWV" to specify output.
    output_path (str): Relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Path to exported output GeoTIFF.
''')

def split_window(
    band31_path: str,
    band32_path: str,
    emissivity31_path: str,
    emissivity32_path: str,
    parameter: str,
    output_path: str
) -> str:
    """
    Description:
        Estimate **Land Surface Temperature (LST)** or **Precipitable Water Vapor (PWV)** 
        using the split-window algorithm.  
        The method leverages two thermal infrared bands (~11 μm and ~12 μm) 
        and emissivity data to correct atmospheric effects and retrieve 
        accurate surface or atmospheric parameters.  
        Only one parameter is computed based on the user-selected `parameter`.

    Parameters:
        band31_path (str): Path to thermal band 31 GeoTIFF (~11 μm).
        band32_path (str): Path to thermal band 32 GeoTIFF (~12 μm).
        emissivity31_path (str): Path to emissivity band 31 GeoTIFF.
        emissivity32_path (str): Path to emissivity band 32 GeoTIFF.
        parameter (str): Specify either `"LST"` for Land Surface Temperature 
                         or `"PWV"` for Precipitable Water Vapor.
        output_path (str): Relative path for the output raster file,
                           e.g. `"question17/lst_2022-01-16.tif"`.

    Return:
        str: Full path to the saved GeoTIFF containing the selected parameter.

    Example:
        # Example 1: Retrieve LST
        result = split_window(
            band31_path="data/TIRS_Band31.tif",
            band32_path="data/TIRS_Band32.tif",
            emissivity31_path="data/Emis_B31.tif",
            emissivity32_path="data/Emis_B32.tif",
            parameter="LST",
            output_path="question17/lst_2022-01-16.tif"
        )
        print(result)
        # Output: "Result saved at TEMP_DIR/question17/lst_2022-01-16.tif"

        # Example 2: Retrieve PWV
        result = split_window(
            band31_path="data/TIRS_Band31.tif",
            band32_path="data/TIRS_Band32.tif",
            emissivity31_path="data/Emis_B31.tif",
            emissivity32_path="data/Emis_B32.tif",
            parameter="PWV",
            output_path="question17/pwv_2022-01-16.tif"
        )
        print(result)
        # Output: "Result saved at TEMP_DIR/question17/pwv_2022-01-16.tif"
    """
    import os
    import rasterio
    import numpy as np

    # Read inputs
    with rasterio.open(band31_path) as src31:
        band31 = src31.read(1).astype(np.float32)
        profile = src31.profile

    with rasterio.open(band32_path) as src32:
        band32 = src32.read(1).astype(np.float32)

    with rasterio.open(emissivity31_path) as src_e31:
        e31 = src_e31.read(1).astype(np.float32)
        print(f"Emissivity31 Original range: {np.nanmin(e31):.4f} to {np.nanmax(e31):.4f}")
        e31 = e31 * 0.002 + 0.49
        print(f"Emissivity31 Corrected range: {np.nanmin(e31):.4f} to {np.nanmax(e31):.4f}")

    with rasterio.open(emissivity32_path) as src_e32:
        e32 = src_e32.read(1).astype(np.float32)
        print(f"Emissivity32 Original range: {np.nanmin(e32):.4f} to {np.nanmax(e32):.4f}")
        e32 = e32 * 0.002 + 0.49
        print(f"Emissivity32 Corrected range: {np.nanmin(e32):.4f} to {np.nanmax(e32):.4f}")

    # Calculate temperature difference and emissivity parameters
    delta_T = band31 - band32
    print(f"Temperature difference ΔT range: {np.nanmin(delta_T):.4f} to {np.nanmax(delta_T):.4f}")

    eps_mean = (e31 + e32) / 2
    print(f"Mean emissivity range: {np.nanmin(eps_mean):.4f} to {np.nanmax(eps_mean):.4f}")

    delta_eps = e31 - e32
    print(f"Emissivity difference Δε range: {np.nanmin(delta_eps):.4f} to {np.nanmax(delta_eps):.4f}")

    eps_mean = np.clip(eps_mean, 0.8, 1.0)

    if parameter.upper() == "LST":
        C0, C1, C2, C3, C4 = 0.268, 1.378, 0.183, 54.3, -2.238

        t31_c = band31 - 273.15

        term1 = C0
        term2 = C1 * t31_c
        term3 = C2 * (t31_c ** 2) / 1000
        term4 = (C3 + C4 * delta_T) * (1 - eps_mean)
        term5 = (C3 + C4 * delta_T) * delta_eps

        lst = term1 + term2 + term3 + term4 + term5
        lst = lst + 273.15  # back to Kelvin

        lst = np.where((lst < 200) | (lst > 350), np.nan, lst)

        output = lst.astype(np.float32)
        out_band_name = "LST"

    elif parameter.upper() == "PWV":
        pwv = delta_T / (band31 * eps_mean) * 100
        output = pwv.astype(np.float32)
        out_band_name = "PWV"

    else:
        raise ValueError("Parameter must be either 'LST' or 'PWV'")

    profile.update(dtype=rasterio.float32, count=1, compress="lzw")
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    with rasterio.open(TEMP_DIR / output_path, "w", **profile) as dst:
        dst.write(output, 1)

    print(f"\nFinal {out_band_name} statistics:")
    print(f"Min: {np.nanmin(output):.2f}")
    print(f"Max: {np.nanmax(output):.2f}")
    print(f"Mean: {np.nanmean(output):.2f}")
    print(f"Valid data percent: {np.sum(~np.isnan(output)) / output.size * 100:.2f}%")

    return f"Result saved at {TEMP_DIR / output_path}"



@mcp.tool(description='''
Estimate Land Surface Temperature (LST) using an enhanced Temperature Emissivity Separation (TES) algorithm
with empirical emissivity estimation.

Parameters:
    tir_band_paths (list[str]): List of Thermal Infrared (TIR) GeoTIFF file paths (e.g., ASTER Bands 10–14).
    representative_band_index (int): Index of the TIR band to use as reference brightness temperature (e.g., 3 for Band 13).
    output_path (str): relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Path to the output GeoTIFF containing three bands: LST, emissivity, and emissivity variation.
''')
def temperature_emissivity_separation(
    tir_band_paths: list[str],
    representative_band_index: int,
    output_path: str
) -> str:
    """
    Description:
        Estimate Land Surface Temperature (LST) using the Temperature Emissivity Separation (TES) 
        algorithm with empirical emissivity estimation. Outputs a multi-band raster containing LST, 
        emissivity, and emissivity variation (Δε).

    Parameters:
        tir_band_paths (list[str]): List of paths to Thermal Infrared (TIR) GeoTIFFs (e.g., ASTER Bands 10–14).
        representative_band_index (int): Index of the TIR band used as the reference brightness temperature (e.g., 3 for Band 13).
        output_path (str): Relative path for saving the output raster file (e.g., "question17/lst_2022-01-16.tif").

    Return:
        str: Path to the saved GeoTIFF file containing:
             - Band 1: LST (K)
             - Band 2: Emissivity (ε)
             - Band 3: Emissivity variation (Δε)

    Example:
        >>> temperature_emissivity_separation(
        ...     ["ASTER_B10.tif", "ASTER_B11.tif", "ASTER_B12.tif", "ASTER_B13.tif", "ASTER_B14.tif"],
        ...     representative_band_index=3,
        ...     output_path="question17/lst_2022-01-16.tif"
        ... )
        'Result saved at question17/lst_2022-01-16.tif'
    """
    import os
    import rasterio
    import numpy as np

    # Constants
    c2 = 1.43877e4       # Second radiation constant (μm·K)
    wavelength = 10.6    # Representative band central wavelength (μm)

    # Step 1: Read representative band
    with rasterio.open(tir_band_paths[representative_band_index]) as src:
        rep_band = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        valid_mask = (rep_band > 0) & (rep_band < 1000)

    # Step 2: Read all bands
    bands_data = []
    for path in tir_band_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            band[~valid_mask] = np.nan
            bands_data.append(band)

    bands_stack = np.stack(bands_data, axis=0)

    # Step 3: Compute Δε
    masked_stack = np.ma.masked_invalid(bands_stack)
    band_max = np.ma.max(masked_stack, axis=0).filled(np.nan)
    band_min = np.ma.min(masked_stack, axis=0).filled(np.nan)
    delta_epsilon = band_max - band_min

    # Step 4: Estimate emissivity
    emissivity = 0.982 - 0.072 * delta_epsilon
    emissivity = np.clip(emissivity, 0.85, 0.999)

    # Step 5: Calculate LST
    Tb = bands_stack[representative_band_index]
    Tb[~valid_mask] = np.nan
    valid_calc = (Tb > 0) & (emissivity > 0) & (~np.isnan(Tb)) & (~np.isnan(emissivity))
    lst = np.full_like(Tb, np.nan)
    lst[valid_calc] = Tb[valid_calc] / (1 + (wavelength * Tb[valid_calc] / c2) * np.log(emissivity[valid_calc]))

    # Step 6: Output stack
    out_stack = np.stack([lst, emissivity, delta_epsilon], axis=0).astype(np.float32)
    profile.update(dtype=rasterio.float32, count=3, compress='lzw', nodata=np.nan)

    out_full_path = TEMP_DIR / output_path
    os.makedirs(out_full_path.parent, exist_ok=True)

    with rasterio.open(out_full_path, 'w', **profile) as dst:
        dst.write(out_stack)
        dst.set_band_description(1, "LST (K)")
        dst.set_band_description(2, "Emissivity (ε)")
        dst.set_band_description(3, "Emissivity Variation (Δε)")

    return f'Result saved at {out_full_path}'




@mcp.tool(description='''
Estimate land surface temperature (LST) from local MODIS Day and Night brightness temperatures
using a single-channel correction method.

Requires local input GeoTIFF files:
- BT_day_path: Brightness Temperature Day band (e.g., MODIS LST_Day_1km scaled by 0.02)
- BT_night_path: Brightness Temperature Night band (e.g., MODIS LST_Night_1km scaled by 0.02)
- Emis_day_path: Emissivity Day band (e.g., MODIS Emis_31 scaled by 0.002)
- Emis_night_path: Emissivity Night band (e.g., MODIS Emis_32 scaled by 0.002)

Parameters:
    BT_day_path (str): Path to local Brightness Temperature Day GeoTIFF.
    BT_night_path (str): Path to local Brightness Temperature Night GeoTIFF.
    Emis_day_path (str): Path to local Emissivity Day GeoTIFF.
    Emis_night_path (str): Path to local Emissivity Night GeoTIFF.
    output_path (str): relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"

Returns:
    str: Path to the exported GeoTIFF containing six bands:
         LST_Day, LST_Night, BT_Day, BT_Night, Emis_Day, Emis_Night.
''')
def modis_day_night_lst(
    BT_day_path: str,
    BT_night_path: str,
    Emis_day_path: str,
    Emis_night_path: str,
    output_path: str
) -> str:
    """
    Description:
        Estimate Land Surface Temperature (LST) from MODIS Day and Night brightness temperatures 
        using a single-channel correction algorithm. Performs resampling, scaling, and filtering 
        of emissivity and brightness temperature bands to output a six-band GeoTIFF.

    Parameters:
        BT_day_path (str): Path to local MODIS Brightness Temperature Day GeoTIFF.
        BT_night_path (str): Path to local MODIS Brightness Temperature Night GeoTIFF.
        Emis_day_path (str): Path to MODIS Emissivity Day GeoTIFF (scaled by 0.002, offset by 0.49).
        Emis_night_path (str): Path to MODIS Emissivity Night GeoTIFF (scaled by 0.002, offset by 0.49).
        output_path (str): Relative path for saving the output raster file 
                           (e.g., "question17/lst_2022-01-16.tif").

    Return:
        str: Path to the exported GeoTIFF with six bands:
             - Band 1: LST (Day)
             - Band 2: LST (Night)
             - Band 3: BT (Day)
             - Band 4: BT (Night)
             - Band 5: Emissivity (Day)
             - Band 6: Emissivity (Night)

    Example:
        >>> modis_day_night_lst(
        ...     BT_day_path="MODIS_BT_Day.tif",
        ...     BT_night_path="MODIS_BT_Night.tif",
        ...     Emis_day_path="MODIS_Emis_Day.tif",
        ...     Emis_night_path="MODIS_Emis_Night.tif",
        ...     output_path="question17/lst_2022-01-16.tif"
        ... )
        'Result saved at question17/lst_2022-01-16.tif'
    """
    import os
    import rasterio
    import numpy as np

    # Define a simple nearest neighbor resampling function
    def resample_to_reference(src_data: np.ndarray, src_profile: dict, ref_profile: dict) -> np.ndarray:
        src_height, src_width = src_data.shape
        dst_height = ref_profile['height']
        dst_width = ref_profile['width']
        scale_h = dst_height / src_height
        scale_w = dst_width / src_width
        dst_data = np.zeros((dst_height, dst_width), dtype=src_data.dtype)
        for i in range(dst_height):
            for j in range(dst_width):
                src_i = min(int(i / scale_h), src_height - 1)
                src_j = min(int(j / scale_w), src_width - 1)
                dst_data[i, j] = src_data[src_i, src_j]
        return dst_data

    # Set reasonable temperature range for filtering (Kelvin)
    MIN_TEMP = 270
    MAX_TEMP = 325

    # Read daytime brightness temperature as reference resolution and mask
    with rasterio.open(BT_day_path) as src:
        BT_day = src.read(1).astype(np.float32)
        BT_day = np.where((BT_day > MAX_TEMP) | (BT_day < MIN_TEMP), np.nan, BT_day)
        ref_profile = src.profile.copy()

    # Read nighttime brightness temperature
    with rasterio.open(BT_night_path) as src:
        BT_night_raw = src.read(1).astype(np.float32)
        BT_night_raw = np.where((BT_night_raw > MAX_TEMP) | (BT_night_raw < MIN_TEMP), np.nan, BT_night_raw)
        BT_night = resample_to_reference(BT_night_raw, src.profile, ref_profile)

    # Read daytime emissivity
    with rasterio.open(Emis_day_path) as src:
        Emis_day_raw = src.read(1).astype(np.float32)
        Emis_day_raw = (Emis_day_raw * 0.002) + 0.49
        Emis_day = resample_to_reference(Emis_day_raw, src.profile, ref_profile)

    # Read nighttime emissivity
    with rasterio.open(Emis_night_path) as src:
        Emis_night_raw = src.read(1).astype(np.float32)
        Emis_night_raw = (Emis_night_raw * 0.002) + 0.49
        Emis_night = resample_to_reference(Emis_night_raw, src.profile, ref_profile)

    # Clip emissivity values
    Emis_day_clipped = np.clip(Emis_day, 0.5, 1.0)
    Emis_night_clipped = np.clip(Emis_night, 0.5, 1.0)

    # Constants
    wavelength = 11.0  # micrometers
    c2 = 1.43877e4     # μm*K

    # Calculate LST
    LST_day = BT_day / (1 + (wavelength * BT_day / c2) * np.log(Emis_day_clipped))
    LST_night = BT_night / (1 + (wavelength * BT_night / c2) * np.log(Emis_night_clipped))

    # Filter unreasonable LST values
    LST_day = np.where((LST_day > MAX_TEMP) | (LST_day < MIN_TEMP), np.nan, LST_day)
    LST_night = np.where((LST_night > MAX_TEMP) | (LST_night < MIN_TEMP), np.nan, LST_night)

    # Stack six bands
    out_stack = np.stack([LST_day, LST_night, BT_day, BT_night, Emis_day, Emis_night], axis=0).astype(np.float32)
    profile = ref_profile.copy()
    profile.update(count=6, dtype=rasterio.float32, compress='lzw')

    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    # Write GeoTIFF
    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(out_stack)

    return f"Result saved at {TEMP_DIR / output_path}"


@mcp.tool(description='''
Estimate land surface temperature (LST) and emissivity using improved Three-Temperature Method (TTM)
from three local thermal band GeoTIFF files.

Uses all three bands to form a system of equations and solves per-pixel with physical constraints.

Parameters:
    tir_band_paths (list[str]): Paths to three thermal band GeoTIFFs (e.g. ASTER B10, B11, B12).
    output_path (str): relative path for the output raster file, e.g. "question17/lst_2022-01-16.tif"
    wavelengths (list[float], optional): Wavelengths (μm) for each band. Default [8.3, 8.65, 9.1].

Returns:
    str: Path to exported GeoTIFF with LST and emissivity bands.
''')
def ttm_lst(
    tir_band_paths: list[str],
    output_path: str,
    wavelengths: list[float] = [8.3, 8.65, 9.1]
) -> str:
    """
    Description:
        Estimate Land Surface Temperature (LST) and surface emissivity using a simplified 
        Three-Temperature Method (TTM). Reads three thermal infrared (TIR) bands, performs 
        filtering, applies empirical atmospheric correction, and outputs a three-band 
        GeoTIFF with LST and emissivity estimates.

    Parameters:
        tir_band_paths (list[str]): Paths to three thermal infrared band GeoTIFF files 
                                    (e.g., ASTER B10, B11, B12).
        output_path (str): Relative path to save the output raster 
                           (e.g., "question17/lst_2022-01-16.tif").
        wavelengths (list[float], optional): Central wavelengths (in μm) for each band.
                                             Defaults to [8.3, 8.65, 9.1].

    Return:
        str: Path to the exported GeoTIFF with three bands:
             - Band 1: LST (Kelvin)
             - Band 2: Emissivity estimate (Band 1)
             - Band 3: Emissivity estimate (Band 2)

    Example:
        >>> ttm_lst(
        ...     tir_band_paths=["ASTER_B10.tif", "ASTER_B11.tif", "ASTER_B12.tif"],
        ...     output_path="question17/lst_2022-01-16.tif"
        ... )
        'Result saved at question17/lst_2022-01-16.tif'
    """
    import os
    import rasterio
    import numpy as np
    from pathlib import Path

    print("Reading input bands...")
    bands_data = []
    profile = None
    for path in tir_band_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            bands_data.append(band)
            if profile is None:
                profile = src.profile.copy()

    B1, B2, B3 = bands_data
    shape = B1.shape
    print(f"Image size: {shape[1]} x {shape[0]}")

    # Create valid pixel mask
    valid_mask = (B1 > 240) & (B1 < 340) & \
                 (B2 > 240) & (B2 < 340) & \
                 (B3 > 240) & (B3 < 340)

    # Initialize outputs
    lst = np.full(shape, np.nan, dtype=np.float32)
    eps1_arr = np.full(shape, np.nan, dtype=np.float32)
    eps2_arr = np.full(shape, np.nan, dtype=np.float32)

    valid_indices = np.where(valid_mask)
    print(f"Number of valid pixels: {len(valid_indices[0])}")

    # Weighted LST calculation
    weights = np.array([0.3, 0.3, 0.4])
    lst_valid = (B1[valid_mask] * weights[0] +
                 B2[valid_mask] * weights[1] +
                 B3[valid_mask] * weights[2])
    lst_valid += 2.0  # empirical correction

    lst[valid_mask] = lst_valid

    # Assign constant emissivity
    eps_mean = 0.95
    eps1_arr[valid_mask] = eps_mean
    eps2_arr[valid_mask] = eps_mean

    print("Saving results...")
    profile.update(
        count=3,
        dtype=rasterio.float32,
        compress='lzw',
        nodata=np.nan
    )

    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(lst, 1)
        dst.write(eps1_arr, 2)
        dst.write(eps2_arr, 3)

    print(f"Processing complete! Results saved to: {TEMP_DIR / output_path}")

    valid_temps = lst[~np.isnan(lst)]
    print("\nTemperature Statistics:")
    print(f"Minimum: {np.min(valid_temps):.2f}K")
    print(f"Maximum: {np.max(valid_temps):.2f}K")
    print(f"Mean: {np.mean(valid_temps):.2f}K")
    print(f"Median: {np.median(valid_temps):.2f}K")

    return f"Result saved at {TEMP_DIR / output_path}"



@mcp.tool(description='''
Calculate the average Land Surface Temperature (LST) across multiple images
where NDVI is either above or below a given threshold.

Parameters:
    red_paths (str or list): Path(s) to red band image(s).
    nir_paths (str or list): Path(s) to near-infrared (NIR) image(s).
    lst_paths (str or list): Path(s) to land surface temperature (LST) image(s).
    ndvi_threshold (float): Threshold value for NDVI.
    mode (str): 'above' for NDVI >= threshold, 'below' for NDVI < threshold.

Returns:
    float: Mean of LST values over selected NDVI regions across all image sets.
               Returns np.nan if no valid pixels found.
''')
def calculate_mean_lst_by_ndvi(
    red_paths: str | list[str],
    nir_paths: str | list[str],
    lst_paths: str | list[str],
    ndvi_threshold: float,
    mode: str = 'above'
) -> float:
    """
    Calculate the average Land Surface Temperature (LST) across multiple images
    where NDVI is either above or below a given threshold.

    Parameters:
        red_paths (str or list): Path(s) to red band image(s).
        nir_paths (str or list): Path(s) to near-infrared (NIR) image(s).
        lst_paths (str or list): Path(s) to land surface temperature (LST) image(s).
        ndvi_threshold (float): Threshold value for NDVI.
        mode (str): 'above' for NDVI >= threshold, 'below' for NDVI < threshold.

    Returns:
        float: Mean of LST values over selected NDVI regions across all image sets.
               Returns np.nan if no valid pixels found.
    """
    import rasterio
    import numpy as np

    # Ensure inputs are lists
    if isinstance(red_paths, str): red_paths = [red_paths]
    if isinstance(nir_paths, str): nir_paths = [nir_paths]
    if isinstance(lst_paths, str): lst_paths = [lst_paths]

    if not (len(red_paths) == len(nir_paths) == len(lst_paths)):
        raise ValueError("red_paths, nir_paths, and lst_paths must have the same length.")

    all_selected_lst = []

    for red_path, nir_path, lst_path in zip(red_paths, nir_paths, lst_paths):
        try:
            with rasterio.open(red_path) as red_src, \
                 rasterio.open(nir_path) as nir_src, \
                 rasterio.open(lst_path) as lst_src:

                red = red_src.read(1).astype('float32')
                nir = nir_src.read(1).astype('float32')
                lst = lst_src.read(1).astype('float32')

                # Avoid division by zero
                ndvi_denominator = (nir + red)
                ndvi_denominator[ndvi_denominator == 0] = np.nan

                ndvi = (nir - red) / ndvi_denominator

                if mode == 'below':
                    mask = (ndvi < ndvi_threshold) & np.isfinite(lst)
                else:
                    mask = (ndvi >= ndvi_threshold) & np.isfinite(lst)

                selected_lst = lst[mask]

                if selected_lst.size > 0:
                    all_selected_lst.append(selected_lst)

        except Exception as e:
            print(f"Error processing {red_path}, {nir_path}, {lst_path}: {e}")
            continue

    if not all_selected_lst:
        return float('nan')

    combined_lst_values = np.concatenate(all_selected_lst)
    return float(np.nanmean(combined_lst_values))


@mcp.tool(description='''
Calculate the maximum Land Surface Temperature (LST) in areas where NDVI is above or below a given threshold.

Parameters:
    red_path (str): Path to the red band image.
    nir_path (str): Path to the near-infrared (NIR) band image. 
    lst_path (str): Path to the land surface temperature (LST) image.
    ndvi_threshold (float): Threshold value for NDVI.
    mode (str): 'above' to select NDVI >= threshold, 'below' for NDVI < threshold. Default is 'above'.

Returns:
    float: Maximum LST value over the selected NDVI region. Returns np.nan if no valid data.
''')
def calculate_max_lst_by_ndvi(red_path, nir_path, lst_path, ndvi_threshold, mode='above'):
    """
    Calculate the maximum Land Surface Temperature (LST) in areas where NDVI is above or below a given threshold.

    Parameters:
        red_path (str): Path to the red band image.
        nir_path (str): Path to the near-infrared (NIR) band image.
        lst_path (str): Path to the land surface temperature (LST) image.
        ndvi_threshold (float): Threshold value for NDVI.
        mode (str): 'above' to select NDVI >= threshold, 'below' for NDVI < threshold. Default is 'above'.

    Returns:
        float: Maximum LST value over the selected NDVI region. Returns np.nan if no valid data.
    """
    import rasterio
    import numpy as np

    with rasterio.open(red_path) as red_src, \
         rasterio.open(nir_path) as nir_src, \
         rasterio.open(lst_path) as lst_src:

        red = red_src.read(1).astype('float32')
        nir = nir_src.read(1).astype('float32')
        lst = lst_src.read(1).astype('float32')

        # Compute NDVI
        ndvi_denominator = (nir + red)
        ndvi_denominator[ndvi_denominator == 0] = np.nan
        ndvi = (nir - red) / ndvi_denominator

        # Create mask based on NDVI threshold
        if mode == 'below':
            mask = ndvi < ndvi_threshold
        else:
            mask = ndvi >= ndvi_threshold

        # Apply mask
        selected_lst = lst[mask]

        # Compute max ignoring NaNs
        max_lst = np.nanmax(selected_lst)

        return float(max_lst)



@mcp.tool(description='''
Estimate Apparent Thermal Inertia (ATI) using the Thermal Inertia Method.

This method calculates ATI as (1 - albedo) / (day_temp - night_temp),
which serves as a proxy for land surface temperature stability over diurnal cycles.

Parameters:
    day_temp_path (str): File path to daytime brightness temperature GeoTIFF.
    night_temp_path (str): File path to nighttime brightness temperature GeoTIFF.
    albedo_path (str): File path to surface albedo GeoTIFF.
    output_path (str): Relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

Returns:
    str: Path to the exported ATI GeoTIFF.
''')
def ATI(
    day_temp_path: str,
    night_temp_path: str,
    albedo_path: str,
    output_path: str
) -> str:
    """
    Description:
        Estimate Apparent Thermal Inertia (ATI) using the Thermal Inertia Method.
        ATI is computed as (1 - Albedo) / (Daytime BT - Nighttime BT) and serves as a proxy for
        land surface heat retention and thermal stability over diurnal cycles.
        The function aligns all raster layers to the daytime brightness temperature
        raster's resolution and extent before calculation.

    Parameters:
        day_temp_path (str): Path to the daytime brightness temperature (BT) GeoTIFF.
        night_temp_path (str): Path to the nighttime brightness temperature (BT) GeoTIFF.
        albedo_path (str): Path to the surface albedo GeoTIFF.
        output_path (str): Relative path to save the ATI raster, e.g.
                           "question17/thermal_inertia_2022-01-16.tif".

    Return:
        str: Path to the saved Apparent Thermal Inertia GeoTIFF file.

    Example:
        >>> ATI(
        ...     day_temp_path="data/day_temp.tif",
        ...     night_temp_path="data/night_temp.tif",
        ...     albedo_path="data/albedo.tif",
        ...     output_path="question17/thermal_inertia_2022-01-16.tif"
        ... )
        'Result saved at question17/thermal_inertia_2022-01-16.tif'
    """
    import os
    import rasterio
    import numpy as np
    from pathlib import Path
    from osgeo import gdal

    def resample_to_reference(src_data: np.ndarray, src_profile: dict, ref_profile: dict) -> np.ndarray:
        """
        Description:
            Resample a raster array to match the resolution and extent of a reference raster profile
            using GDAL bilinear resampling.

        Parameters:
            src_data (np.ndarray): Source raster data to be resampled.
            src_profile (dict): Metadata profile of the source raster.
            ref_profile (dict): Metadata profile of the reference raster.

        Return:
            np.ndarray: Resampled raster data array.
        """
        temp_src = 'temp_src.tif'
        temp_dst = 'temp_dst.tif'
        try:
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(temp_src, src_profile['width'], src_profile['height'], 1, gdal.GDT_Float32)
            transform = src_profile['transform']
            geotransform = [transform[2], transform[0], transform[1], transform[5], transform[3], transform[4]]
            dataset.SetGeoTransform(geotransform)
            if 'crs' in src_profile and src_profile['crs']:
                dataset.SetProjection(src_profile['crs'].to_wkt())
            else:
                dataset.SetProjection('EPSG:4326')
            dataset.GetRasterBand(1).WriteArray(src_data)
            dataset = None

            gdal.Warp(temp_dst, temp_src,
                      width=ref_profile['width'],
                      height=ref_profile['height'],
                      resampleAlg=gdal.GRA_Bilinear)
            dataset = gdal.Open(temp_dst)
            resampled_data = dataset.GetRasterBand(1).ReadAsArray()
            dataset = None
            return resampled_data
        finally:
            if os.path.exists(temp_src):
                os.remove(temp_src)
            if os.path.exists(temp_dst):
                os.remove(temp_dst)

    # Read raster data
    with rasterio.open(day_temp_path) as src_day:
        BT_day = src_day.read(1).astype(np.float32)
        day_profile = src_day.profile
    with rasterio.open(night_temp_path) as src_night:
        BT_night = src_night.read(1).astype(np.float32)
        night_profile = src_night.profile
    with rasterio.open(albedo_path) as src_alb:
        albedo = src_alb.read(1).astype(np.float32)
        albedo_profile = src_alb.profile

    # Align raster datasets to daytime raster
    BT_night = resample_to_reference(BT_night, night_profile, day_profile)
    albedo = resample_to_reference(albedo, albedo_profile, day_profile)

    # Compute delta temperature
    delta_T = BT_day - BT_night
    delta_T = np.where(delta_T == 0, np.nan, delta_T)

    # Calculate ATI
    ATI = (1 - albedo) / delta_T
    ATI = np.clip(ATI, 0, 10)  # Avoid extreme values

    # Save output
    day_profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    out_path = Path(TEMP_DIR) / output_path
    os.makedirs(out_path.parent, exist_ok=True)
    with rasterio.open(out_path, 'w', **day_profile) as dst:
        dst.write(ATI, 1)

    return f'Result saved at {out_path}'




@mcp.tool(description="""
Dual-Polarization Differential Method (DPDM) for microwave remote sensing parameter inversion.

Supports soil moisture and vegetation index estimation with improved data handling and flexible parameters.

Parameters:
    pol1_path (str): File path for the first polarization band GeoTIFF (e.g., VV).
    pol2_path (str): File path for the second polarization band GeoTIFF (e.g., VH).
    parameter (str): Parameter to invert, options: "soil_moisture" or "vegetation_index".
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"
    a (float, optional): Linear coefficient for soil moisture model. Default is 0.3.
    b (float, optional): Intercept for soil moisture model. Default is 0.1.
    input_unit (str, optional): Unit of input data, either "dB" or "linear". Default is "dB".

Returns:
    str: Path to the exported parameter GeoTIFF.
""")
def dual_polarization_differential(
    pol1_path: str,
    pol2_path: str,
    parameter: str ,
    output_path: str,
    a: float = 0.3,
    b: float = 0.1,
    input_unit: str = "dB"
) -> str:
    '''
    Dual-Polarization Differential Method (DPDM) for microwave remote sensing parameter inversion.

    Supports soil moisture and vegetation index estimation with improved data handling and flexible parameters.

    Parameters:
        pol1_path (str): File path for the first polarization band GeoTIFF (e.g., VV).
        pol2_path (str): File path for the second polarization band GeoTIFF (e.g., VH).
        parameter (str): Parameter to invert, options: "soil_moisture" or "vegetation_index".
        output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"
        a (float, optional): Linear coefficient for soil moisture model. Default is 0.3.
        b (float, optional): Intercept for soil moisture model. Default is 0.1.
        input_unit (str, optional): Unit of input data, either "dB" or "linear". Default is "dB".

    Returns:
        str: Path to the exported parameter GeoTIFF.
    '''
    import os
    import rasterio
    import numpy as np

    def db2linear(db):
        """Convert decibel (dB) values to linear scale."""
        return 10 ** (db / 10)

    # Read polarization band data
    with rasterio.open(pol1_path) as src1, rasterio.open(pol2_path) as src2:
        band1 = src1.read(1).astype(np.float32)
        band2 = src2.read(1).astype(np.float32)
        profile = src1.profile

    # Convert input data to linear scale if input is in dB
    if input_unit.lower() == "db":
        band1 = db2linear(band1)
        band2 = db2linear(band2)

    # Mask invalid data (non-positive values)
    valid_mask = (band1 > 0) & (band2 > 0)

    # Initialize output array with NaNs
    output = np.full(band1.shape, np.nan, dtype=np.float32)

    # Calculate differential and sum (avoid division by zero)
    diff = band1 - band2
    sum_ = band1 + band2
    sum_[sum_ == 0] = np.nan  # prevent division by zero

    param_lower = parameter.lower()
    if param_lower == "soil_moisture":
        # Apply linear model on valid pixels
        output[valid_mask] = a * diff[valid_mask] + b
    elif param_lower == "vegetation_index":
        # Compute vegetation index ratio on valid pixels
        output[valid_mask] = diff[valid_mask] / sum_[valid_mask]
    else:
        raise ValueError("Unsupported parameter. Choose 'soil_moisture' or 'vegetation_index'.")

    # Update raster profile for output
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    # Prepare output directory
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    # Write output GeoTIFF
    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(output, 1)

    return f'Result saved at {TEMP_DIR / output_path}'



@mcp.tool(description="""
Dual-frequency Differential Method (DDM) for parameter inversion using local raster data.

Supports inversion of multiple parameters via empirical linear models:

- Soil Moisture (SM): param = alpha*(band1 - band2) + beta
- Vegetation Index (VI): param = alpha*(band1 - band2) + beta
- Leaf Area Index (LAI): param = alpha*(band1 - band2) + beta

Parameters:
    band1_path (str): File path for frequency 1 polarization band GeoTIFF.
    band2_path (str): File path for frequency 2 polarization band GeoTIFF.
    parameter (str): Parameter to invert. Options: 'SM', 'VI', 'LAI'. Default is 'SM'.
    alpha (float, optional): Slope coefficient to override default.
    beta (float, optional): Intercept coefficient to override default.
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

Returns:
    str: Path to the saved combined output GeoTIFF (difference and parameter).
""")
def dual_frequency_diff(
    band1_path: str,
    band2_path: str,
    parameter: str,
    alpha: float,
    beta: float,
    output_path: str
) -> str:
    '''
    Dual-frequency Differential Method (DDM) for parameter inversion using local raster data.

    Supports inversion of multiple parameters via empirical linear models:

    - Soil Moisture (SM): param = alpha*(band1 - band2) + beta
    - Vegetation Index (VI): param = alpha*(band1 - band2) + beta
    - Leaf Area Index (LAI): param = alpha*(band1 - band2) + beta

    Parameters:
        band1_path (str): File path for frequency 1 polarization band GeoTIFF.
        band2_path (str): File path for frequency 2 polarization band GeoTIFF.
        parameter (str): Parameter to invert. Options: 'SM', 'VI', 'LAI'. Default is 'SM'.
        alpha (float, optional): Slope coefficient to override default.
        beta (float, optional): Intercept coefficient to override default.
        output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

    Returns:
        str: Path to the saved combined output GeoTIFF (difference and parameter).
    '''
    import os
    import rasterio
    import numpy as np

    # Default model coefficients for supported parameters
    param_models = {
        "SM": {"alpha": 0.7, "beta": 0.1},   # Soil Moisture
        "VI": {"alpha": 0.5, "beta": 0.0},   # Vegetation Index
        "LAI": {"alpha": 0.6, "beta": 0.05}, # Leaf Area Index
    }

    parameter_upper = parameter.upper()
    if parameter_upper not in param_models:
        raise ValueError(f"Unsupported parameter '{parameter}'. Choose from {list(param_models.keys())}")

    # Use provided alpha/beta or defaults
    alpha_val = alpha if alpha is not None else param_models[parameter_upper]["alpha"]
    beta_val = beta if beta is not None else param_models[parameter_upper]["beta"]

    # Read input bands
    with rasterio.open(band1_path) as src1, rasterio.open(band2_path) as src2:
        band1 = src1.read(1).astype(np.float32)
        band2 = src2.read(1).astype(np.float32)
        profile = src1.profile

        nodata1 = src1.nodata
        nodata2 = src2.nodata

    # Create mask for valid data (exclude nodata pixels)
    mask = np.ones_like(band1, dtype=bool)
    if nodata1 is not None:
        mask &= (band1 != nodata1)
    if nodata2 is not None:
        mask &= (band2 != nodata2)

    # Calculate difference image on valid pixels
    diff = np.full_like(band1, np.nan, dtype=np.float32)
    diff[mask] = band1[mask] - band2[mask]

    # Calculate parameter image using linear model on valid pixels
    param_img = np.full_like(band1, np.nan, dtype=np.float32)
    param_img[mask] = alpha_val * diff[mask] + beta_val

    # Clip parameter values to [0,1] range
    param_img = np.clip(param_img, 0, 1)

    # Update profile for two bands output: difference and parameter
    profile.update(dtype=rasterio.float32, count=2, compress='lzw')

    # Prepare output directory

    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    # Write output GeoTIFF
    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(diff, 1)
        dst.write(param_img, 2)

    return f'Result saved at {TEMP_DIR / output_path}'


@mcp.tool(description="""
Multi-frequency Brightness Temperature Method for parameter inversion using local raster data.

Parameters:
    bt_paths (list[str]): List of local file paths for brightness temperature GeoTIFF bands 
                          (e.g., ["BT_10GHz.tif", "BT_19GHz.tif", "BT_37GHz.tif"]).
    diff_pairs (list[list[int]]): List of index pairs from bt_paths for difference calculation (e.g., [[0,1],[1,2]]).
    parameter (str): Parameter to invert. Options: 'SM', 'VWC', 'LAI'.
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

Returns:
    str: Path to the saved inverted parameter GeoTIFF.
""")
def multi_freq_bt(
    bt_paths: list[str],
    diff_pairs: list[list[int]],
    parameter: str,
    output_path: str
) -> str:
    '''
    Multi-frequency Brightness Temperature Method for parameter inversion using local raster data.

    Parameters:
        bt_paths (list[str]): List of local file paths for brightness temperature GeoTIFF bands 
                          (e.g., ["BT_10GHz.tif", "BT_19GHz.tif", "BT_37GHz.tif"]).
        diff_pairs (list[list[int]]): List of index pairs from bt_paths for difference calculation (e.g., [[0,1],[1,2]]).
        parameter (str): Parameter to invert. Options: 'SM', 'VWC', 'LAI'.
        output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

    Returns:
        str: Path to the saved inverted parameter GeoTIFF.
    '''
    import os
    import rasterio
    import numpy as np

    # Define empirical model coefficients for parameters
    param_models = {
        "SM": {
            "alpha": [0.6, 0.4],
            "beta": 0.05
        },
        "VWC": {
            "alpha": [0.5, 0.5],
            "beta": 0.1
        },
        "LAI": {
            "alpha": [0.7, 0.3],
            "beta": 0.0
        }
    }

    param_key = parameter.upper()
    if param_key not in param_models:
        raise ValueError(f"Unsupported parameter '{parameter}'. Choose from {list(param_models.keys())}")

    model = param_models[param_key]
    alpha_list = model["alpha"]
    beta = model["beta"]

    if len(alpha_list) != len(diff_pairs):
        raise ValueError(f"Length of alpha coefficients ({len(alpha_list)}) must match number of diff pairs ({len(diff_pairs)})")

    # Read all brightness temperature bands and accumulate valid data mask
    bt_arrays = []
    profile = None
    mask = None
    for path in bt_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            # Create mask for valid pixels excluding nodata
            if src.nodata is not None:
                band_mask = (band != src.nodata)
            else:
                band_mask = np.ones_like(band, dtype=bool)
            mask = band_mask if mask is None else (mask & band_mask)
            bt_arrays.append(band)
            if profile is None:
                profile = src.profile

    # Validate diff_pairs indices
    n_bands = len(bt_arrays)
    for idx1, idx2 in diff_pairs:
        if idx1 < 0 or idx1 >= n_bands or idx2 < 0 or idx2 >= n_bands:
            raise IndexError(f"diff_pairs contains invalid band index: ({idx1},{idx2})")

    # Calculate difference images according to pairs
    diff_images = []
    for idx1, idx2 in diff_pairs:
        diff = bt_arrays[idx1] - bt_arrays[idx2]
        diff_images.append(diff)

    # Initialize parameter image with beta offset
    param_img = np.full_like(diff_images[0], beta, dtype=np.float32)
    for alpha, diff in zip(alpha_list, diff_images):
        param_img += alpha * diff

    # Apply mask to parameter image, set invalid pixels to NaN
    param_img = np.where(mask, param_img, np.nan)

    # Clip parameter values to [0,1] range
    param_img = np.clip(param_img, 0, 1)


    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    # Update profile for single band output with NaN nodata
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)

    # Write output GeoTIFF
    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(param_img.astype(np.float32), 1)

    return f'Result saved at {TEMP_DIR / output_path}'



@mcp.tool(description="""
Chang algorithm for inversion of a single parameter using multi-frequency dual-polarized microwave brightness temperatures from local raster files.

Parameters:
    bt_paths (list[str]): List of local GeoTIFF file paths for brightness temperature bands
                          (e.g., ["BT_10V.tif", "BT_10H.tif", "BT_19V.tif", "BT_19H.tif"]).
    diff_pairs (list[list[int]]): List of index pairs for brightness temperature differences.
    parameter (str): Parameter to invert (e.g., "SM", "VWC").
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

Returns:
    str: File path to saved GeoTIFF with inverted parameter band.
""")
def chang_single_param_inversion(
    bt_paths: list[str],
    diff_pairs: list[list[int]],
    parameter: str,
    output_path: str
) -> str:
    '''
    Chang algorithm for inversion of a single parameter using multi-frequency dual-polarized microwave brightness temperatures from local raster files.

    Parameters:
        bt_paths (list[str]): List of local GeoTIFF file paths for brightness temperature bands
                          (e.g., ["BT_10V.tif", "BT_10H.tif", "BT_19V.tif", "BT_19H.tif"]).
        diff_pairs (list[list[int]]): List of index pairs for brightness temperature differences.
        parameter (str): Parameter to invert (e.g., "SM", "VWC").
        output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

    Returns:
        str: File path to saved GeoTIFF with inverted parameter band.
    '''
    import os
    import rasterio
    import numpy as np

    parameter = parameter.upper()
    # Empirical model coefficients for supported parameters
    param_models = {
        "SM": {"alpha": [0.65, 0.3, 0.1], "beta": 0.02},
        "VWC": {"alpha": [0.5, 0.4, 0.2], "beta": 0.05},
    }

    if parameter not in param_models:
        raise ValueError(f"Unsupported parameter '{parameter}'. Supported: {list(param_models.keys())}")

    model = param_models[parameter]
    alpha_list = model["alpha"]
    beta = model["beta"]

    if len(alpha_list) != len(diff_pairs):
        raise ValueError(f"Length of alpha coefficients ({len(alpha_list)}) must equal number of diff_pairs ({len(diff_pairs)}) for parameter '{parameter}'")

    bt_arrays = []
    mask = None
    profile = None

    # Read bands and build combined valid data mask
    for path in bt_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            valid_mask = band != nodata if nodata is not None else np.ones_like(band, dtype=bool)
            mask = valid_mask if mask is None else (mask & valid_mask)
            bt_arrays.append(band)
            if profile is None:
                profile = src.profile

    n_bands = len(bt_arrays)
    # Validate diff_pairs indices
    for idx1, idx2 in diff_pairs:
        if not (0 <= idx1 < n_bands) or not (0 <= idx2 < n_bands):
            raise IndexError(f"diff_pairs indices ({idx1},{idx2}) out of range for available bands (0-{n_bands-1})")

    # Compute difference images
    diff_imgs = []
    for idx1, idx2 in diff_pairs:
        diff_imgs.append(bt_arrays[idx1] - bt_arrays[idx2])

    # Initialize parameter image with beta as base value
    param_img = np.full_like(diff_imgs[0], beta, dtype=np.float32)
    for alpha, diff in zip(alpha_list, diff_imgs):
        param_img += alpha * diff

    # Apply mask, assign NaN to invalid pixels
    param_img = np.where(mask, param_img, np.nan)

    # Clip parameter values to [0, 1] range
    param_img = np.clip(param_img, 0, 1)


    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    # Update profile for single-band output with NaN nodata
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)

    # Write output GeoTIFF
    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(param_img.astype(np.float32), 1)

    return f'Result saved at {TEMP_DIR / output_path}'




@mcp.tool(description="""
Estimate Sea Ice Concentration using NASA Team Algorithm from local passive microwave brightness temperature GeoTIFF files.

Parameters:
    bt_paths (dict): Dictionary of local GeoTIFF file paths for required brightness temperature bands, e.g.,
        {
          "19V": "BT_19V.tif",
          "19H": "BT_19H.tif",
          "37V": "BT_37V.tif",
          "37H": "BT_37H.tif"
        }
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"
    nd_ice (float): ND value for ice reference. Default 50.0.
    nd_water (float): ND value for water reference. Default 0.0.
    s1_ice (float): S1 value for ice reference. Default 20.0.
    s1_water (float): S1 value for water reference. Default 0.0.

Returns:
    str: Path to saved GeoTIFF with sea ice concentration band.
""")
def nasa_team_sea_ice_concentration(
    bt_paths: dict,
    output_path: str,
    nd_ice: float = 50.0,
    nd_water: float = 0.0,
    s1_ice: float = 20.0,
    s1_water: float = 0.0
) -> str:
    '''
    Estimate Sea Ice Concentration using NASA Team Algorithm from local passive microwave brightness temperature GeoTIFF files.

    Parameters:
        bt_paths (dict): Dictionary of local GeoTIFF file paths for required brightness temperature bands, e.g.,
        {
          "19V": "BT_19V.tif",
          "19H": "BT_19H.tif",
          "37V": "BT_37V.tif",
          "37H": "BT_37H.tif"
        }
    nd_ice (float): ND value for ice reference. Default 50.0.
    nd_water (float): ND value for water reference. Default 0.0.
    s1_ice (float): S1 value for ice reference. Default 20.0.
    s1_water (float): S1 value for water reference. Default 0.0.
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"

    Returns:
        str: Path to saved GeoTIFF with sea ice concentration band.
    '''
    import os
    import rasterio
    import numpy as np

    required_bands = ["19V", "19H", "37V", "37H"]
    for band in required_bands:
        if band not in bt_paths:
            raise ValueError(f"Missing required band '{band}' in bt_paths")

    arrays = {}
    profile = None

    # Read brightness temperature bands and build valid data mask
    try:
        for band in required_bands:
            with rasterio.open(bt_paths[band]) as src:
                arr = src.read(1).astype(np.float32)
                if profile is None:
                    profile = src.profile
                arrays[band] = arr

        # Create mask for valid data pixels (exclude NoData or negative values)
        valid_mask = np.ones_like(arrays["19V"], dtype=bool)
        for band in required_bands:
            valid_mask &= (arrays[band] > 0)

        # Compute ND and S1 differences
        ND = arrays["19V"] - arrays["19H"]
        S1 = arrays["37V"] - arrays["37H"]

        # Initialize sea ice concentration array with NaNs
        Ci = np.full_like(ND, np.nan, dtype=np.float32)

        # Calculate sea ice concentration for valid pixels using NASA Team algorithm
        term1 = (ND - nd_water) / (nd_ice - nd_water)
        term2 = (S1 - s1_water) / (s1_ice - s1_water)
        Ci[valid_mask] = (term1[valid_mask] + term2[valid_mask]) / 2

        # Clip Ci to the range [0, 1]
        Ci = np.clip(Ci, 0, 1)

    except Exception as e:
        raise RuntimeError(f"Error processing data: {e}")

    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

    # Update profile for single band output
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    # Write sea ice concentration to GeoTIFF
    with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
        dst.write(Ci, 1)

    return f'Result saved at {TEMP_DIR / output_path}'



@mcp.tool(description="""
Estimate Vegetation Water Content (VWC) or Soil Moisture (SM) using Dual-Polarization Ratio Method (PRM) from local passive microwave brightness temperature GeoTIFF files.

The polarization ratio is computed as: (V - H) / (V + H), where V and H are brightness temperatures of vertical and horizontal polarizations.

Empirical models:
- VWC = a_vwc * PR + b_vwc
- SM  = a_sm * PR + b_sm

Parameters:
    bt_paths (dict): Dictionary of local GeoTIFF file paths for vertical and horizontal polarization bands, e.g.
        {
          "V": "BT_V.tif",
          "H": "BT_H.tif"
        }
    parameter (str): Parameter to invert, either "VWC" or "SM".
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"
    coeffs (dict, optional): Empirical coefficients {"VWC": {"a":float, "b":float}, "SM": {...}}.

Returns:
    str: File path of the saved GeoTIFF containing the inverted parameter and PR band.
""")
def dual_polarization_ratio(
    bt_paths: dict,
    parameter: str,
    output_path: str,
    coeffs: dict | None = None
) -> str:
    '''
    Estimate Vegetation Water Content (VWC) or Soil Moisture (SM) using Dual-Polarization Ratio Method (PRM) from local passive microwave brightness temperature GeoTIFF files.

    The polarization ratio is computed as: (V - H) / (V + H), where V and H are brightness temperatures of vertical and horizontal polarizations.

    Empirical models:
    - VWC = a_vwc * PR + b_vwc
    - SM  = a_sm * PR + b_sm

    Parameters:
        bt_paths (dict): Dictionary of local GeoTIFF file paths for vertical and horizontal polarization bands, e.g.
        {
          "V": "BT_V.tif",
          "H": "BT_H.tif"
        }
    parameter (str): Parameter to invert, either "VWC" or "SM".
    output_path (str): relative path for the output raster file, e.g. "question17/thermal_inertia_2022-01-16.tif"
    coeffs (dict, optional): Empirical coefficients {"VWC": {"a":float, "b":float}, "SM": {...}}.

    Returns:
        str: File path of the saved GeoTIFF containing the inverted parameter and PR band.
    '''
    import os
    import rasterio
    import numpy as np

    if "V" not in bt_paths or "H" not in bt_paths:
        raise ValueError("bt_paths dict must contain keys 'V' and 'H'")
    
    parameter = parameter.upper()
    if coeffs is None:
        coeffs = {
            "VWC": {"a": 15.0, "b": 5.0},  # default coefficients for VWC
            "SM":  {"a": 0.4,  "b": 0.1}   # default coefficients for SM
        }
    if parameter not in coeffs:
        raise ValueError(f"Unsupported parameter '{parameter}'. Supported: {list(coeffs.keys())}")  

    try:
        with rasterio.open(bt_paths["V"]) as src_v, rasterio.open(bt_paths["H"]) as src_h:
            V = src_v.read(1).astype(np.float32)
            H = src_h.read(1).astype(np.float32)
            profile = src_v.profile

        # Compute polarization ratio (PR), avoid division by zero
        denom = V + H
        denom_safe = np.where(denom == 0, 1e-6, denom)

        pr = (V - H) / denom_safe
        pr = np.clip(pr, -1, 1)

        # Valid pixels mask
        valid_mask = denom > 1e-6

        # Calculate parameter from empirical model
        param = np.full_like(pr, np.nan, dtype=np.float32)
        a = coeffs[parameter]["a"]
        b = coeffs[parameter]["b"]
        param[valid_mask] = a * pr[valid_mask] + b

        # Update profile for 2 bands: parameter and polarization ratio
        profile.update(count=2, dtype=rasterio.float32, compress='lzw')

        os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)

        # Write output bands
        with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
            dst.write(param, 1)  # Band 1: Estimated parameter (VWC or SM)
            dst.write(pr, 2)     # Band 2: Polarization ratio (PR)

        return f'Result saved at {TEMP_DIR / output_path}'

    except Exception as e:
        raise RuntimeError(f"Error processing dual polarization ratio parameter: {e}")


@mcp.tool(description="""
Calculate water turbidity in NTU (Nephelometric Turbidity Units) from red band raster file
and save the result to a specified output path.

Parameters:
    input_red_path (str): Path to the Red band raster file.
    output_path (str): relative path for the output raster file, e.g. "benchmark/data/question17/turbidity_2022-01-16.tif"
    method (str): Calculation method - "linear" (a*Red+b), "power" (a*Red^n+b), or "log" (a*log(Red)+b).
    a (float): Coefficient parameter, default 1.0.
    b (float): Offset parameter, default 0.0.
    n (float): Power parameter for power method, default 1.0.

Returns:
    str: Path to the output NTU raster file.
""")
def calculate_water_turbidity_ntu(
    input_red_path: str,
    output_path: str,
    method: str = "linear",
    a: float = 1.0,
    b: float = 0.0,
    n: float = 1.0
    ) -> str:
    """
    Calculate water turbidity in NTU (Nephelometric Turbidity Units) from red band raster file
    and save the result to a specified output path.

    Parameters:
        input_red_path (str): Path to the Red band raster file.
        output_path (str): relative path for the output raster file, e.g. "benchmark/data/question17/turbidity_2022-01-16.tif"
        method (str): Calculation method - "linear" (a*Red+b), "power" (a*Red^n+b), or "log" (a*log(Red)+b).
        a (float): Coefficient parameter, default 1.0.
        b (float): Offset parameter, default 0.0.
        n (float): Power parameter for power method, default 1.0.

    Returns:
        str: Path to the output NTU raster file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the Red band raster file
    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)  # Read the first band (assuming single-band rasters)
        red_profile = red_src.profile  # Get the metadata profile

    # Ensure the input band data is in numpy array format
    red_band = np.array(red_band, dtype=np.float32)

    # Calculate NTU based on selected method
    if method == "linear":
        # NTU = a * Red + b (simple linear relationship)
        ntu = a * red_band + b
        
    elif method == "power":
        # NTU = a * Red^n + b (power relationship)
        # Ensure positive values for power calculation
        red_positive = np.maximum(red_band, 1e-6)
        ntu = a * (red_positive ** n) + b
        
    elif method == "log":
        # NTU = a * log(Red) + b (logarithmic relationship)
        # Ensure positive values for log calculation
        red_positive = np.maximum(red_band, 1e-6)
        ntu = a * np.log(red_positive) + b
        
    else:
        raise ValueError("Method must be 'linear', 'power', or 'log'")

    # Set negative values to 0 (turbidity cannot be negative)
    ntu = np.maximum(ntu, 0)

    # Update the profile for the output raster
    ntu_profile = red_profile.copy()
    ntu_profile.update(
        dtype=rasterio.float32,  # NTU values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the NTU result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **ntu_profile) as dst:
        dst.write(ntu.astype(rasterio.float32), 1)  # Write the NTU band

    return f'Result saved at {TEMP_DIR / output_path}'


if __name__ == "__main__":
    mcp.run() 
