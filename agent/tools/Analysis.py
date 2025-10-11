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
Description

Computes the linear trend (slope and intercept) of a time series by fitting a line of the form:

y = a \\cdot x + b

using the least squares method.

Parameters
    • y (list):
The dependent variable — typically your time series data.
    • x (list):
The independent variable — usually time indices. If not provided, the function will use np.arange(len(y)) as a default.

Returns
    • slope (float):
The coefficient a — represents the trend.
    • > 0: upward trend
    • < 0: downward trend
    • ≈ 0: no trend
    • intercept (float):
The y-intercept b of the fitted line.
''')
def compute_linear_trend(y: list, x: list|None = None):
    '''
    Description

    Computes the linear trend (slope and intercept) of a time series by fitting a line of the form:

    y = a \\cdot x + b

    using the least squares method.

    Parameters
        • y (list):
    The dependent variable — typically your time series data.
        • x (list):
    The independent variable — usually time indices. If not provided, the function will use np.arange(len(y)) as a default.

    Returns
        • slope (float):
    The coefficient a — represents the trend.
        • > 0: upward trend
        • < 0: downward trend
        • ≈ 0: no trend
        • intercept (float):
    The y-intercept b of the fitted line.
    '''
    import numpy as np
    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)

    if len(x) != len(y):
        raise ValueError("len(x) != len(y)")

    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return float(a), float(b)
    # return list(a), list(b)


@mcp.tool(description='''
Description:
Perform the non-parametric Mann-Kendall trend test on a univariate time series. 
The test evaluates whether there is a monotonic upward or downward trend 
without requiring the data to conform to any particular distribution.

Parameters:
- x (list[float]): Input time series values (numeric). Missing values should be removed before calling.

Returns:
- trend (str): Type of detected trend. One of:
    * "increasing" — statistically significant upward trend
    * "decreasing" — statistically significant downward trend
    * "no trend" — no statistically significant trend
- p_value (float): Two-tailed p-value of the test.
- z (float): Standard normal test statistic.
- tau (float): Kendall’s Tau statistic (rank correlation, between -1 and 1).
''')
def mann_kendall_test(x: list):
    """
    Description:
        Conduct the Mann-Kendall trend test on a time series to assess
        whether a statistically significant monotonic trend exists.
        The test is non-parametric and does not assume normality.
        Handles tied ranks with variance correction.

    Parameters:
        x (list[float]):
            The input time series data as a list of floats or ints.
            Any missing values (NaN) should be removed beforehand.

    Returns:
        trend (str):
            Type of detected trend:
              - "increasing" if a significant upward trend is found
              - "decreasing" if a significant downward trend is found
              - "no trend" if no significant trend is detected
        p_value (float):
            Two-tailed p-value of the test.
        z (float):
            Standard normal test statistic.
        tau (float):
            Kendall’s Tau statistic (measure of rank correlation, range -1 to 1).

    Example:
        >>> mann_kendall_test([1, 2, 3, 4, 5])
        ('increasing', 0.03, 2.12, 0.9)
    """
    import numpy as np
    from scipy.stats import norm
    x = np.asarray(x)
    n = len(x)

    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k+1:] - x[k]))

    # Calculate the unique data ranks for variance correction
    unique_x, counts = np.unique(x, return_counts=True)
    g = len(counts)

    # Variance of S
    if n == len(np.unique(x)):
        var_s = (n*(n-1)*(2*n+5)) / 18
    else:
        var_s = (n*(n-1)*(2*n+5) - np.sum(counts*(counts-1)*(2*counts+5))) / 18

    # Compute Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # Two-tailed p-value
    p = 2 * (1 - norm.cdf(abs(z)))

    # Kendall's Tau
    tau = s / (0.5 * n * (n - 1))

    # Determine trend
    alpha = 0.05  # significance level
    if p < alpha:
        trend = "increasing" if z > 0 else "decreasing"
    else:
        trend = "no trend"

    return trend, float(p), float(z), float(tau)


@mcp.tool(description='''
Description:
Compute Sen’s Slope estimator for a univariate time series. 
Sen’s Slope is a robust non-parametric method for estimating 
the median rate of change over time, often used with the 
Mann-Kendall test to assess both trend and magnitude.

Parameters:
- x (list[float]): Input time series values (numeric). 
                   Must contain at least two observations.

Returns:
- slope (float): Sen’s Slope estimate (the median of all pairwise slopes).
- slopes (list[float]): List of all individual pairwise slopes, 
                        useful for optional further analysis.
''')
def sens_slope(x: list):
    """
    Description:
        Compute Sen’s Slope estimator for a univariate time series.
        This robust non-parametric method calculates the median of
        all pairwise slopes between observations, providing an estimate
        of the overall monotonic trend magnitude.

    Parameters:
        x (list[float]):
            The input time series data as a list of floats or ints.
            Must have at least two data points.

    Returns:
        slope (float):
            The Sen’s Slope estimate (median of all pairwise slopes).
        slopes (list[float]):
            List of all pairwise slopes, which can be used for
            further distributional or variability analysis.

    Example:
        >>> sens_slope([2, 4, 6, 8, 10])
        (2.0, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    """
    import numpy as np
    x = np.asarray(x)
    n = len(x)

    if n < 2:
        raise ValueError("At least two data points are required.")

    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slope_ij = (x[j] - x[i]) / (j - i)
            slopes.append(slope_ij)

    slopes = np.array(slopes)
    median_slope = np.median(slopes)

    return float(median_slope), [float(s) for s in slopes]



@mcp.tool(description='''
Description:
Apply Seasonal-Trend decomposition using LOESS (STL) to a univariate time series. 
Decomposes the series into trend, seasonal, and residual components.

Parameters:
- x (list[float]): Input time series values (numeric).
- period (int): Number of observations per cycle (e.g., 12 for monthly data with yearly seasonality).
- robust (bool, optional): Whether to use a robust version of STL (less sensitive to outliers). Default = True.

Returns:
- result (dict): Dictionary with three keys:
    * "trend" (list[float]): Estimated long-term trend component.
    * "seasonal" (list[float]): Estimated seasonal component.
    * "resid" (list[float]): Residual (remainder) component.
''')
def stl_decompose(
    x: list, 
    period: int, 
    robust: bool = True
):
    """
    Description:
        Apply STL (Seasonal-Trend decomposition using LOESS) to a univariate time series.
        This method decomposes the input data into three additive components:
        - trend
        - seasonal
        - residual

    Parameters:
        x (list[float]):
            Input time series values. Will be internally converted to a pandas Series.
        period (int):
            The number of observations in one seasonal cycle.
            Example: 12 for monthly data with yearly seasonality.
        robust (bool, default=True):
            If True, use the robust version of STL, which is less sensitive to outliers.

    Returns:
        dict:
            A dictionary containing three keys with list outputs:
              - "trend" (list[float]): Long-term trend component
              - "seasonal" (list[float]): Seasonal cycle component
              - "resid" (list[float]): Residual (noise) component

    Example:
        >>> stl_decompose([10, 12, 15, 20, 18, 16, 14, 13, 11, 10, 9, 8], period=12)
        {
          "trend": [...],
          "seasonal": [...],
          "resid": [...]
        }
    """
    import pandas as pd
    from statsmodels.tsa.seasonal import STL
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    stl = STL(x, period=period, robust=robust)
    result = stl.fit()
    # Convert STL result components to regular Python lists
    return {
        'trend': [float(x) for x in result.trend],
        'seasonal': [float(x) for x in result.seasonal], 
        'resid': [float(x) for x in result.resid]
    }


@mcp.tool(description='''
Description:
Detect structural change points in a univariate time series using the 
ruptures library with the PELT algorithm. A change point marks a location 
where the statistical properties of the signal shift (e.g., mean or variance).

Parameters:
- signal (list[float]): Input 1D time series data.
- model (str, optional): Cost model type for segmentation. 
                         Options include "l1", "l2" (default; mean shift), "rbf", etc.
- penalty (float, optional): Penalty value that controls sensitivity. 
                             Higher penalty = fewer detected change points. Default = 10.

Returns:
- change_points (list[int]): Indices where change points are detected. 
                             Includes the last index of the signal by default.
''')
def detect_change_points(signal, model="l2", penalty=10):
    """
    Description:
        Detect change points in a one-dimensional time series using the 
        PELT algorithm from the ruptures library. This identifies indices 
        where the statistical structure of the signal changes.

    Parameters:
        signal (list[float]):
            Input time series data.
        model (str, default="l2"):
            Segmentation cost model to use.
            - "l1": robust to outliers (absolute loss)
            - "l2": mean shift model (squared loss, default)
            - "rbf": kernel-based model for nonlinear changes
            - others supported by ruptures
        penalty (float, default=10):
            Penalty value controlling sensitivity.
            Higher values detect fewer change points (more conservative).

    Returns:
        change_points (list[int]):
            List of indices marking change points in the series.
            The final index of the signal is always included.

    Example:
        >>> detect_change_points([1, 1, 1, 10, 10, 10, 2, 2, 2], model="l2", penalty=5)
        [3, 6, 9]
    """
    import numpy as np
    import ruptures as rpt

    signal = np.asarray(signal)
    algo = rpt.Pelt(model=model).fit(signal)
    change_points = algo.predict(pen=penalty)
    return [int(cp) for cp in change_points]


@mcp.tool(description='''
Description:
Compute the Autocorrelation Function (ACF) for a univariate time series. 
The ACF measures the correlation of the series with its own lags, which is 
useful for detecting seasonality, persistence, and lag dependence.

Parameters:
- x (list[float]): Input time series data.
- nlags (int, optional): Number of lags to compute. Default = 20.

Returns:
- acf (list[float]): List of autocorrelation values from lag 0 up to lag `nlags`. 
                     Value at lag 0 is always 1.0.
''')
def autocorrelation_function(x: list, nlags: int = 20):
    """
    Description:
        Compute the Autocorrelation Function (ACF) of a univariate time series.
        The ACF describes the correlation between the series and its lagged values,
        and is commonly used to detect seasonality or serial dependence.

    Parameters:
        x (list[float]):
            Input time series data as a list of floats or ints.
        nlags (int, default=20):
            The number of lags to compute autocorrelation for.

    Returns:
        acf (list[float]):
            List of autocorrelation values for lags 0 through `nlags`.
            - acf[0] = 1.0 (perfect correlation with itself)
            - acf[k] = correlation between series and series lagged by k steps

    Example:
        >>> autocorrelation_function([1, 2, 3, 4, 5], nlags=3)
        [1.0, 0.5, 0.0, -0.5]
    """
    import numpy as np
    x = np.asarray(x)
    x = x - np.mean(x)
    n = len(x)

    acf = np.empty(nlags + 1)
    var = np.dot(x, x) / n

    for lag in range(nlags + 1):
        if lag == 0:
            acf[lag] = 1.0  # autocorrelation at lag 0 is always 1
        else:
            acf[lag] = np.dot(x[:-lag], x[lag:]) / (n - lag) / var

    return [float(val) for val in acf]


@mcp.tool(description='''
Description:
Detect the dominant seasonality (period) in a univariate time series using the 
Autocorrelation Function (ACF). The method searches for significant peaks in the 
ACF beyond lag=1 to identify repeating cycles.

Parameters:
- values (list[float]): Input time series data.
- min_acf (float, optional): Threshold for ACF value to consider a lag significant. Default = 0.3.

Returns:
- result (int | str): 
    * Dominant period (lag) as an integer if a significant seasonal cycle is detected.
    * "Data is not cyclical" if no significant seasonality is found.
''')
def detect_seasonality_acf(values: list, min_acf: float = 0.3):
    """
    Description:
        Detect the dominant seasonality (cycle length) in a univariate time series
        using the Autocorrelation Function (ACF). A peak in the ACF beyond lag=1 
        indicates potential periodicity.

    Parameters:
        values (list[float]):
            Input time series data as a list of numeric values.
        min_acf (float, default=0.3):
            ACF threshold to consider a lag significant. 
            Lags with autocorrelation above this value are considered candidates.

    Returns:
        result (int | str):
            - An integer representing the dominant period (lag) if detected.
            - "Data is not cyclical" if no significant seasonality is found.

    Example:
        >>> detect_seasonality_acf([1, 2, 1, 2, 1, 2], min_acf=0.3)
        2

        >>> detect_seasonality_acf([1, 2, 3, 4, 5], min_acf=0.3)
        'Data is not cyclical'
    """
    import numpy as np
    from statsmodels.tsa.stattools import acf
    values = np.asarray(values)
    n = len(values)
    
    # Compute ACF up to 1/3 of the series length, capped at 40
    nlags = min(n // 3, 40)
    if nlags < 2:
        return "Data is not cyclical"
    
    try:
        acf_values = acf(values, nlags=nlags, fft=True)
    except:
        return "Data is not cyclical"
    
    # Search for the lag with the highest ACF value (excluding lag=0 and lag=1)
    max_acf = 0
    best_lag = None
    
    for lag in range(2, len(acf_values)):  # Start from lag=2 to avoid noise from lag=1
        if acf_values[lag] > max_acf and acf_values[lag] > min_acf:
            max_acf = acf_values[lag]
            best_lag = lag
    
    if best_lag is None:
        return "Data is not cyclical"
    else:
        return best_lag


@mcp.tool(description='''
Description:
Compute the Getis-Ord Gi* statistic for local spatial autocorrelation on a raster image. 
This method identifies statistically significant spatial clusters of high (hot spots) 
or low (cold spots) values using a user-specified spatial weight kernel.

Parameters:
- image_path (str): Path to the input single-band raster image (GeoTIFF).
- weight_matrix (list[list[float]]): 2D list representing the spatial weight kernel 
                                     (e.g., 3x3 matrix for neighborhood influence).
- output_path (str): Relative file path to save the Gi* result GeoTIFF 
                     (e.g., "question17/cloud_mask_2022-01-16.tif").

Returns:
- str: Path to the saved GeoTIFF image containing computed Gi* statistics.
       Output raster has the same dimensions as the input image, 
       with float32 values representing local Gi* scores.
''')
def getis_ord_gi_star(image_path: str, weight_matrix: list, output_path: str) -> str:
    """
    Description:
        Compute the Getis-Ord Gi* statistic for a single-band raster image 
        to identify local spatial autocorrelation. 
        Positive Gi* values indicate hot spots (clusters of high values), 
        while negative values indicate cold spots (clusters of low values).

    Parameters:
        image_path (str):
            Path to the input single-band raster image (GeoTIFF).
        weight_matrix (list[list[float]]):
            2D list representing the spatial weight kernel 
            (e.g., 3x3 or 5x5 neighborhood weights). 
            The sum of weights must not be zero.
        output_path (str):
            Relative file path to save the Gi* statistic result as GeoTIFF 
            (e.g., "question17/cloud_mask_2022-01-16.tif").

    Returns:
        str:
            Path to the saved GeoTIFF image containing computed Gi* statistics.
            - Output is float32 raster.
            - Preserves georeference and projection metadata from the input if available.

    Example:
        >>> getis_ord_gi_star(
        ...     image_path="ndvi_2022-01-16.tif",
        ...     weight_matrix=[[1,1,1],[1,0,1],[1,1,1]],
        ...     output_path="outputs/gi_star_ndvi.tif"
        ... )
        'Result save at outputs/gi_star_ndvi.tif'
    """
    import numpy as np
    from osgeo import gdal
    from scipy.ndimage import convolve
    
    
    ds = gdal.Open(image_path)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {image_path}")
    img = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)

    # Handle nodata
    img[np.isnan(img)] = 0

    # Convert weight list to array
    W = np.array(weight_matrix, dtype=np.float64)
    W_sum = np.sum(W)
    if W_sum == 0:
        raise ValueError("Sum of weights must not be zero.")

    # Basic statistics
    x_bar = np.mean(img)
    s = np.std(img)
    n = img.size

    # Numerator: ∑j wij xj (using convolution)
    numerator = convolve(img, W, mode='constant', cval=0)

    # Denominator: s * sqrt([ (n * ∑j wij² - (∑j wij)²) / (n - 1) ])
    W_sq_sum = np.sum(W ** 2)
    denom_part = (n * W_sq_sum - W_sum ** 2) / (n - 1)
    denominator = s * np.sqrt(denom_part)

    # Gi* computation
    gi_star = (numerator - x_bar * W_sum) / denominator


    (TEMP_DIR / output_path).parent.mkdir(parents=True, exist_ok=True)
    # Save as float32 TIFF without georeference
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        TEMP_DIR / output_path,
        xsize=gi_star.shape[1],
        ysize=gi_star.shape[0],
        bands=1,
        eType=gdal.GDT_Float32
    )
    out_ds.GetRasterBand(1).WriteArray(gi_star.astype(np.float32))

    # Copy georeference if available
    if ds.GetGeoTransform():
        out_ds.SetGeoTransform(ds.GetGeoTransform())
    if ds.GetProjection():
        out_ds.SetProjection(ds.GetProjection())

    out_ds.FlushCache()
    out_ds = None

    return f'Result save at {TEMP_DIR / output_path}'



@mcp.tool(description='''
Description:
Analyze the main directional concentration of hotspots in a binary hotspot map. 
The function counts the number of hotspot pixels (value=1) in each cardinal direction 
relative to the map center, and returns the dominant direction.

Parameters:
- hotspot_map_path (str): Path to the binary hotspot map GeoTIFF. 
                          Hotspot pixels must be encoded as value=1.

Returns:
- str: Main direction of hotspot concentration, one of:
       * "north"
       * "south"
       * "east"
       * "west"
       * "no hotspots found" (if no hotspot pixels are present)
''')
def analyze_hotspot_direction(hotspot_map_path: str) -> str:
    """
    Description:
        Analyze the dominant direction of hotspot concentration in a binary hotspot map.
        The function computes the relative location of hotspot pixels (value=1) with
        respect to the raster center and determines which cardinal direction 
        (north, south, east, west) contains the majority of hotspots.

    Parameters:
        hotspot_map_path (str):
            Path to the binary hotspot map GeoTIFF file.
            Hotspot pixels should be encoded with value=1; all other values are ignored.

    Returns:
        str:
            - "north", "south", "east", or "west": The dominant hotspot direction.
            - "no hotspots found": Returned if the map contains no hotspot pixels.

    Example:
        >>> analyze_hotspot_direction("outputs/hotspot_map.tif")
        'north'
    """
    import rasterio
    import numpy as np
    with rasterio.open(hotspot_map_path) as src:
        hotspot_data = src.read(1)
    
    # Find all hotspot locations (pixels with value 1)
    hotspot_indices = np.where(hotspot_data == 1)
    
    if len(hotspot_indices[0]) == 0:
        return "no hotspots found"
    
    # Calculate image center point
    center_y = hotspot_data.shape[0] // 2
    center_x = hotspot_data.shape[1] // 2
    
    # Calculate direction of each hotspot relative to center
    directions = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
    
    for y, x in zip(hotspot_indices[0], hotspot_indices[1]):
        # Calculate relative position
        dy = center_y - y  # Image coordinate system has y-axis pointing down, so invert
        dx = x - center_x
        
        # Determine main direction (which component is larger)
        if abs(dy) > abs(dx):
            if dy > 0:
                directions['north'] += 1
            else:
                directions['south'] += 1
        else:
            if dx > 0:
                directions['east'] += 1
            else:
                directions['west'] += 1
    
    # Find direction with most hotspots
    max_direction = max(directions.keys(), key=lambda x: directions[x])
    return max_direction


@mcp.tool(description=
    """
    Count the number of upward spikes in a sequence of numerical values.

    A spike is defined as a positive difference between consecutive valid
    values greater than the given threshold.

    Parameters:
        values (list of float):
            Input sequence of values (can include None or NaN).
        spike_threshold (float):
            Minimum positive change required to count as a spike.
        verbose (bool):
            If True, prints details for each detected spike.

    Returns:
        int:
            Number of detected upward spikes.

    Example:
        >>> count_spikes_from_values([0.1, 0.15, 0.5, 0.55], spike_threshold=0.2)
        1
    """)
def count_spikes_from_values(values, spike_threshold=0.1, verbose=True):
    """
    Count the number of upward spikes in a sequence of numerical values.

    A spike is defined as a positive difference between consecutive valid
    values greater than the given threshold.

    Parameters:
        values (list of float):
            Input sequence of values (can include None or NaN).
        spike_threshold (float):
            Minimum positive change required to count as a spike.
        verbose (bool):
            If True, prints details for each detected spike.

    Returns:
        int:
            Number of detected upward spikes.

    Example:
        >>> count_spikes_from_values([0.1, 0.15, 0.5, 0.55], spike_threshold=0.2)
        1
    """
    import numpy as np

    # Convert to NumPy array and filter out invalid values
    values = np.array(values, dtype=np.float32)
    valid_indices = ~np.isnan(values)
    valid_values = values[valid_indices]

    if len(valid_values) < 2:
        if verbose:
            print("Warning: Insufficient valid data points for spike analysis")
        return 0

    spike_count = 0
    for i in range(1, len(valid_values)):
        diff = valid_values[i] - valid_values[i - 1]
        if diff > spike_threshold:
            spike_count += 1
            if verbose:
                print(f"Spike detected: {valid_values[i-1]:.4f} -> {valid_values[i]:.4f}, Δ = {diff:.4f}")

    if verbose:
        print(f"\nTotal number of spikes detected: {spike_count}")

    return spike_count


if __name__ == "__main__":
    mcp.run()
