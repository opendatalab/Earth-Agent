# Earth-Agent 工具参考手册

Earth-Agent 集成了约 130 个专业遥感分析工具，分为五大类。本文档提供每个工具的功能说明和使用指南。

## 一、遥感指数计算 (Index, 22 tools)

用于植被、水体、建筑等遥感指数的栅格计算。

### 植被指数

| 工具 | 功能 | 输入 |
|------|------|------|
| `calculate_ndvi` | 归一化植被指数 | NIR + Red 波段路径 |
| `calculate_batch_ndvi` | 批量 NDVI | 波段路径列表 |
| `calculate_evi` | 增强植被指数 | NIR + Red + Blue 波段 |
| `calculate_savi` | 土壤调节植被指数 | NIR + Red + L 因子 |
| `calculate_vari` | 可见光大气阻抗指数 | R + G + B 波段 |

### 水体/建筑指数

| 工具 | 功能 | 输入 |
|------|------|------|
| `calculate_ndwi` | 归一化水体指数 | NIR + SWIR |
| `calculate_batch_ndwi` | 批量 NDWI | 波段路径列表 |
| `calculate_ndbi` | 归一化建筑指数 | SWIR + NIR |
| `calculate_ndsi` | 归一化积雪指数 | Green + SWIR |

### 其他

| 工具 | 功能 |
|------|------|
| `calculate_mndwi` | 改进水体指数 |
| `calculate_bsi` | 裸土指数 |
| `calculate_ndmi` | 归一化水分指数 |
| `calculate_mndvi` | 改进 NDVI |

---

## 二、地表参数反演 (Inversion, 21 tools)

定量反演地表物理参数。

| 工具 | 功能 | 典型场景 |
|------|------|---------|
| `band_ratio` | 双波段比值计算 | 矿物识别 |
| `lst_single_channel` | 单通道地表温度反演 | Landsat 热红外 |
| `lst_multi_channel` | 多通道地表温度 | MODIS/ASTER |
| `split_window` | 劈窗算法 | MODIS LST |
| `albedo` | 反照率反演 | 地表能量平衡 |
| `emissivity` | 地表发射率估算 | NDVI 阈值法 |

---

## 三、图像感知处理 (Perception, 16 tools)

遥感图像预处理、目标检测辅助。

| 工具 | 功能 |
|------|------|
| `threshold_segmentation` | 阈值分割生成二值图 |
| `bbox_expansion` | 边界框按空间分辨率扩展 |
| `count_above_threshold` | 统计像素值超阈值数量 |
| `count_skeleton_contours` | 骨架提取 + 轮廓计数 |
| `bboxes2centroids` | 边界框 → 中心点坐标转换 |
| `calc_batch_image_mean` | 批量图像均值计算 |

---

## 四、时间序列分析 (Analysis, 10 tools)

时序统计分析与趋势检验。

| 工具 | 功能 | 输出 |
|------|------|------|
| `compute_linear_trend` | 线性趋势拟合（最小二乘） | 斜率 + 截距 + R² |
| `mann_kendall_test` | Mann-Kendall 趋势检验 | 统计量 + p-value |
| `sens_slope` | Sen's Slope 估算 | 稳健趋势值 |
| `stl_decompose` | 季节-趋势分解 (STL) | 趋势 + 季节 + 残差 |
| `detect_change_points` | 变点检测 (PELT) | 变点位置列表 |

---

## 五、统计计算 (Statistics, 61 tools)

基本统计量、分布检验、空间统计。

| 工具 | 功能 |
|------|------|
| `coefficient_of_variation` | 变异系数 |
| `skewness` | 偏度系数 |
| `kurtosis` | 峰度系数 |
| `calc_single_image_mean` | 单幅图像均值 |
| `calc_batch_image_mean` | 批量图像均值 |
| `correlation_matrix` | 相关系数矩阵 |
| `principal_components` | PCA 主成分分析 |
| `z_score_normalization` | Z-score 标准化 |
| `histogram_equalization` | 直方图均衡化 |

---

## 使用示例

### 计算 NDVI

```python
from tools.Index import calculate_ndvi

calculate_ndvi(
    input_nir_path="sentinel2_B08.tif",
    input_red_path="sentinel2_B04.tif",
    output_path="ndvi_result.tif"
)
```

### 地表温度反演

```python
from tools.Inversion import lst_single_channel

lst = lst_single_channel(
    thermal_band="landsat_B10.tif",
    emissivity=0.97,
    atmospheric_params={"tau": 0.85, "Lu": 1.2, "Ld": 0.8}
)
```

### 趋势检验

```python
from tools.Analysis import mann_kendall_test, sens_slope

# 输入: 时间序列 NDVI 值
ndvi_series = [0.32, 0.35, 0.31, 0.38, 0.40, 0.42]
trend, p_value = mann_kendall_test(ndvi_series)
slope = sens_slope(ndvi_series)
```

---

*本手册基于 Earth-Agent v1 源码自动提取，欢迎补充纠错。*
