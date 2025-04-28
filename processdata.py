import xarray as xr
import numpy as np
import cv2
from pathlib import Path

# 1. 读取数据
nc_path = Path("glo_3ice.nc")
ds = xr.open_dataset(nc_path, chunks={"time": 64})
ssha    = ds["zos_cglo"]       # (time, lat, lon)
sithick = ds["sithick_cglo"]   # (time, lat, lon)
times   = ds["time"].values    # datetime64[ns] array

# 2. 输出目录：训练 / 测试，各自的灰度图和标签
train_ssha = Path("train_ssha_png"); train_ssha.mkdir(exist_ok=True)
train_mask = Path("train_mask_png"); train_mask.mkdir(exist_ok=True)
test_ssha  = Path("test_ssha_png");  test_ssha.mkdir(exist_ok=True)
test_mask  = Path("test_mask_png");  test_mask.mkdir(exist_ok=True)

# 3. 判断“最后一年”起始时间
#    取最后一个时间点所属的年度（如 2010-01-01 → 2010），减 1 年即 2009-01-01
last_year_start = times[-1].astype("datetime64[Y]") - np.timedelta64(1, "Y")

# 4. 对称性判别函数
def is_symmetric(arr, thr=0.05):
    vmax, vmin = np.nanmax(arr), np.nanmin(arr)
    return abs(vmax + vmin) / max(abs(vmax), abs(vmin)) < thr

# 5. 批量生成并按时间划分
for t, tval in enumerate(times):
    # 根据时间决定是训练集还是测试集
    if tval >= last_year_start:
        ssha_dir = test_ssha
        mask_dir = test_mask
    else:
        ssha_dir = train_ssha
        mask_dir = train_mask

    # 5.1 生成 SSHA 灰度图
    arr = ssha.isel(time=t).values           # (lat, lon)
    if is_symmetric(arr):
        vm   = np.nanmax(np.abs(arr))
        norm = (arr + vm) / (2 * vm)
    else:
        anomaly = arr - np.nanmean(arr)
        vm      = np.nanmax(np.abs(anomaly))
        norm    = (anomaly + vm) / (2 * vm)
    norm = np.nan_to_num(norm, nan=0.0)
    gray = np.uint8(np.clip(norm * 255, 0, 255))

    # 放大到 512²
    if gray.shape[0] < 512 or gray.shape[1] < 512:
        gray = cv2.resize(gray, (512,512), interpolation=cv2.INTER_CUBIC)

    date_str = np.datetime_as_string(tval, unit="D")
    cv2.imwrite(str(ssha_dir / f"ssha_{date_str}.png"), gray)

    # 5.2 生成三分类标签
    land_mask = np.isnan(arr).astype(np.uint8)             # 1=陆地
    ice_mask  = (sithick.isel(time=t).values > 0).astype(np.uint8)  # 1=海冰

    # 放大到 512²
    land_mask = cv2.resize(land_mask, (512,512), interpolation=cv2.INTER_NEAREST)
    ice_mask  = cv2.resize(ice_mask,  (512,512), interpolation=cv2.INTER_NEAREST)

    label = np.zeros_like(land_mask, dtype=np.uint8)
    label[land_mask == 1] = 2
    label[(land_mask == 0) & (ice_mask == 1)] = 1

    cv2.imwrite(str(mask_dir / f"lbl_{date_str}.png"), label)
