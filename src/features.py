"""OpenCV part: image decoding and nucleus feature extraction.

These features are shown next to the CNN result so a clinician can sanity-check it.
They are NOT fed into the CNN (the CNN was trained on raw pixels).
The thresholds are heuristics tuned for C-NMC style single-cell crops, so always
look at the segmentation overlay before trusting the numbers.
"""
import cv2
import numpy as np


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Bytes from an upload -> BGR image."""
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not read the image. Upload a PNG, JPG, BMP or TIFF file.")
    return bgr


def _contours(mask):
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[-2]          # works for both the 2-value and 3-value return styles


def analyze_cell(bgr: np.ndarray) -> dict:
    """Segment the nucleus and return shape features plus an overlay image."""
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    green = blur[:, :, 1]                                  # nuclei have the best contrast here
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Cell mask: C-NMC crops have a black background around the cell.
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    black_bg = float(border.mean()) < 20
    if black_bg:
        _, cell_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    else:
        cell_mask = np.full(gray.shape, 255, np.uint8)

    out = {"nucleus_area": None, "nucleus_circularity": None,
           "nucleus_solidity": None, "nc_ratio": None, "overlay": bgr.copy()}

    vals = np.ascontiguousarray(green[cell_mask > 0]).reshape(-1, 1)
    if vals.size < 200:
        return out

    t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nuc = ((green <= t) & (cell_mask > 0)).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_OPEN, k, iterations=2)
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_CLOSE, k, iterations=2)
    # Smooth the ragged edge caused by chromatin texture, otherwise perimeter (and so
    # circularity/solidity) is badly overestimated.
    nuc = cv2.GaussianBlur(nuc, (0, 0), 5)
    _, nuc = cv2.threshold(nuc, 127, 255, cv2.THRESH_BINARY)

    cnts = _contours(nuc)
    if black_bg:
        cell_cnts = _contours(cell_mask)
        if cell_cnts:
            cv2.drawContours(out["overlay"], cell_cnts, -1, (255, 200, 0), 1)   # cell outline
    if not cnts:
        return out

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 100:
        return out
    peri = float(cv2.arcLength(c, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(c))) or 1.0
    cv2.drawContours(out["overlay"], [c], -1, (0, 255, 0), 2)                   # nucleus outline

    out["nucleus_area"] = area
    out["nucleus_circularity"] = float(4 * np.pi * area / (peri ** 2 + 1e-6))   # 1.0 = perfect circle
    out["nucleus_solidity"] = area / hull_area                                  # 1.0 = convex
    if black_bg:
        cell_area = float(cv2.countNonZero(cell_mask))
        if cell_area > 0:
            out["nc_ratio"] = area / cell_area                                  # approximate N/C ratio
    return out
