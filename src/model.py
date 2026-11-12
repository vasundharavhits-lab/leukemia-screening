"""Two-model router with an input check.

Single cell on a black background -> EfficientNetB0 (binary Normal / Leukemia)
Wide-field blood smear            -> MobileNetV2 (4 classes, P(leukemia) = 1 - P(Benign))
Anything else                     -> rejected with a reason (no prediction, nothing saved)
"""
import json
import os

import cv2
import numpy as np
import tensorflow as tf

SINGLE = "EfficientNetB0 (single cell, black background)"
SMEAR = "MobileNetV2 (smear image, research prototype)"
# Limits for wide-field smears, set from the 489 real test images (real range in comments)
SMEAR_BRIGHT_MIN, SMEAR_BRIGHT_MAX = 170, 250   # real: 187 to 244
SMEAR_CORNER_MIN = 165                          # real: 185 to 244
SMEAR_STAIN_MIN = 0.008                         # real: 0.01 to 0.73
SMEAR_SHARP_MIN, SMEAR_SHARP_MAX = 40, 3000     # real: 61 to 1907
# Limits for single cells on black: a first guess, not measured yet
SINGLE_CORNER_MAX = 15
SINGLE_DARK_MIN, SINGLE_DARK_MAX = 0.30, 0.98
ENABLE_SINGLE = False
    # single-cell model is not validated yet; set True to turn it back on
SMEAR_ENABLED = True    # wide-field model not validated yet; True only after validation

def _load_cfg(path):
    with open(path) as f:
        return json.load(f)


def image_stats(bgr):
    """Simple measurements for the input check (always on a 224 x 224 copy)."""
    small = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
    k = 20
    corners = [gray[:k, :k], gray[:k, -k:], gray[-k:, :k], gray[-k:, -k:]]
    stained = (s > 50) & (h >= 100) & (h <= 175) & (v > 40)
    return {
        "corner_mean": max(float(c.mean()) for c in corners),
        "dark_fraction": float((gray < 20).mean()),
        "stained_fraction": float(stained.mean()),
        "mean_brightness": float(gray.mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def check_image(bgr):
    """Returns (route, None) if the image can be analyzed, or (None, reason) if not."""
    if bgr is None or bgr.ndim != 3 or min(bgr.shape[:2]) < 64:
        return None, "The image is missing or too small (at least 64 x 64 pixels are needed)."
    st = image_stats(bgr)

    # looks like a single cell on a black background
    if st["corner_mean"] < SINGLE_CORNER_MAX and st["dark_fraction"] > SINGLE_DARK_MIN:
                if st["dark_fraction"] > SINGLE_DARK_MAX:
                    return None, "The image is almost completely black. No cell was found. Not analyzed."
                if not ENABLE_SINGLE:
                    return None, ("This looks like a single-cell image. Single-cell analysis is switched "
                          "off because it has not been validated yet. Upload a wide-field "
                          "blood smear image instead. Not analyzed.")
                return SINGLE, None

    # otherwise it must look like a wide-field blood smear
    problems = []
    if not (SMEAR_BRIGHT_MIN <= st["mean_brightness"] <= SMEAR_BRIGHT_MAX):
        problems.append(f"brightness {st['mean_brightness']:.0f} is outside the smear range "
                        f"{SMEAR_BRIGHT_MIN}-{SMEAR_BRIGHT_MAX}")
    if st["corner_mean"] < SMEAR_CORNER_MIN:
        problems.append("the background (image corners) is too dark")
    if st["stained_fraction"] < SMEAR_STAIN_MIN:
        problems.append("no purple or blue stained cells were found")
    if not (SMEAR_SHARP_MIN <= st["sharpness"] <= SMEAR_SHARP_MAX):
        problems.append(f"sharpness {st['sharpness']:.0f} is outside the smear range "
                        f"{SMEAR_SHARP_MIN}-{SMEAR_SHARP_MAX} (flat or noisy image)")
    if problems:
        return None, ("Unable to analyze: this image does not look like the smear images the model was tested on: "
                      + "; ".join(problems) + ". Not analyzed.")
    return SMEAR, None


class LeukemiaModel:
    def __init__(self, model_path: str, config_path: str):
        self.models_dir = os.path.dirname(model_path) or "."
        # wide-field smear model (loaded now)
        self.mnv2 = tf.keras.models.load_model(model_path, compile=False)
        c = _load_cfg(config_path)
        self.cfg = {
            "smear": {"threshold": float(c["threshold"]),
                      "img_size": int(c.get("img_size", 224)),
                      "band": float(c.get("uncertain_band", 0.10))},
            "single": None,
        }
        self._bb = next(i for i, l in enumerate(self.mnv2.layers)
                        if isinstance(l, tf.keras.Model))
        # single-cell model (loaded the first time it is needed)
        self.eff = None
        self.last_route = SMEAR

    # ---------------------------------------------------------- single-cell model
    def _get_eff(self):
        if self.eff is None:
            mp = os.path.join(self.models_dir, "leukemia_effnet.keras")
            cp = os.path.join(self.models_dir, "model_config.json")
            if not (os.path.exists(mp) and os.path.exists(cp)):
                raise ValueError(
                    "This looks like a single-cell image, but the single-cell model files "
                    "(models/leukemia_effnet.keras and models/model_config.json) were not found.")
            self.eff = tf.keras.models.load_model(mp, compile=False)
            c = _load_cfg(cp)
            self.cfg["single"] = {"threshold": float(c["threshold"]),
                                  "img_size": int(c.get("img_size", 224)),
                                  "band": 0.10}
        return self.eff

    def _prep_eff(self, bgr):
        s = self.cfg["single"]["img_size"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = tf.image.resize(rgb, (s, s))          # same resize as that model's training
        return tf.cast(x, tf.float32).numpy()

    # ---------------------------------------------------------- smear model
    def _prep_smear(self, bgr):
        s = self.cfg["smear"]["img_size"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA)
        return rgb.astype("float32")              # the model scales pixels itself

    # ---------------------------------------------------------- interface used by app.py
    @property
    def _s(self):
        return self.cfg["single"] if self.last_route == SINGLE else self.cfg["smear"]

    @property
    def threshold(self):
        return self._s["threshold"]

    def predict(self, bgr: np.ndarray) -> float:
        route, reason = check_image(bgr)
        if route is None:
            raise ValueError(reason)              # app.py shows this as a red message
        self.last_route = route
        if route == SMEAR and not SMEAR_ENABLED:
            raise ValueError("Wide-field smear analysis is not validated yet. Not analyzed.")
        if route == SINGLE:
            m = self._get_eff()
            x = self._prep_eff(bgr)
            return float(m(x[None], training=False).numpy()[0][0])
        probs = self.mnv2(self._prep_smear(bgr)[None], training=False).numpy()[0]
        return float(1.0 - probs[0])

    def label(self, p: float) -> str:
        if p >= self.threshold:
            return "Leukemia (ALL)"
        return "Normal (HEM)" if self.last_route == SINGLE else "Benign (non-malignant)"

    def is_uncertain(self, p: float) -> bool:
        return abs(p - self.threshold) < self._s["band"]

    # ---------------------------------------------------------- Grad-CAM
    @staticmethod
    def _blend(bgr, cam, s):
        base = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (s, s))
        heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (s, s))), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(base, 0.6, heat, 0.4, 0)

    def gradcam_overlay(self, bgr: np.ndarray) -> np.ndarray:
        if self.last_route == SINGLE:
            return self._gradcam_single(bgr)
        return self._gradcam_smear(bgr)

    def _gradcam_single(self, bgr):
        m = self._get_eff()
        s = self.cfg["single"]["img_size"]
        x = tf.convert_to_tensor(self._prep_eff(bgr)[None], tf.float32)
        backbone = m.get_layer("efficientnetb0")
        with tf.GradientTape() as tape:
            conv = backbone(x, training=False)
            tape.watch(conv)
            h = m.get_layer("gap")(conv)
            h = m.get_layer("drop")(h, training=False)
            score = m.get_layer("out")(h)[:, 0]
        grads = tape.gradient(score, conv)
        w = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
        cam = cam / (cam.max() + 1e-8)
        return self._blend(bgr, cam, s)

    def _gradcam_smear(self, bgr):
        s = self.cfg["smear"]["img_size"]
        x = tf.convert_to_tensor(self._prep_smear(bgr)[None], tf.float32)
        layers = self.mnv2.layers
        i = self._bb
        for l in layers[:i]:                      # scaling layer (augmentation is off here)
            if isinstance(l, tf.keras.layers.InputLayer):
                continue
            x = l(x, training=False)
        with tf.GradientTape() as tape:
            conv = layers[i](x, training=False)
            tape.watch(conv)
            h = conv
            for l in layers[i + 1:]:              # pooling, dropout, dense
                h = l(h, training=False)
            score = 1.0 - h[:, 0]
        grads = tape.gradient(score, conv)
        w = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
        cam = cam / (cam.max() + 1e-8)
        return self._blend(bgr, cam, s)