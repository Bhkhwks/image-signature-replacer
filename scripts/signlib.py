# -*- coding: utf-8 -*-
"""
公共工具库：1_做标记.py 和 2_批量换签名.py 共用。
包含：中文路径读写图片、自动转正（EXIF 方向优先，无 EXIF 时用 OCR 判断）。

两个脚本必须用同一套读图逻辑，才能保证"做标记时的坐标"和"批处理时的坐标"完全一致。
"""
import os

import numpy as np
import cv2


# ---------- 中文路径读写图片（cv2.imread/imwrite 不认中文路径，必须绕） ----------
def imread_cn(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img


def imwrite_cn(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("图片编码失败: " + path)
    buf.tofile(path)


# ---------- 自动转正 ----------
def _fix_by_exif(path, img):
    """按照片 EXIF 方向信息转正。返回 (图, 是否已处理)"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            o = im.getexif().get(274)   # 274 = Orientation 标签
    except Exception:
        o = None
    if o == 3:
        return cv2.rotate(img, cv2.ROTATE_180), True
    if o == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True
    if o == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), True
    if o == 1:
        return img, True                # 明确记录为正，不用再猜
    return img, False                   # 没有 EXIF（微信转存图常见），交给 OCR


_OCR_ENGINE = None
def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr import RapidOCR
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _detect_upright_by_ocr(img):
    """无 EXIF 时的兜底：把图转 4 个方向各做一次完整 OCR（检测+识别），
    识别得分总和最高的方向就是正的。
    原理：正着放的印刷字"认得出"（识别得分高），倒立/侧躺的字"认不出"（得分低）。
    之前用"检测框数量"判断的教训：倒立的字照样能框出文字块（检测不管内容），
    所以 180 度上下颠倒在框数量上区分不开，必须用识别得分。"""
    engine = _get_ocr_engine()
    # 缩小到长边 1600 做识别（方向判断不需要全分辨率，速度快很多）
    h, w = img.shape[:2]
    scale = 1600.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img

    best_code, best_total = None, -1.0
    for code in (None, cv2.ROTATE_90_CLOCKWISE,
                 cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        r = small if code is None else cv2.rotate(small, code)
        result = engine(r, use_det=True, use_cls=False, use_rec=True)
        total = 0.0
        if result.boxes is not None and result.scores is not None:
            total = sum(float(s) for s in result.scores if s is not None)
        if total > best_total:
            best_total, best_code = total, code
    if best_code is not None and best_total > 0:
        return cv2.rotate(img, best_code)
    return img


def load_upright_image(path, flags=cv2.IMREAD_COLOR):
    """读图并自动转正。两个脚本统一用这个读原图，坐标才对得上。"""
    img = imread_cn(path, flags)
    if img is None:
        return None
    img, fixed = _fix_by_exif(path, img)
    if fixed:
        return img
    return _detect_upright_by_ocr(img)
