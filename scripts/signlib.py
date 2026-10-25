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


# ---------- OCR 定位栏目标题（1_做标记.py 的 auto 模式和 3_自动标记.py 共用） ----------
def ocr_find_keyword_boxes_full(engine, img, keyword, verbose=True):
    """OCR 全图，返回 (命中标题框列表, 完整OCR结果)。
    命中框每项为 (x1,y1,x2,y2)；完整结果给 suggest_sign_box_hand 做手写定位用。"""
    result = engine(img)
    if result.boxes is None or result.txts is None:
        if verbose:
            print("    OCR 没有识别到任何文字")
        return [], result

    kw = keyword.replace(" ", "").replace(":", "").replace("：", "")
    hits = []
    for box, txt in zip(result.boxes, result.txts):
        t = (txt or "").replace(" ", "").replace(":", "").replace("：", "")
        if not t:
            continue
        if t == kw:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            hits.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    if not hits and verbose:
        print(f"    严格匹配 '{keyword}' 没有命中（标题可能被手写字盖住）")
    return hits, result


def ocr_find_keyword_boxes(engine, img, keyword, verbose=True):
    """OCR 全图，返回与关键词严格匹配的标题框列表，每项为 (x1,y1,x2,y2)。
    严格匹配 = 去掉空格冒号后完全相等，多一个字都不算。"""
    hits, _ = ocr_find_keyword_boxes_full(engine, img, keyword, verbose)
    return hits


# 宽松匹配时排除的别的栏目（含这些词的不是"工作许可人"签名栏）
_LOOSE_EXCLUDE = ("时间", "内容", "方式", "填写", "补充")


def filter_loose_boxes(ocr_result, keyword, verbose=True):
    """从已有 OCR 结果里筛宽松候选（不重复跑 OCR，省时间）。
    规则见 loose_keyword_boxes_full 的说明。返回候选框列表。"""
    import re
    result = ocr_result
    if result is None or result.boxes is None or result.txts is None:
        return []

    kw = keyword.replace(" ", "")
    cands = []
    for box, txt in zip(result.boxes, result.txts):
        if not txt:
            continue
        t = re.sub(r"[0-9\.\s:：()（）]", "", txt)   # 去掉编号、标点、空格
        # 双向包含：OCR 把"工作许可人"认成截断的"工作许可"（少末尾字）也算命中；
        # 但截断文本必须够长（比关键词少 2 字以内），防止"许可"这类短碎片混进来
        if kw not in t and (len(t) < len(kw) - 2 or t not in kw):
            continue
        if any(x in t for x in _LOOSE_EXCLUDE):
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cands.append([min(xs), min(ys), max(xs), max(ys)])

    # 合并：x 左缘接近（差<60）且 y 相邻（间隔<120）的候选，视为同一竖排标签被切碎
    cands.sort(key=lambda b: (round(b[0] / 60), b[1]))
    merged = []
    for b in cands:
        if merged:
            m = merged[-1]
            if abs(b[0] - m[0]) < 60 and b[1] - m[3] < 120:
                m[2] = max(m[2], b[2])
                m[3] = max(m[3], b[3])
                m[1] = min(m[1], b[1])
                continue
        merged.append(list(b))

    if not merged and verbose:
        print(f"    宽松匹配 '{keyword}' 也没有命中")
    return [tuple(int(v) for v in b) for b in merged]


def suggest_sign_box(title_box, img_shape):
    """根据栏目标题框生成建议签名区。
    按标题框宽高比分两种版式：
    - 横排标题（w>=h，表头列）：签名在正下方同宽单元格里
    - 竖排标题（w<h，竖排标签）：签名在标签左侧的格子里（竖排版式从右往左读）"""
    x1, y1, x2, y2 = title_box
    w, h = x2 - x1, y2 - y1
    H, W = img_shape[:2]

    if h > w:
        # 竖排标签：签名区在左侧 350px、y 与标签对齐（上下小幅扩展）
        bx1 = max(0, x1 - 350)
        bx2 = x1
        by1 = max(0, y1 - 60)
        by2 = min(y2 + 80, H - 5)
        return (bx1, by1, bx2, by2)

    sw = int(w * 1.6)                  # 宽度：标题宽 1.6 倍（略宽覆盖单元格边线）
    sh = max(int(h * 3.5), 200)        # 高度：往下拉长，盖住表头下方 1-3 行手写
    bx1 = max(0, x1 - int(w * 0.1))
    bx2 = min(bx1 + sw, W - 5)
    by1 = min(y2 + 3, H - 5)
    by2 = min(by1 + sh, H - 5)
    return (bx1, by1, bx2, by2)


def _is_handwritten(score):
    """OCR 置信度低于 0.88 视为手写字（印刷字一般 >0.95，手写认不准）"""
    return score is None or score < 0.88


def suggest_sign_box_hand(ocr_result, title_box, img_shape):
    """增强版签名区定位：用 OCR 低置信度框（手写字）精确锁定签名位置。
    - 竖排标题（h>w，竖排标签）：找标签左侧 200px 内、y 与标签对齐的手写框
      （只擦与标签对齐那段，避免误擦同一竖列里相邻栏目的签名）
    - 横排标题（w>=h，表头）：在标题上/下 450px 窗口内找手写框，
      哪边有手写签名区就在哪边（有的票签名在表头下方，有的在上方）
    - 都没找到手写 -> 退回 suggest_sign_box 的默认格
    ocr_result: RapidOCR 的完整结果（含 boxes/scores）"""
    tx1, ty1, tx2, ty2 = title_box
    H, W = img_shape[:2]
    vertical = (ty2 - ty1) > (tx2 - tx1)

    boxes, scores = ocr_result.boxes, ocr_result.scores
    if boxes is None or scores is None:
        return suggest_sign_box(title_box, img_shape)

    hw_boxes = []
    if vertical:
        win_y1, win_y2 = ty1 - 60, ty2 + 80     # 与标签 y 对齐的窗口
        for box, sc in zip(boxes, scores):
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
            if bx2 > tx1 + 20 or bx2 < tx1 - 200:   # 只收标签左侧紧邻 200px 内
                continue
            if by2 < win_y1 or by1 > win_y2:        # y 与标签不对齐，跳过
                continue
            if not _is_handwritten(sc):
                continue
            hw_boxes.append((bx1, max(by1, win_y1), bx2, min(by2, win_y2)))
        if not hw_boxes:
            return suggest_sign_box(title_box, img_shape)
        # 联合区域，x 不越过标签
        bx1 = max(0, min(b[0] for b in hw_boxes) - 15)
        bx2 = min(tx1, max(b[2] for b in hw_boxes) + 15)
        by1 = max(0, min(b[1] for b in hw_boxes) - 15)
        by2 = min(H - 5, max(b[3] for b in hw_boxes) + 15)
    else:
        # 横排：分别收集标题正下方/正上方窗口内的手写框。
        # 框必须与标题列 x 有交集（整行连片手写大框也行），错位框（旁边栏目的）跳过
        below, above = [], []
        for box, sc in zip(boxes, scores):
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
            if bx2 <= tx1 or bx1 >= tx2:            # 与标题列无交集，跳过
                continue
            if not _is_handwritten(sc):
                continue
            if ty2 + 3 <= by1 < ty2 + 450:          # 标题下方窗口
                below.append((bx1, by1, bx2, by2))
            elif ty1 - 450 < by2 <= ty1 - 3:        # 标题上方窗口
                above.append((bx1, by1, bx2, by2))
        hw_boxes = below if below else above        # 下方优先（表头列最常见）
        if not hw_boxes:
            return suggest_sign_box(title_box, img_shape)
        # x 裁剪到标题列 ±80px（整行大框只取标题列下方那段），y 用手写行范围
        bx1 = max(0, tx1 - 80)
        bx2 = min(W - 5, tx2 + 80)
        by1 = max(0, min(b[1] for b in hw_boxes) - 15)
        by2 = min(H - 5, max(b[3] for b in hw_boxes) + 15)

    if bx2 - bx1 < 20 or by2 - by1 < 20:
        return suggest_sign_box(title_box, img_shape)
    return (int(bx1), int(by1), int(bx2), int(by2))
