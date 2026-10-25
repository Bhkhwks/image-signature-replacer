# -*- coding: utf-8 -*-
"""
用途：一键批量"擦掉旧签名 + 贴上你的新签名"。

前置条件：
  1. venv 里已安装 opencv-python
  2. images/  里放原图
  3. masks/   里有标记图（先用 1_做标记.py 生成；
              单独标记 masks/图名.png 优先，否则用通用 masks/default_mask.png）
  4. my_sign/ 里放你手写新签名的照片（白纸黑字拍照/扫描，一张即可，
              优先用文件名 sign.png / sign.jpg）

用法（在本项目目录下，用 venv 的 python 运行）：
  python 2_批量换签名.py

流程：逐张图 -> 标记区填周边纸面色+细颗粒（不用 AI 模型） -> 新签名二值化裁边 ->
      缩放到旧签名区域大小 -> 贴到区域中心 -> 存 output/
"""
import glob
import os
import sys

import numpy as np
import cv2

from signlib import imread_cn, imwrite_cn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 脚本在 scripts/ 里，项目根取上一级
IMAGES_DIR = os.path.join(BASE, "images")    # 已由 0_批量转正.py 原地转正
MASKS_DIR = os.path.join(BASE, "masks")
SIGN_DIR = os.path.join(BASE, "my_sign")
OUTPUT_DIR = os.path.join(BASE, "output")

# migan 模型权重存放位置（必须在 iopaint 导入前设好，iopaint 在函数内延迟导入）
# 模型缓存在项目内 models 目录，换机器/分享给别人都能直接跑；已设过 XDG_CACHE_HOME 的用户按自己的来
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(BASE, "models"))

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------- 可调参数 ----------------
SIGN_WIDTH_RATIO = 0.8   # 新签名宽度 = 旧签名标记区宽度的比例（0.8 = 占八成）
OUTPUT_GRAY = False      # True = 成品统一转灰度（黑白需求）；False = 保持原样彩色
ERASE_METHOD = "ai"      # "ai"  = 双模型(migan+zits)x多档观察范围全试跑，自动选残留最少的结果
                          # "fill" = 纯填色（不依赖模型，搬周边纸面盖掉）
# -----------------------------------------


# ---------- 新签名预处理：照片 -> 纯黑白字形的透明通道 ----------
def load_sign_alpha():
    """返回 (字形alpha图, 高, 宽)：字的地方=255，空白=0，已裁掉四周空白"""
    if not os.path.isdir(SIGN_DIR):
        return None
    # 优先找 sign.png/sign.jpg，否则拿文件夹里第一张图
    candidates = ([os.path.join(SIGN_DIR, "sign" + e) for e in (".png", ".jpg", ".jpeg")]
                  + sorted(glob.glob(os.path.join(SIGN_DIR, "*"))))
    sign_path = next((p for p in candidates
                      if os.path.isfile(p)
                      and os.path.splitext(p)[1].lower() in EXTS), None)
    if sign_path is None:
        return None
    print(f"新签名素材: {os.path.basename(sign_path)}")

    img = imread_cn(sign_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("签名图读取失败，请确认是有效的图片文件")
        sys.exit(1)

    # OTSU 自动算门槛二值化：白纸黑字照片 -> 反转后"字=255"
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 3)  # 去孤立噪点

    ys, xs = np.where(bw > 0)
    if len(ys) == 0:
        print("签名图里没检测到笔画（可能拍的是白纸），请重新拍照")
        sys.exit(1)
    bw = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # 裁掉四周空白
    return bw


# ---------- 签名风格匹配：笔画粗细和墨色深度向图上原有手写字看齐 ----------
def _col_runs(bin_img):
    """逐列统计连续白像素段的长度（= 笔画竖向厚度，反映粗细）"""
    runs = []
    for col in bin_img.T:
        d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        runs.extend((ends - starts).tolist())
    return runs


def measure_stroke_style(gray, mask):
    """量擦除区周边的手写字风格，返回 (笔画平均灰度, 笔画中位竖向厚度)。
    采样范围：擦除区 bbox 外扩 260px 的环带，排除擦除区+80px 边。
    笔画阈值自适应：先估环带纸面亮度（亮像素中位数），比纸面暗 60 以上算笔画
    （固定阈值在这批整体偏暗的票上会把纸面全判成笔画）。
    连通域里长宽比>=8 的细长条是表格线要排除，太小的(面积<12)是噪点。
    量不到时返回 None（调用方跳过匹配）。"""
    H, W = gray.shape[:2]
    m = mask > 127
    ys_, xs_ = np.where(m)
    if len(ys_) == 0:
        return None
    bx0, by0 = xs_.min(), ys_.min()
    bx1, by1 = xs_.max() + 1, ys_.max() + 1

    band = np.zeros_like(m)
    y0, y1 = max(0, by0 - 260), min(H, by1 + 260)
    x0, x1 = max(0, bx0 - 260), min(W, bx1 + 260)
    band[y0:y1, x0:x1] = True
    band &= ~cv2.dilate((m * 255).astype(np.uint8),
                        np.ones((81, 81), np.uint8)).astype(bool)   # 排除擦除区+80px边

    roi_gray = gray[band]
    bright = roi_gray[roi_gray > 120]
    paper = float(np.median(bright)) if len(bright) > 500 else 190.0
    dark = ((gray < paper - 60) & band).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    grays, runs = [], []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 12:                       # 噪点
            continue
        if max(w, h) / max(1, min(w, h)) >= 8:   # 细长条 = 表格线
            continue
        comp = labels[y:y + h, x:x + w]
        sel = comp == k
        grays.append(float(gray[y:y + h, x:x + w][sel].mean()))
        runs.extend(_col_runs(sel))         # 列行程 = 竖向厚度，不受笔画长短影响
    if not grays or not runs or len(runs) < 8:
        return None
    return (float(np.mean(grays)), float(np.median(runs)))


def style_match_alpha(sign_alpha, style):
    """把新签名笔画调到与图上原有手写字相似：先调粗细（形态学膨胀/腐蚀），
    再调墨色（alpha 按灰度比缩放）。style = measure_stroke_style 的返回值。
    行程长度统一按"逐列扫描竖向厚度"统计（笔画横竖都有，竖扫量的是笔画厚度，
    不受笔画长短影响）。"""
    if style is None:
        return sign_alpha
    ref_gray, ref_run = style

    # ---- 1) 粗细：我方笔画竖向厚度 vs 对手中位厚度，差 1px 以上才动 ----
    runs = _col_runs(sign_alpha > 127)
    if runs:
        my_run = float(np.median(runs))
        diff = ref_run - my_run
        kern = int(round(abs(diff)))
        # 安全上限：膨胀最多 4px、腐蚀最多 3px，防止极端值把签名吃没或糊成团
        kern = min(kern, 4 if diff > 0 else 3)
        if kern >= 1:
            k = np.ones((kern, kern), np.uint8)
            if diff > 0:                    # 对手更粗 -> 膨胀加粗
                sign_alpha = cv2.dilate(sign_alpha, k)
            else:                           # 对手更细 -> 腐蚀收细
                sign_alpha = cv2.erode(sign_alpha, k)
            print(f"    笔画粗细匹配: 对手中位行程 {ref_run:.1f}px / 我方 {my_run:.1f}px -> "
                  f"{'加粗' if diff > 0 else '收细'} {kern}px")

    # ---- 2) 墨色深度：对手笔画平均灰度 vs 我方纯黑，按比例缩放 alpha ----
    # 对手灰度 110（浅字）时，alpha 乘 110/255 后峰值=110：贴出的字深浅与对手一致
    # 注意用 cv2.multiply 的 scale 参数（结果四舍五入、alpha>0 的笔画不会被截成 0）
    ratio = ref_gray / 255.0
    if 0.4 <= ratio < 0.98:
        sign_alpha = cv2.multiply(sign_alpha, 1.0, scale=ratio).astype(np.uint8)
        print(f"    墨色匹配: 对手笔画灰度 {ref_gray:.0f} -> alpha 缩放 {ratio:.2f}")
    return sign_alpha


# ---------- 贴签名：软边混合，把黑色字形融进底图 ----------
def paste_sign(base, sign_alpha, cx, cy, target_w):
    h0, w0 = sign_alpha.shape
    target_w = max(8, int(target_w))
    target_h = max(8, int(h0 * target_w / w0))
    alpha = cv2.resize(sign_alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 贴图左上角坐标（目标点为中心），越界部分裁掉
    x0 = cx - target_w // 2
    y0 = cy - target_h // 2
    H, W = base.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1 = target_w - max(0, x0 + target_w - W)
    sy1 = target_h - max(0, y0 + target_h - H)
    if sx1 <= sx0 or sy1 <= sy0:
        return base  # 完全在图外，不贴
    x0, y0 = max(0, x0), max(0, y0)
    a = alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0

    roi = base[y0:y0 + a.shape[0], x0:x0 + a.shape[1]]
    if base.ndim == 3:
        a3 = a[..., None]                       # 彩色图：三通道同比例混合
        blended = (roi * (1 - a3)).astype(np.uint8)  # 字=黑色(0)，越不透明越黑
    else:
        blended = (roi * (1 - a)).astype(np.uint8)
    base[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = blended
    return base


# ---------- 擦除：填纸面色+细颗粒+羽化（纯数学计算，不用 AI 模型） ----------
# 为什么弃用 LaMa：它训练自风景照，在文字密集的票据上会把周围文字笔画
# "脑补"进擦除区，成品出现浅灰残影（实测 805：输入全均匀色仍生成笔画纹理）。
def erase_by_fill(img, mask):
    """只改用户框出来的内部：搬框外的真实纸面进来盖掉旧签名
    （上下条带镜像，纹理颗粒光照全保真），边缘羽化。
    不外扩、不碰框外——用户画框沿格子线拉，表格线在框外天然完好，无需重画。"""
    dil = cv2.dilate(mask, np.ones((3, 3), np.uint8))       # 擦除范围 = 用户框（仅1px盖锯齿，绝不外扩碰线）
    dil_mask = dil > 0
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # ---- 擦除区外接矩形 ----
    ys_, xs_ = np.where(dil_mask)
    bx0, by0 = xs_.min(), ys_.min()
    bw, bh = xs_.max() + 1 - bx0, ys_.max() + 1 - by0

    # ---- 找框外的干净纸面条带（无线无字），上下各取擦除区半高 ----
    # 纸面参考亮度：防止把"票外的桌面背景"当纸搬进来
    outer0 = cv2.dilate(dil, np.ones((141, 141), np.uint8)) > 0
    ring0 = gray[outer0 & ~dil_mask]
    bright0 = ring0[ring0 > 150]
    if len(bright0) < 500:                                   # 环带亮像素太少就用全图亮像素
        bright0 = gray[gray > 150]
    paper_ref = float(np.median(bright0)) if len(bright0) else 190.0

    def clean_strip(y_from, y_to, x_from, x_to):
        if y_from < 0 or y_to > H or x_from < 0 or x_to > W or y_to <= y_from:
            return False
        s = gray[y_from:y_to, x_from:x_to]
        # 干净 = 暗像素<5% 且 亮度像纸（排除桌面/阴影等票外背景）
        return (s.size > 0 and (s < 150).mean() < 0.05
                and float(s.mean()) >= paper_ref - 12)

    half = bh // 2 + 1
    SKIP = 15   # 条带取样窗先跳过擦除区边缘 15px（用户框沿格子线拉，框边贴着表格线）

    def find_strip(above):
        # 在擦除区上方/下方找等宽条带（跳过边缘的表格线），不干净就左右平移再试
        for dx in (0, -bw // 2, bw // 2, -bw, bw, -2 * bw, 2 * bw):
            sx0, sx1 = bx0 + dx, bx0 + dx + bw
            y_from, y_to = ((by0 - half - SKIP, by0 - SKIP) if above
                            else (by0 + bh + SKIP, by0 + bh + half + SKIP))
            if clean_strip(y_from, y_to, sx0, sx1):
                return img[y_from:y_to, sx0:sx1]
        return None

    top = find_strip(True)
    bot = find_strip(False)

    canvas = img.copy()

    # ---- 构造填充块（按优先级：上下条带 > 全图找干净块 > 均匀涂色兜底） ----
    if top is not None or bot is not None:
        # 上条带镜像贴上半 + 下条带镜像贴下半，接缝渐变
        if top is not None and bot is not None:
            p_top = cv2.flip(top, 0)                          # 垂直镜像：边缘行完美衔接
            p_bot = cv2.flip(bot, 0)
            patch = np.vstack([p_top[:bh - p_bot.shape[0]] if p_top.shape[0] + p_bot.shape[0] > bh
                               else p_top, p_bot])[-bh:]
            # 接缝 15 行线性渐变，消除亮度跳变
            seam = max(0, bh - p_bot.shape[0])
            bh_blend = min(15, seam, bh - seam)
            if bh_blend > 0:
                for i in range(bh_blend):
                    wt = i / bh_blend
                    r = seam - bh_blend + i
                    patch[r] = np.clip(patch[r].astype(np.float32) * (1 - wt)
                                       + patch[min(bh - 1, r + bh_blend)].astype(np.float32) * wt,
                                       0, 255).astype(np.uint8)
        else:
            strip = top if top is not None else bot
            patch = cv2.flip(strip, 0)
            patch = cv2.resize(patch, (bw, bh), interpolation=cv2.INTER_AREA)
        canvas[by0:by0 + bh, bx0:bx0 + bw] = patch[:bh, :bw]
    else:
        # 上下条带都没有：积分图加速全图扫描，找离擦除区最近的一块同尺寸干净纸面
        dark = (gray < 150).astype(np.float32)                # cv2.integral 只认 8U/32F/64F
        ii = cv2.integral(dark)                               # 积分图：任意矩形暗像素数 O(1)
        iig = cv2.integral(gray.astype(np.float32))           # 灰度和积分图：判亮度像不像纸
        step_y, step_x = max(1, bh // 3), max(1, bw // 3)
        cy, cx = by0 + bh // 2, bx0 + bw // 2
        best = None
        for ty in range(0, H - bh + 1, step_y):
            for tx in range(0, W - bw + 1, step_x):
                d_cnt = int(ii[ty + bh, tx + bw] - ii[ty, tx + bw]
                            - ii[ty + bh, tx] + ii[ty, tx])
                if d_cnt < 0.05 * bw * bh:
                    g_sum = int(iig[ty + bh, tx + bw] - iig[ty, tx + bw]
                                - iig[ty + bh, tx] + iig[ty, tx])
                    if g_sum >= (paper_ref - 12) * bw * bh:   # 亮度像纸（排除桌面背景）
                        d = abs(ty + bh // 2 - cy) + abs(tx + bw // 2 - cx)
                        if best is None or d < best[0]:
                            best = (d, ty, tx)
        if best is not None:
            _, ty, tx = best
            canvas[by0:by0 + bh, bx0:bx0 + bw] = img[ty:ty + bh, tx:tx + bw]
        else:
            # 最终兜底（全图都找不到干净块）：环带亮像素 BGR 中位数 + 柔化颗粒
            ring_mask = outer0 & ~dil_mask
            sel = gray[ring_mask] > 150
            if sel.sum() >= 500:
                px = img[ring_mask][sel].astype(np.float32)      # 环带亮像素
            else:
                sel_full = gray > 150
                px = img[sel_full].astype(np.float32) if sel_full.any() else None
            if px is not None and len(px):
                mean = np.median(px, axis=0)
                std = float(np.clip(px.std(), 2.0, 6.0))
            else:
                mean = np.array([190.0, 190.0, 190.0])
                std = 4.0
            rng = np.random.default_rng(42)
            nz = rng.normal(0.0, 1.0, img.shape[:2]).astype(np.float32)
            nz = cv2.GaussianBlur(nz, (0, 0), 1.5)
            nz = nz / (nz.std() + 1e-6) * std
            if img.ndim == 3:
                canvas = np.clip(mean + nz[..., None], 0, 255).astype(np.uint8)
            else:
                canvas = np.clip(mean + nz, 0, 255).astype(np.uint8)

    # ---- 羽化混合：擦除区用搬进来的纸面，边缘 5px 渐变（框边贴着表格线，渐变小才不糊线） ----
    alpha = cv2.GaussianBlur(dil, (11, 11), 0).astype(np.float32) / 255.0
    if img.ndim == 3:
        a3 = alpha[..., None]
        erased = np.clip(img.astype(np.float32) * (1 - a3) + canvas.astype(np.float32) * a3,
                         0, 255).astype(np.uint8)
    else:
        erased = np.clip(img.astype(np.float32) * (1 - alpha) + canvas.astype(np.float32) * alpha,
                         0, 255).astype(np.uint8)

    # 不重画表格线：用户画框沿格子线拉，线在框外天然完好；
    # 框边缘 1-2px 的线残留由上面的小羽化自然过渡，不动框外任何像素。
    return erased


# ---------- AI 擦除：migan/zits 双模型 x 多档观察范围，块状残留打分选最优 ----------
# 模型生成填充前会"看"擦除区周边环境找参考：看的范围大容易被远处表格线带偏
# （脑补假线假字），看的范围小学不到纹理。每张图最优组合不同（实测 740 最优
# zits、766 最优 migan 小范围），所以每张图全组合试跑打分选最优。
# 打分规则：擦除区内暗像素中，细长条(长宽比>=8)是正常表格线不计分，块状才算残留。
_AI = {"models": None, "cfg": None}

AI_MODELS = ("migan", "zits")        # 参赛模型
AI_MARGINS = (16, 24, 32, 48, 64, 96, 128)   # 参赛观察范围（擦除框外扩像素数）


def _get_ai():
    """延迟加载参赛模型（GPU 优先，无显卡用 CPU）"""
    if _AI["models"] is not None:
        return _AI["models"], _AI["cfg"]
    import torch
    from iopaint.model_manager import ModelManager
    from iopaint.schema import HDStrategy, InpaintRequest
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for name in AI_MODELS:
        print(f"加载 AI 擦除模型 {name}（{device}）...")
        models[name] = ModelManager(name=name, device=device)
    _AI["models"] = models
    _AI["cfg"] = InpaintRequest(
        image="", mask="",
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=200,
        hd_strategy_resize_limit=200)
    return _AI["models"], _AI["cfg"]


def _blob_residue(g, inner):
    """块状残留像素数：暗像素里细长条是表格线不算，块状暗区才是残留"""
    dark = ((g < 140) & inner).astype(np.uint8)
    if dark.sum() == 0:
        return 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    blob = 0
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if max(w, h) / max(1, min(w, h)) < 8:
            blob += area
    return blob


def _ai_erase_once(model, cfg, img, mask, margin):
    """裁出 mask 周边 ±margin 的小图送模型（控制观察范围，也防大图爆显存），
    结果只贴回 mask 内，框外原样不动"""
    m = mask > 127
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    H, W = mask.shape
    cy0, cy1 = max(0, y0 - margin), min(H, y1 + margin)
    cx0, cx1 = max(0, x0 - margin), min(W, x1 + margin)
    sub = img[cy0:cy1, cx0:cx1]
    sub_mask = mask[cy0:cy1, cx0:cx1]
    res = np.asarray(model(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB), sub_mask, cfg))
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    out = img.copy()
    mm = sub_mask > 127
    out[cy0:cy1, cx0:cx1][mm] = res[mm]
    return out


def erase_by_ai_auto(img, mask):
    """双模型 x 多档观察范围试跑，选块状残留最少的结果。
    残留 0 分已是理论最优（不可能更低），提前收手，省掉剩余组合的推理时间。"""
    models, cfg = _get_ai()
    m = mask > 127
    inner = cv2.erode((m * 255).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    best = None
    tried = 0
    for name, model in models.items():
        for margin in AI_MARGINS:
            out = _ai_erase_once(model, cfg, img, mask, margin)
            score = _blob_residue(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), inner)
            tried += 1
            if best is None or score < best[0]:
                best = (score, out, name, margin)
            if best[0] == 0:
                break
        if best[0] == 0:
            break
    total = len(models) * len(AI_MARGINS)
    print(f"  AI 擦除最优: {best[2]} 观察范围{best[3]}px 块状残留{best[0]}px（试了 {tried}/{total} 个组合）")
    return best[1]


def main():
    # ---- 前置检查 ----
    if not os.path.isdir(IMAGES_DIR):
        print("找不到 images 文件夹")
        sys.exit(1)

    images = [f for f in sorted(os.listdir(IMAGES_DIR))
              if os.path.splitext(f)[1].lower() in EXTS]
    if not images:
        print("images 文件夹里没有图片")
        sys.exit(1)

    has_default = os.path.isfile(os.path.join(MASKS_DIR, "default_mask.png"))

    # 每张图确定用哪个 mask：单独标记优先，其次通用标记
    tasks = []  # (原图文件名, mask路径) 或 (文件名, None)=没标记
    for f in images:
        stem = os.path.splitext(f)[0]
        m = os.path.join(MASKS_DIR, stem + ".png")
        if os.path.isfile(m):
            tasks.append((f, m))
        elif has_default:
            tasks.append((f, os.path.join(MASKS_DIR, "default_mask.png")))
        else:
            tasks.append((f, None))

    no_mask = [f for f, m in tasks if m is None]
    if no_mask:
        print("以下图没有任何标记图，将被跳过（先用 1_做标记.py 生成标记）:")
        for f in no_mask:
            print("  " + f)
    tasks = [(f, m) for f, m in tasks if m is not None]
    if not tasks:
        print("没有可处理的图，退出")
        sys.exit(1)

    sign_alpha = load_sign_alpha()
    if sign_alpha is None:
        print("\n注意: my_sign 文件夹里没有签名素材，本次只擦旧签名、不贴新签名。")
        print("      放一张白纸黑字的签名照片进去即可。")
    else:
        print(f"签名笔画区尺寸: {sign_alpha.shape[1]} x {sign_alpha.shape[0]}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n开始处理 {len(tasks)} 张图...")
    done = 0
    for i, (fname, mask_path) in enumerate(tasks):
        # 1. 读转正图（0_批量转正.py 已处理好方向）和 mask，尺寸对齐
        img = imread_cn(os.path.join(IMAGES_DIR, fname))
        if img is None:
            print(f"[{i+1}/{len(tasks)}] 跳过 {fname}（读取失败）")
            continue
        mask = imread_cn(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[{i+1}/{len(tasks)}] 跳过 {fname}（标记图读取失败: {mask_path}）")
            continue
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # 2. 量图上原有手写字的粗细和墨色（贴签名前量，擦除不影响周边字）
        #    在擦除之前量：擦除只动 mask 内部，环带里的其它手写字不受影响，前后都行；
        #    但擦除前量可以保证"量到的纸面状态 = 原图状态"，最真实
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        style = measure_stroke_style(gray, mask) if sign_alpha is not None else None

        # 3. 擦除旧签名：AI 双模型自动选优；模型不可用时退回纯填色
        if ERASE_METHOD == "ai":
            try:
                erased = erase_by_ai_auto(img, mask)
            except Exception as e:
                print(f"  AI 模型不可用（{e}），本张退回纯填色")
                erased = erase_by_fill(img, mask)
        else:
            erased = erase_by_fill(img, mask)

        # 4. 贴新签名（mask 里每个独立标记框各贴一个，贴各自中心）
        if sign_alpha is not None:
            # 风格匹配：把新签名笔画调到与图上原有手写字粗细、深浅相似
            matched = style_match_alpha(sign_alpha, style)
            # 连通区域分析：一次拉了多个框时，每个分离的框是一个区域，各贴一个签名
            n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
                (mask > 127).astype(np.uint8), 8)
            placed = 0
            for k in range(1, n_cc):            # 0 号是背景，跳过
                x, y, w, h, area = stats[k]
                if area < 50:                   # 太小的碎片忽略（误点）
                    continue
                cx = x + w // 2
                cy = y + h // 2
                target_w = int(w * SIGN_WIDTH_RATIO)
                erased = paste_sign(erased, matched, cx, cy, target_w)
                placed += 1
            if placed == 0:
                print(f"[{i+1}/{len(tasks)}] 警告: {fname} 的标记区没找到有效区域，只擦除不贴签名")
            else:
                print(f"         贴了 {placed} 个新签名")

        # 5. 输出
        if OUTPUT_GRAY and erased.ndim == 3:
            erased = cv2.cvtColor(erased, cv2.COLOR_BGR2GRAY)
        out_path = os.path.join(OUTPUT_DIR, fname)
        imwrite_cn(out_path, erased)
        done += 1
        print(f"[{i+1}/{len(tasks)}] 完成 {fname} -> output/{fname}")

    print(f"\n全部结束: 成功 {done} 张，成品在 output 文件夹")


if __name__ == "__main__":
    main()
