# -*- coding: utf-8 -*-
"""
用途：用鼠标拉框，把"旧签名所在的位置"标出来，生成黑底白框的标记图（mask）。
标记图的规则：白色区域 = 要擦掉的地方，黑色 = 保留不动。

用法（在本项目目录下，用 venv 的 python 运行）：
  python 1_做标记.py                 # 连续模式：按顺序逐张弹窗，s 保存自动下一张，
                                     # 直到全部图标记完（已标记过的自动跳过）
  python 1_做标记.py 1.jpg           # 直接指定图名，给该图做单独标记（重做某张用这个）
  python 1_做标记.py default         # 用 images 里第一张图做"通用标记"，全部图共用
  python 1_做标记.py auto 工作许可人              # OCR 自动定位栏目标题，选图后单独标记
  python 1_做标记.py auto 工作许可人 default      # 同上，但存成通用标记
  python 1_做标记.py auto 工作许可人 1.jpg        # 同上，直接指定图名

auto 模式说明：OCR 读图找到"工作许可人"这类栏目标题的位置（绿框显示），
自动在标题下方生成建议签名区（红框）。按 s 直接采纳，或 r 清空后手动拉框。

保存位置：
  单独标记 -> masks/图名.png（只对同名图生效，优先级高）
  通用标记 -> masks/default_mask.png（没做单独标记的图都用它）

窗口操作：
  鼠标左键按住拉框（可以拉多个框，全算要擦的区域）
  滚轮 = 缩放（以鼠标为中心放大缩小）
  按住右键拖动 = 平移浏览（放大后移动视野）
  左转90 / 右转90 按钮 = 旋转图（保存时自动把转正图写回 images，成品也是正的）
  b = 返回上一张（本张改动不保存；上一张已保存的会还原成红框，可改后重存）
  r = 清空本图所有框，重新画
  s = 保存并退出
  ESC 或 q = 不保存退出（跳过这张）
  e = 结束整个标记流程（连续模式用，单张模式无此键）
"""
import os
import shutil
import sys

import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

# ---- 高分屏（Windows 显示缩放 125%/150%）修正 ----
# 不加这段时：系统把图像窗口整幅拉伸，但鼠标坐标按未拉伸计算，
# 表现为"拉出来的框和光标位置错开"。声明 DPI 感知后窗口按原始像素显示，坐标一致。
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Win 8.1 及以上
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()        # 更旧的系统

from signlib import (imread_cn, imwrite_cn, load_upright_image,
                     ocr_find_keyword_boxes, suggest_sign_box)

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
MASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "masks")
BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "images_backup")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_W, MAX_H = 1280, 720  # 显示窗口最大尺寸，大图自动缩小显示（保存时换算回原尺寸）
END_FLOW = object()   # "结束整个流程"的返回信号（区别 None=跳过这张、ndarray=保存的 mask）
PREV_FLOW = object()  # "返回上一张"的返回信号（本张不保存，退回前一张）


# ---------- 选图 ----------
def list_images():
    if not os.path.isdir(IMAGES_DIR):
        print("找不到 images 文件夹")
        sys.exit(1)
    files = [f for f in sorted(os.listdir(IMAGES_DIR))
             if os.path.splitext(f)[1].lower() in EXTS]
    return files


def pick_image():
    files = list_images()
    if not files:
        print("images 文件夹里没有图片，请先放几张原图进去")
        sys.exit(1)
    print("images 里的图片：")
    for i, f in enumerate(files):
        print(f"  [{i}] {f}")
    while True:
        try:
            idx = int(input("输入序号选择要标记的图: "))
            if 0 <= idx < len(files):
                return files[idx]
        except ValueError:
            pass
        print("输入无效，请重新输入序号")


# ---------- 拉框标记（tkinter 版） ----------
# 为什么不用 cv2.imshow 窗口：实测它在 Windows 上有鼠标坐标错位、
# 连续点击只响应第一次等问题；tkinter 的画面和光标天然同一坐标系，
# 事件模型也是标准 Windows 消息，稳定且能显示中文提示。
class TkMarker:
    def __init__(self, master, img, preset_boxes=None, ref_boxes=None, banner=""):
        h, w = img.shape[:2]
        self.img = img           # 保留原图，缩放时从原图取窗口渲染，放大后仍是原图清晰度
        # 打开时的初始缩放（把整图塞进 1280x720 窗口）
        self.base_scale = min(MAX_W / w, MAX_H / h, 1.0)
        self.img_shape = img.shape
        self.boxes = list(preset_boxes or [])       # 红框，会计入 mask
        self.ref_boxes = list(ref_boxes or [])      # 绿框，仅参考

        # 显示状态：zoom = 相对初始缩放的放大倍数（1.0=打开时的初始大小）
        # off_x/off_y = 空白偏移（视口左上角的映射，放大后平移浏览用）
        self.zoom = 1.0
        self.off_x = 0.0
        self.off_y = 0.0
        self.zoom_min = 0.2
        # 放大上限：显示图达到原图 1:1（eff_scale=1.0）就够看清了，再放只是插值虚糊
        self.zoom_max = max(1.0, 1.0 / self.base_scale) if self.base_scale > 0 else 4.0

        self.banner_h = 36 if banner else 0         # 顶部中文提示条高度
        self.rotated = False    # 本图在标记窗口里是否被旋转过（离开本张时据此把转正图写回 images）
        self.rot_k = 0          # 累计净旋转量：0/1/2/3 = 0/90/180/270 度（顺时针），用于同步旋转旧标记图
        self.disp_w = max(1, int(w * self.base_scale))     # 图在 zoom=1 时的显示宽（固定值，不随窗口变）
        self.disp_h = max(1, int(h * self.base_scale))     # 图在 zoom=1 时的显示高（固定值，不随窗口变）
        self.canvas_w = self.disp_w   # 窗口可视宽（初始=图宽，随窗口拉伸变化）
        self.canvas_h = self.disp_h   # 窗口可视高（初始=图高，随窗口拉伸变化）
        self.canvas = tk.Canvas(master, width=self.canvas_w,
                                height=self.canvas_h + self.banner_h,
                                cursor="tcross", highlightthickness=0,   # 黑色十字光标（crosshair 是白色）
                                background="#555555")   # 图外露出的部分中灰，看清图边界
        self.canvas.pack(fill="both", expand=True)      # 让画布跟随窗口拉伸（最大化/拖边框）
        self._resize_after = None                       # 防抖：合并连续 resize 事件，最后一刻才重绘
        self.canvas.bind("<Configure>", self._on_resize)
        if banner:
            self.canvas.create_rectangle(0, 0, self.canvas_w, self.banner_h,
                                         fill="#222222", width=0, tags="bar")
            self.canvas.create_text(self.canvas_w // 2, self.banner_h // 2, text=banner,
                                    fill="white", font=("Microsoft YaHei", 11), tags="bar")

        self.cur_id = None                          # 正在拉的预览虚线框
        self.box_ids = []                           # 已确认红框的 canvas id
        self._rebuild()                             # 首次画图片 + 参考框 + 已确认框

        # 鼠标：左键拉框，右键拖动平移，滚轮缩放
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<MouseWheel>", self._wheel)          # 滚轮缩放
        self.canvas.bind("<ButtonPress-3>", self._pan_start)   # 右键按下开始平移
        self.canvas.bind("<B3-Motion>", self._pan_move)

    # ---- 坐标换算：canvas 坐标(含提示条) <-> 原图坐标 ----
    @property
    def eff_scale(self):
        """当前显示图相对原图的缩放倍数（基准缩放 x 用户放大倍数）"""
        return self.base_scale * self.zoom

    def to_orig(self, x, y):
        return int((x - self.off_x) / self.eff_scale), \
               int((y - self.banner_h - self.off_y) / self.eff_scale)

    def to_disp(self, b):
        s = self.eff_scale
        return (int(b[0] * s + self.off_x), int(b[1] * s + self.off_y) + self.banner_h,
                int(b[2] * s + self.off_x), int(b[3] * s + self.off_y) + self.banner_h)

    # ---- 显示重绘与视图操作 ----
    def _rebuild(self):
        """按当前 zoom/off 从原图取可视窗口重新渲染，保证放大后仍是原图清晰度。
        缩放改变可视范围，必须重取窗口并 resize；平移不改变尺寸，用 canvas.move 高效移动。"""
        self.canvas.delete("disp")
        s = self.eff_scale
        H, W = self.img.shape[:2]
        # 视口（可视窗口）左上角对应的原图坐标
        view_x0 = -self.off_x / s
        view_y0 = -self.off_y / s
        x1f = view_x0 + self.canvas_w / s   # 视口右/下边界的原图坐标
        y1f = view_y0 + self.canvas_h / s
        # 原图裁剪区（与图片边界求交，取整）
        ix0 = max(0, int(np.floor(view_x0)))
        iy0 = max(0, int(np.floor(view_y0)))
        ix1 = min(W, int(np.ceil(x1f)))
        iy1 = min(H, int(np.ceil(y1f)))
        if ix1 > ix0 and iy1 > iy0:
            crop = self.img[iy0:iy1, ix0:ix1]
            cw = max(1, int(round((ix1 - ix0) * s)))   # 裁剪区缩放后的 canvas 尺寸
            ch = max(1, int(round((iy1 - iy0) * s)))
            # 放大用三次插值、缩小用区域插值，都比默认线性更清晰
            upscale = cw * ch >= (ix1 - ix0) * (iy1 - iy0)
            interp = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
            disp = cv2.resize(crop, (cw, ch), interpolation=interp)
            self.photo = ImageTk.PhotoImage(          # 保持引用防被垃圾回收
                Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
            # 裁剪区左上角在 canvas 里的位置（图超出边界时露出的部分留灰边）
            px = int(round((ix0 - view_x0) * s))
            py = int(round((iy0 - view_y0) * s)) + self.banner_h
            self.canvas.create_image(px, py, anchor="nw", image=self.photo, tags="disp")
        for b in self.ref_boxes:
            self.canvas.create_rectangle(*self.to_disp(b), outline="#00c800",
                                         width=2, dash=(6, 4), tags="disp")
        self.box_ids = []
        for b in self.boxes:
            self.box_ids.append(self.canvas.create_rectangle(
                *self.to_disp(b), outline="#ff2020", width=2, tags="disp"))

    def _on_resize(self, e):
        """窗口尺寸变化时，画布跟随变大/变小，画面按当前缩放重新铺满窗口。
        拖拽边框会触发一连串 Configure，防抖到最后一刻才重绘，避免卡顿。"""
        new_w = max(1, e.width)
        new_h = max(1, e.height - self.banner_h)
        if new_w == self.canvas_w and new_h == self.canvas_h:
            return
        self.canvas_w = new_w
        self.canvas_h = new_h
        if self._resize_after is not None:
            self.canvas.after_cancel(self._resize_after)
        # 先按新窗口尺寸修正偏移（图小于新窗口时居中），再重绘
        self._resize_after = self.canvas.after(
            50, lambda: (self._clamp_offset(), self._rebuild()))

    def _clamp_offset(self):
        """限制平移范围：图片比窗口大时最多拖到边缘，比窗口小时居中显示
        （业界标准：视口大于图片时图片居中，不是贴左上角）。
        注意图的显示尺寸用固定的 disp_w/disp_h 算，不能用当前窗口宽，
        否则窗口拉大后图的尺寸估算跟着变大，居中逻辑永远不触发。"""
        img_w = max(1, int(self.disp_w * self.zoom))
        img_h = max(1, int(self.disp_h * self.zoom))
        if img_w <= self.canvas_w:
            self.off_x = (self.canvas_w - img_w) / 2.0   # 图窄于窗口：水平居中
        else:
            self.off_x = min(0.0, max(float(self.canvas_w - img_w), self.off_x))
        if img_h <= self.canvas_h:
            self.off_y = (self.canvas_h - img_h) / 2.0   # 图矮于窗口：垂直居中
        else:
            self.off_y = min(0.0, max(float(self.canvas_h - img_h), self.off_y))

    def _wheel(self, e):
        """滚轮缩放：以鼠标指向的原图点为锚点，放大缩小后该点保持在原地"""
        ox, oy = self.to_orig(e.x, e.y)
        factor = 1.2 if e.delta > 0 else 1.0 / 1.2
        new_zoom = min(max(self.zoom * factor, self.zoom_min), self.zoom_max)
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        self.zoom = new_zoom
        s = self.eff_scale
        self.off_x = e.x - ox * s
        self.off_y = (e.y - self.banner_h) - oy * s
        self._clamp_offset()
        self._rebuild()

    def _pan_start(self, e):
        self.pan_last = (e.x, e.y)

    def _pan_move(self, e):
        dx = e.x - self.pan_last[0]
        dy = e.y - self.pan_last[1]
        self.pan_last = (e.x, e.y)
        old_x, old_y = self.off_x, self.off_y
        self.off_x += dx
        self.off_y += dy
        self._clamp_offset()
        self.canvas.move("disp", self.off_x - old_x, self.off_y - old_y)

    def reset_view(self):
        """复位：缩放归 1、平移归零，回到打开时的全图视角
        （窗口被拉大/最大化时，全图居中显示）"""
        self.zoom = 1.0
        self.off_x = 0.0
        self.off_y = 0.0
        self._clamp_offset()   # 图小于当前窗口时按居中修正偏移
        self._rebuild()

    def rotate(self, clockwise=True):
        """整图旋转 90 度（按钮触发，转的是标记用的图本身，保存时写回 images 转正）。
        已画的框坐标同步变换；图尺寸变了，缩放基准全部刷新，视图复位到全图居中。
        坐标变换（顺时针）：原(x,y)->(H-1-y,x)，框(x1,y1,x2,y2)->(H-y2,x1,H-y1,x2)；
        逆时针：原(x,y)->(y,W-1-x)，框->(y1,W-x2,y2,W-x1)。"""
        code = cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE
        self.img = cv2.rotate(self.img, code)
        oh, ow = self.img_shape[:2]          # 旋转前的图高、图宽
        if clockwise:
            def tf(b):
                return (oh - b[3], b[0], oh - b[1], b[2])
        else:
            def tf(b):
                return (b[1], ow - b[2], b[3], ow - b[0])
        self.boxes = [tf(b) for b in self.boxes]
        self.ref_boxes = [tf(b) for b in self.ref_boxes]
        # 尺寸相关的基准全部按新图重建
        nh, nw = self.img.shape[:2]
        self.img_shape = self.img.shape
        self.base_scale = min(MAX_W / nw, MAX_H / nh, 1.0)
        self.disp_w = max(1, int(nw * self.base_scale))
        self.disp_h = max(1, int(nh * self.base_scale))
        self.zoom = 1.0
        self.off_x = 0.0
        self.off_y = 0.0
        self.rotated = True
        self.rot_k = (self.rot_k + (1 if clockwise else 3)) % 4   # 逆时针 = 顺时针转 3 次
        self._clamp_offset()
        self._rebuild()

    def _commit_box(self, b):
        self.box_ids.append(self.canvas.create_rectangle(
            *self.to_disp(b), outline="#ff2020", width=2, tags="disp"))

    # ---- 鼠标拉框 ----
    def _press(self, e):
        self.sx, self.sy = e.x, e.y
        self.cur_id = self.canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="#ff2020", width=2, dash=(4, 4))

    def _drag(self, e):
        if self.cur_id is not None:
            self.canvas.coords(self.cur_id, self.sx, self.sy, e.x, e.y)

    def _release(self, e):
        if self.cur_id is not None:
            self.canvas.delete(self.cur_id)
            self.cur_id = None
        x1, y1 = self.to_orig(self.sx, self.sy)
        x2, y2 = self.to_orig(e.x, e.y)
        if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:   # 太小的框忽略（误点）
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.boxes.append(box)
            self._commit_box(box)

    def make_mask(self):
        h, w = self.img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in self.boxes:
            mask[y1:y2, x1:x2] = 255
        return mask


# ---------- 已保存标记的还原与旋转写回 ----------
def boxes_from_mask(mask):
    """从已保存的标记图提取白框列表（返回上一张时还原成红框用）。
    注意：相邻/重叠的框在 mask 里已连成一片，只能还原成合并后的大框，
    数量对不上属正常，不合适就按 r 清空重画。"""
    m = (mask > 127).astype(np.uint8)
    n, labels = cv2.connectedComponents(m, connectivity=8)
    boxes = []
    for lb in range(1, n):
        ys, xs = np.where(labels == lb)
        boxes.append((int(xs.min()), int(ys.min()),
                      int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def save_rotated_image(fname, img):
    """标记时旋转过的图写回 images（旋转即转正）：后续批量换签名读到的原图
    和输出的成品都是转正方向。写回前先备份原始图到 tmp/images_backup，
    只备份一次（已有备份不覆盖，保证 tmp 里永远是最原始的版本）。"""
    src = os.path.join(IMAGES_DIR, fname)
    dst = os.path.join(BACKUP_DIR, fname)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if os.path.isfile(src) and not os.path.isfile(dst):
        shutil.copy2(src, dst)
    imwrite_cn(src, img)


def rotate_image_by_k(img, k):
    """按累计净旋转量旋转图：k=0/1/2/3 对应 0/90/180/270 度（顺时针）"""
    codes = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
             cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img if k % 4 == 0 else cv2.rotate(img, codes[k % 4])


def persist_rotation(fname, minfo, mask_just_saved=False):
    """把标记窗口里的旋转落到磁盘（在"离开本张"的时刻统一调用）：
    1) 转正图写回 images（先备份原件）——不管这张最终是保存/跳过/返回上一张，旋转都生效；
    2) 若磁盘上已有这张的标记图且没有被刚存的新 mask 覆盖，同步旋转它，
       保证图和标记的坐标系一致（否则下次打开框会错位）。
    mask_just_saved=True 表示调用方刚按旋转后的方向存过新 mask，旧 mask 不用再转。"""
    if not minfo.get("rotated"):
        return
    save_rotated_image(fname, minfo["img"])
    print(f"    -> 已转正写回 images/{fname}（原件备份在 tmp/images_backup）")
    if mask_just_saved:
        return
    mp = os.path.join(MASKS_DIR, os.path.splitext(fname)[0] + ".png")
    if os.path.isfile(mp):
        old = imread_cn(mp, cv2.IMREAD_GRAYSCALE)
        if old is not None:
            imwrite_cn(mp, rotate_image_by_k(old, minfo.get("rot_k", 0)))
            print("    -> 已有标记图同步旋转（保持图和框坐标一致）")


# 上一张标记窗口的状态（是否最大化/普通模式的位置尺寸）。
# 单张模式每次运行都是新进程新窗口，靠它把上次的最大化/窗口大小带给下一次运行。
_last_win = {"zoomed": False, "geo": None}


def _marker_in_root(root, img, preset_boxes=None, ref_boxes=None,
                    banner="", allow_end=False, allow_prev=False):
    """在已有 root 窗口里布置一张图的标记界面，等用户操作完返回结果。
    连续模式复用同一个窗口：换图时销毁旧控件重新布置，窗口的最大化
    状态和大小位置全程保持不动，不用每张图重开窗口。
    返回 (结果, 图信息)：
      结果 = ndarray(保存的mask) | None(跳过这张) | END_FLOW(结束流程) | PREV_FLOW(返回上一张)
      图信息 = {"img": 当前图数据(可能被旋转过), "rotated": 是否旋转过}"""
    for w in root.winfo_children():
        w.destroy()                      # 清掉上一张的画布和按钮条（窗口本身保留）
    m = TkMarker(root, img, preset_boxes, ref_boxes, banner)
    print("操作: 左键拉框(可多个) | 滚轮缩放 | 右键拖动平移 | u 撤销 | r 清空 | s 下一张(有框保存) | ESC 跳过这张")
    if allow_end:
        print("      e = 结束整个标记流程（后面的图都不弹了）")
    print("      b = 返回上一张（本张改动不保存）")
    print("框的颜色: 红框=擦除区(计入标记) 绿框=栏目标题(仅参考)")
    print("提示: 键盘没反应（多半是中文输入法把按键拦了）就直接点窗口下方的按钮")
    state = {"result": None}
    done_var = tk.IntVar(root)           # 完成信号：finish 里置 1，唤醒下面的 wait_variable
    done_var.set(0)

    def finish(result=None):
        if result is not None:
            state["result"] = result
        done_var.set(1)

    def on_save():
        if not m.boxes:
            print("一个框都没拉，不算保存。先拉框再保存（跳过这张按 ESC 或点跳过按钮）")
            return
        finish(m.make_mask())

    def on_end():
        finish(END_FLOW)

    def on_prev():
        # 旋转属于"图的转正"，返回上一张时也会由 persist_rotation 落盘，不用在此处理
        finish(PREV_FLOW)

    def on_next():
        """下一张（s）：拉了框就保存，没拉过框（也没改动）就直接跳过不保存"""
        if m.boxes:
            finish(m.make_mask())
        else:
            print("本张没有框，不保存直接下一张")
            finish(None)

    def on_undo():
        if m.boxes and m.box_ids:
            m.boxes.pop()
            m.canvas.delete(m.box_ids.pop())
            print("已撤销上一框，剩余 %d 个框" % len(m.boxes))
        else:
            print("没有可撤销的框")

    def on_clear():
        m.boxes = []
        for i in m.box_ids:
            m.canvas.delete(i)
        m.box_ids = []
        print("已清空，重新拉框")

    # ---- 底部按钮条：键盘被输入法/大写锁定/焦点问题拦截时，鼠标点按钮同样有效 ----
    bar = tk.Frame(root)
    bar.pack(fill="x", pady=4)
    # 第一张也显示"上一张"按钮（点了会提示已是第一张），避免用户以为按钮丢了
    tk.Button(bar, text="上一张 (b)", command=on_prev).pack(side="left", padx=4)
    tk.Button(bar, text="左转90", command=lambda: m.rotate(False)).pack(side="left", padx=4)
    tk.Button(bar, text="右转90", command=lambda: m.rotate(True)).pack(side="left", padx=4)
    tk.Button(bar, text="复位视图", command=m.reset_view).pack(side="left", padx=4)
    tk.Button(bar, text="撤销 (u)", command=on_undo).pack(side="left", padx=4)
    tk.Button(bar, text="清空 (r)", command=on_clear).pack(side="left", padx=4)
    if allow_end:
        tk.Button(bar, text="结束 (e)", command=on_end).pack(side="left", padx=4)
    tk.Button(bar, text="下一张(有框保存) (s)", command=on_next).pack(side="left", padx=4)

    # ---- 键盘统一入口：把键名转成小写再比对，大小写都能触发 ----
    def on_key(e):
        k = (e.keysym or "").lower()
        if k == "s":
            on_next()
        elif k == "u":
            on_undo()
        elif k == "r":
            on_clear()
        elif allow_end and k == "e":
            on_end()
        elif k == "b":
            on_prev()
        elif k in ("escape", "q"):
            finish()

    root.bind("<Key>", on_key)
    # 点右上角 X：连续复用窗口时 X 不能只跳一张（窗口关不掉），统一按"结束流程"处理；
    # 单张模式没有"结束"概念，等价于跳过/不保存
    root.protocol("WM_DELETE_WINDOW",
                  lambda: finish(END_FLOW if allow_end else None))
    root.focus_force()
    root.wait_variable(done_var)         # 局部事件循环：等这张图操作完，窗口不销毁
    root.unbind("<Key>")
    return state["result"], {"img": m.img, "rotated": m.rotated, "rot_k": m.rot_k}


def run_marker(img, window_title, preset_boxes=None, ref_boxes=None,
               banner="", keep_open=False, allow_end=False):
    """单张模式：新开一个窗口标记一张图（应用上次记住的窗口状态）。
    连续模式不走这里，直接用 _marker_in_root 复用窗口。
    返回 (结果, 图信息)，与 _marker_in_root 相同。"""
    root = tk.Tk()
    root.title(window_title)
    # 应用上一张窗口的状态：最大化直接保持最大化，普通模式照搬位置和尺寸
    if _last_win["zoomed"]:
        root.state("zoomed")          # Windows tkinter 的最大化
    elif _last_win["geo"]:
        root.geometry(_last_win["geo"])
    result, info = _marker_in_root(root, img, preset_boxes, ref_boxes, banner, allow_end)
    # 记住这张窗口的状态，下一次运行照此打开（保住最大化/用户调好的大小）
    try:
        _last_win["zoomed"] = (root.state() == "zoomed")
        _last_win["geo"] = None if _last_win["zoomed"] else root.geometry()
    except tk.TclError:
        pass   # 窗口已销毁等异常时保持上次记录，不影响流程
    root.destroy()
    return result, info


# ---------- 连续标记模式（无参数双击时用这个） ----------
def run_continuous():
    """按顺序逐张弹窗拉框：s 保存自动下一张，ESC 跳过这张。
    已有标记图的自动跳过 -> 中途 Ctrl+C 停止后，下次双击从断点继续。"""
    files = list_images()
    if not files:
        print("images 文件夹里没有图片，请先放几张原图进去")
        return
    os.makedirs(MASKS_DIR, exist_ok=True)

    def mask_of(f):
        return os.path.join(MASKS_DIR, os.path.splitext(f)[0] + ".png")

    # 队列包含全部图：已标记过的也显示（旧标记自动还原成红框，可查可改，
    # 没问题就按 s 走下一张，重存一遍相同内容无害）
    pending = list(files)
    n_done_before = sum(1 for f in files if os.path.isfile(mask_of(f)))
    if n_done_before:
        print(f"共 {len(files)} 张图，其中 {n_done_before} 张之前已标记过（旧框自动还原成红框显示）")

    print(f"本次共过 {len(pending)} 张")
    print("操作: 左键拉框(可多个) | 滚轮缩放 | 右键拖动平移 | u 撤销 | r 清空")
    print("      s = 下一张(有框就保存) | b = 返回上一张 | ESC = 跳过这张(已有标记保留不动)")
    print("      e 或窗口右上角X = 结束整个流程")
    print("      图躺倒了就点 左转90/右转90 按钮，保存时自动转正写回（原件先备份）\n")

    # fname -> "saved"/"skipped"，走完统计用
    results = {}
    # 全程只开一个窗口：换图时窗口里的画布换掉，窗口的最大化/大小位置一直保持
    root = tk.Tk()
    root.title("拉框标记")
    i, n = 0, len(pending)
    try:
        while 0 <= i < n:
            f = pending[i]
            img = imread_cn(os.path.join(IMAGES_DIR, f))
            if img is None:
                print(f"[{i+1}/{n}] 读取失败: {f}")
                results[f] = "skipped"
                i += 1
                continue
            print(f"\n[{i+1}/{n}] {f}  尺寸: {img.shape[1]}x{img.shape[0]}")
            # 已保存过标记的（本次会话存过或之前跑过），还原成红框供修改
            preset = None
            mp = mask_of(f)
            if os.path.isfile(mp):
                old = imread_cn(mp, cv2.IMREAD_GRAYSCALE)
                if old is not None:
                    preset = boxes_from_mask(old)
                    print(f"    已有标记还原成 {len(preset)} 个红框（相邻/重叠的框会合并成大框，不合适就 r 清空重画）")
            banner = f"第 {i+1}/{n} 张：拉框|滚轮缩放|右键拖动|s下一张(有框保存) b上一张 e结束"
            result, minfo = _marker_in_root(root, img, preset_boxes=preset,
                                            banner=banner, allow_end=True)
            # 旋转落盘统一在这里：保存/跳过/返回上一张/结束，任何离开本张的时刻都生效
            mask_saved = isinstance(result, np.ndarray)
            if minfo.get("rotated"):
                persist_rotation(f, minfo, mask_just_saved=mask_saved)
            if result is END_FLOW:
                print("    -> 手动结束，停止后续标记（已保存的都在）")
                break
            if result is PREV_FLOW:
                if i == 0:
                    print("    -> 已经是第一张了")
                    continue
                print("    -> 返回上一张（本张未保存）")
                i -= 1
                continue
            if result is None:
                print("    -> 跳过（未保存）")
                results[f] = "skipped"
            else:
                imwrite_cn(mp, result)
                results[f] = "saved"
                print(f"    -> 已保存 masks/{os.path.splitext(f)[0]}.png")
            i += 1
    except KeyboardInterrupt:
        print("\n\n已手动停止")
    finally:
        root.destroy()

    done = sum(1 for v in results.values() if v == "saved")
    passed = sum(1 for v in results.values() if v == "skipped")
    print(f"\n本次结果: 保存 {done} 张，跳过 {passed} 张")
    left = len(pending) - done - passed
    if left > 0:
        print(f"还有 {left} 张没处理，下次双击 双击_做标记.bat 从断点继续")
    elif passed:
        print("已到最后一遍。跳过的图重做方式: python 1_做标记.py 图名.jpg")
        print("全部就绪后双击 双击_批量换签名.bat 开始修图")
    else:
        print("全部标记完成！双击 双击_批量换签名.bat 开始修图")


# ---------- OCR 自动定位栏目标题 ----------
def main():
    args = sys.argv[1:]

    # ---------- auto 模式：OCR 按栏目标题自动定位 ----------
    if args and args[0] == "auto":
        if len(args) < 2:
            print('用法: python 1_做标记.py auto 关键词 [default|图名]')
            print('例如: python 1_做标记.py auto 工作许可人 default')
            return
        keyword = args[1]
        rest = args[2:]
        files = list_images()
        if not files:
            print("images 文件夹里没有图片，请先放几张原图进去")
            return
        if rest and rest[0] == "default":
            name = files[0]
            out_name = "default_mask.png"
            mode = "通用标记（所有图共用）"
        elif rest:
            name = rest[0]
            if not os.path.isfile(os.path.join(IMAGES_DIR, name)):
                print(f"images 里找不到 {name}")
                return
            out_name = os.path.splitext(name)[0] + ".png"
            mode = "单独标记（只对这张图生效）"
        else:
            name = pick_image()
            out_name = os.path.splitext(name)[0] + ".png"
            mode = "单独标记（只对这张图生效）"

        img = imread_cn(os.path.join(IMAGES_DIR, name))   # upright 里已是转正图，直接读
        if img is None:
            print(f"图片读取失败: {name}")
            return
        print(f"\n当前模式: {mode} | 图片: {name} | 定位关键词: {keyword}")
        from rapidocr import RapidOCR
        engine = RapidOCR()
        hits = ocr_find_keyword_boxes(engine, img, keyword)
        if not hits:
            print(f"OCR 没找到含 '{keyword}' 的栏目，请手动拉框（模式退回普通标记）")
            mask, minfo = run_marker(img, "mask marker")
        else:
            if len(hits) > 1:
                print(f"找到 {len(hits)} 处匹配的标题（绿框全部显示），建议签名区按第一处生成")
            else:
                print("找到栏目标题（绿框），已在其下方生成建议签名区（红框）")
            preset = [suggest_sign_box(hits[0], img.shape)]
            mask, minfo = run_marker(img, "mask marker", preset_boxes=preset, ref_boxes=hits)
        if mask is None:
            # 没保存标记，但旋转过的图仍转正写回（旋转属于图的修正，独立于标记保存）
            if minfo.get("rotated"):
                persist_rotation(name, minfo)
            print("未保存，退出")
            return
        os.makedirs(MASKS_DIR, exist_ok=True)
        out_path = os.path.join(MASKS_DIR, out_name)
        imwrite_cn(out_path, mask)
        # 存的是本图自己的 mask 时旧 mask 已被覆盖，不用同步旋转；存 default_mask 时本图
        # 自己的旧 mask（若有）需要同步转
        if minfo.get("rotated"):
            persist_rotation(name, minfo,
                             mask_just_saved=(out_name == os.path.splitext(name)[0] + ".png"))
        print(f"已保存标记图: {out_path}（白=擦除区，黑=保留）")
        return

    if args and args[0] == "default":
        files = list_images()
        if not files:
            print("images 文件夹里没有图片，请先放几张原图进去")
            return
        name = files[0]
        out_name = "default_mask.png"
        mode = "通用标记（所有图共用）"
    elif args:
        name = args[0]
        if not os.path.isfile(os.path.join(IMAGES_DIR, name)):
            print(f"images 里找不到 {name}")
            return
        out_name = os.path.splitext(name)[0] + ".png"
        mode = "单独标记（只对这张图生效）"
    else:
        # 无参数 = 连续模式：逐张弹窗，s 保存自动下一张
        run_continuous()
        return

    img = imread_cn(os.path.join(IMAGES_DIR, name))   # upright 里已是转正图，直接读
    if img is None:
        print(f"图片读取失败: {name}")
        return

    print(f"\n当前模式: {mode}")
    print(f"图片: {name}  尺寸: {img.shape[1]}x{img.shape[0]}")
    mask, minfo = run_marker(img, "mask marker")
    if mask is None:
        # 没保存标记，但旋转过的图仍转正写回（旋转属于图的修正，独立于标记保存）
        if minfo.get("rotated"):
            persist_rotation(name, minfo)
        print("未保存，退出")
        return

    os.makedirs(MASKS_DIR, exist_ok=True)
    out_path = os.path.join(MASKS_DIR, out_name)
    imwrite_cn(out_path, mask)
    # 存的是本图自己的 mask 时旧 mask 已被覆盖，不用同步旋转；存 default_mask 时本图
    # 自己的旧 mask（若有）需要同步转
    if minfo.get("rotated"):
        persist_rotation(name, minfo,
                         mask_just_saved=(out_name == os.path.splitext(name)[0] + ".png"))
    print(f"已保存标记图: {out_path}（白=擦除区，黑=保留）")


if __name__ == "__main__":
    main()
