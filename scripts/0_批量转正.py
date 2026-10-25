# -*- coding: utf-8 -*-
"""
用途：把 images 里所有图的方向直接转正（原地覆盖）。
转正规则：照片自带方向信息（EXIF）优先；没有的（微信转存图常见）就用 OCR
把图转 4 个方向各识别一遍文字，哪个方向"认出的字最多最准"（识别得分总和
最高），哪个方向就是正的。

安全措施：转正前先把原图备份到 tmp/images_backup/（只备份一次，重跑不覆盖备份）。

之后 1_做标记.py（画框）和 2_批量换签名.py（修图）直接读 images 里的转正图。

用法：python 0_批量转正.py [图名]   # 给图名只转那一张
"""
import os
import shutil
import sys

from signlib import imread_cn, imwrite_cn, load_upright_image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 脚本在 scripts/ 里，项目根取上一级
IMAGES_DIR = os.path.join(BASE, "images")
BACKUP_DIR = os.path.join(BASE, "tmp", "images_backup")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not os.path.isdir(IMAGES_DIR):
        print("找不到 images 文件夹")
        return
    files = [f for f in sorted(os.listdir(IMAGES_DIR))
             if os.path.splitext(f)[1].lower() in EXTS]
    # 命令行指定了图名就只处理那一张
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only:
        files = [f for f in files if f == only]
        if not files:
            print(f"images 里找不到 {only}")
            return

    # 先备份原图（没备份过的才备份）
    os.makedirs(BACKUP_DIR, exist_ok=True)
    n_backup = 0
    for f in files:
        src = os.path.join(IMAGES_DIR, f)
        dst = os.path.join(BACKUP_DIR, f)
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)
            n_backup += 1

    for i, f in enumerate(files, 1):
        img = load_upright_image(os.path.join(IMAGES_DIR, f))
        if img is None:
            print(f"[{i}/{len(files)}] [读取失败] {f}")
            continue
        h, w = img.shape[:2]
        imwrite_cn(os.path.join(IMAGES_DIR, f), img)
        print(f"[{i}/{len(files)}] 已转正并覆盖 {f} ({w}x{h})")

    print(f"\n全部结束: 转正覆盖 {len(files)} 张，本次新备份 {n_backup} 张（备份在 tmp/images_backup/）")
    print("接下来用 1_做标记.py 逐张画框")


if __name__ == "__main__":
    main()
