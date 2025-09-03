#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动检测可用摄像头索引并批量更新项目中的固定索引。

功能：
- 扫描 0..max_index 的摄像头，找到第一个可用索引。
- 将项目中所有 Python 文件里的以下写法替换为检测到的索引：
  - cv2.VideoCapture(<数字> ...)
  - videoSourceIndex = <数字>
- 为每个被修改的文件生成 .bak 备份。

用法：
  python scripts/detect_and_update_camera_index.py [--max 10] [--dry-run]

注意：
- 脚本会跳过自身文件与 __pycache__ 目录。
- 如果未检测到摄像头，默认不会修改文件（除非通过 --force 传入一个指定索引）。
"""

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None


def find_first_available_camera(max_index: int = 10) -> int:
    if cv2 is None:
        return -1
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            # 进一步尝试读取一帧，避免某些驱动假阳性
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
        else:
            cap.release()
    return -1


def replace_in_text(text: str, new_index: int) -> tuple[str, int]:
    replaced_count = 0

    # 替换 cv2.VideoCapture(<int> [ , or ) ])
    pattern_cap = re.compile(r"(cv2\.VideoCapture\(\s*)(\d+)(\s*)([,\)])")

    def repl_cap(m: re.Match) -> str:
        nonlocal replaced_count
        replaced_count += 1
        return f"{m.group(1)}{new_index}{m.group(3)}{m.group(4)}"

    text = pattern_cap.sub(repl_cap, text)

    # 替换 videoSourceIndex = <int>
    pattern_vidsrc = re.compile(r"^(\s*videoSourceIndex\s*=\s*)(\d+)", re.MULTILINE)

    def repl_vidsrc(m: re.Match) -> str:
        nonlocal replaced_count
        replaced_count += 1
        return f"{m.group(1)}{new_index}"

    text = pattern_vidsrc.sub(repl_vidsrc, text)

    return text, replaced_count


def iter_python_files(root: Path):
    for p in root.rglob("*.py"):
        # 跳过缓存与本脚本
        if "__pycache__" in p.parts:
            continue
        if p.name == Path(__file__).name:
            continue
        yield p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=10, help="扫描的最大摄像头索引")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要修改的内容，不写回文件")
    parser.add_argument("--force", type=int, default=None, help="跳过检测，强制使用指定索引")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    if args.force is not None:
        cam_index = int(args.force)
        print(f"[info] 跳过检测，使用 --force 指定的索引: {cam_index}")
    else:
        cam_index = find_first_available_camera(args.max)
        if cam_index < 0:
            print("[error] 未检测到可用摄像头。可用 --force <index> 强制设置。")
            return 2
        print(f"[info] 检测到可用摄像头索引: {cam_index}")

    total_files = 0
    total_replacements = 0
    changed_files: list[Path] = []

    for pyfile in iter_python_files(root):
        text = pyfile.read_text(encoding="utf-8")
        new_text, n = replace_in_text(text, cam_index)
        if n > 0:
            total_files += 1
            total_replacements += n
            changed_files.append(pyfile.relative_to(root))
            if not args.dry_run:
                # 备份
                bak_path = pyfile.with_suffix(pyfile.suffix + ".bak")
                try:
                    if not bak_path.exists():
                        bak_path.write_text(text, encoding="utf-8")
                except Exception as e:
                    print(f"[warn] 备份失败 {bak_path}: {e}")
                # 写回
                pyfile.write_text(new_text, encoding="utf-8")

    if total_files == 0:
        print("[info] 没有发现需要替换的固定摄像头索引。")
    else:
        print(f"[done] 已处理 {total_files} 个文件，共替换 {total_replacements} 处。")
        for f in changed_files:
            print(f"  - {f}")

    if args.dry_run:
        print("[note] dry-run 模式未写回文件。移除 --dry-run 即可应用修改。")

    return 0


if __name__ == "__main__":
    sys.exit(main())

