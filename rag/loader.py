import csv
import os
from pathlib import Path
from typing import Optional
from pypdf import PdfReader

def load_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    
    if ext in [".md", ".txt", ".json"]:
        return _load_text(file_path)
    elif ext == ".csv":
        return _load_csv(file_path)
    elif ext == ".pdf":
        return _load_pdf(file_path)
    else:
        print(f"Warning: Unsupported file type {ext} for {file_path}")
        return ""


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading text file {path}: {e}")
        return ""


def _load_csv(path: Path) -> str:
    """
    将 CSV 转换为语义化的文本格式。
    每一行转换为 "Column1: Value1, Column2: Value2..." 的形式。
    """
    text_content = []
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return ""
            
            for row in reader:
                # 过滤掉空值，构建描述性字符串
                row_parts = [f"{k}: {v}" for k, v in row.items() if v]
                if row_parts:
                    text_content.append(", ".join(row_parts))
        
        return "\n".join(text_content)
    except Exception as e:
        print(f"Error reading CSV {path}: {e}")
        return ""


def _load_pdf(path: Path) -> str:
    if PdfReader is None:
        print(f"Error: pypdf not installed. Cannot read {path}. Please run `pip install pypdf`.")
        return ""

    text_content = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        return "\n\n".join(text_content)
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return ""
