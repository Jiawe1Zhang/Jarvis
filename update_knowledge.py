import os
import glob
from pathlib import Path

def update_knowledge_files():
    # 设定 knowledge 目录的路径
    knowledge_dir = Path("knowledge")
    
    # 检查目录是否存在
    if not knowledge_dir.exists():
        print(f"Directory {knowledge_dir} does not exist.")
        return

    # 获取所有 .md 文件
    files = list(knowledge_dir.glob("*.md"))
    
    if not files:
        print("No markdown files found in knowledge directory.")
        return

    print(f"Found {len(files)} files. Starting update...")

    for file_path in files:
        # --- 在这里定义你的修改逻辑 ---
        
        # 示例 1: 读取原始内容
        # original_content = file_path.read_text(encoding='utf-8')
        
        # 示例 2: 完全替换为新内容
        new_content = f"""# Knowledge Base: {file_path.stem}

This is updated content for {file_path.name}.
Current status: Verified.
Date: 2025-12-04
"""
        
        # 示例 3: 在原始内容后追加
        # new_content = original_content + "\n\n---\nUpdated by script."

        # 写入新内容
        file_path.write_text(new_content, encoding='utf-8')
        print(f"Updated: {file_path}")

    print("All files updated successfully.")

if __name__ == "__main__":
    update_knowledge_files()
