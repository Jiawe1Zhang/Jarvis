from typing import List


### mimic RecursiveCharacterTextSplitter from LangChain 😆😄 

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 分隔符优先级：双换行(段落) > 单换行 > 句号 > 空格 > 字符
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        if not text:
            return final_chunks
            
        # 1. 尝试用最高优先级的分割符切分
        separator = self.separators[-1]
        for sep in self.separators:
            if sep in text:
                separator = sep
                break
        
        # 如果找不到分隔符（比如是一段很长的无标点字符串），直接按字符切
        if separator == "":
            splits = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size-self.chunk_overlap)]
            return splits

        # 2. 初步切分
        splits = text.split(separator)
        
        # 3. 合并碎片 (Merge)
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # 恢复分隔符（除了最后一个）
            segment = split + separator if separator != "" else split 
            segment_len = len(segment)
            
            if current_length + segment_len > self.chunk_size:
                # 当前块满了，保存
                if current_chunk:
                    doc = "".join(current_chunk).strip()
                    if doc:
                        final_chunks.append(doc)
                    
                    # 处理重叠：保留尾部的一些片段作为下一个块的开头
                    # 简化处理：只保留最后一个 segment
                    current_chunk = [segment] 
                    current_length = segment_len
                else:
                    # 单个片段就超过了 chunk_size，强制切分
                    final_chunks.append(segment[:self.chunk_size])
                    # 剩余部分继续处理... (简化起见略过递归)
            else:
                current_chunk.append(segment)
                current_length += segment_len
        
        # 处理最后一个块
        if current_chunk:
            final_chunks.append("".join(current_chunk).strip())
            
        return final_chunks
