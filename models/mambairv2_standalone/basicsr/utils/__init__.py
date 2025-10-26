# 导出必要的函数和类
from .registry import ARCH_REGISTRY
import os

# 添加一个简单的 get_root_logger 函数来避免导入错误
def get_root_logger():
    import logging
    return logging.getLogger()

# 添加 scandir 函数的简单实现
def scandir(dir_path):
    """Scan a directory to find the interested files or subdirectories.
    """
    if not os.path.isdir(dir_path):
        raise Exception(f"'{dir_path}' is not a valid directory.")
    
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_list.append(os.path.join(root, file).replace('\\', '/'))
    return file_list

__all__ = ['ARCH_REGISTRY', 'get_root_logger', 'scandir']