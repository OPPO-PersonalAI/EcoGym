#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from colorama import init, Fore, Style
from typing import Optional

init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """支持彩色输出的日志格式化器"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    MESSAGE_COLORS = {
        '=' * 60: Fore.CYAN + Style.BRIGHT,  
        '=' * 80: Fore.CYAN + Style.BRIGHT,  
        'Step ': Fore.CYAN + Style.BRIGHT,
        '[State]': Fore.WHITE + Style.BRIGHT,
        '[Agent Input]': Fore.GREEN + Style.BRIGHT,
        '[Agent Output]': Fore.BLUE + Style.BRIGHT,
        '[Tool Call]': Fore.YELLOW + Style.BRIGHT,
        '[Tool Output]': Fore.MAGENTA + Style.BRIGHT,
        '[Tools Summary]': Fore.CYAN,
        '[Model Reasoning]': Fore.CYAN,
        '[Model Pre-Tool Output]': Fore.BLUE,
        '[Model Tool Call Request]': Fore.YELLOW,
        '[Token Usage]': Fore.GREEN,
        '[Step Cost]': Fore.GREEN,
        '[Daily Action Limit]': Fore.YELLOW,
        '[LLM API Duration]': Fore.CYAN,
        '[Auto-call task_done]': Fore.YELLOW,
        'Session ID': Fore.CYAN + Style.BRIGHT,
        'Resuming': Fore.YELLOW + Style.BRIGHT,
        'Starting simulation': Fore.GREEN + Style.BRIGHT,
        '✅': Fore.GREEN + Style.BRIGHT,
        '❌': Fore.RED + Style.BRIGHT,
        'Agent System Prompt': Fore.MAGENTA + Style.BRIGHT,
        'Tool Schemas': Fore.MAGENTA + Style.BRIGHT,
        'Token Usage Statistics': Fore.GREEN + Style.BRIGHT,
        '💰': Fore.GREEN + Style.BRIGHT,
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        if not self.use_colors:
            return super().format(record)
        
        formatted = super().format(record)
        
        level_color = self.COLORS.get(record.levelname, '')
        
        message_color = ''
        msg = record.getMessage()
        for prefix, color in self.MESSAGE_COLORS.items():
            if prefix in msg:
                message_color = color
                break
        
        color = message_color if message_color else level_color
        
        if color:
            return f"{color}{formatted}{Style.RESET_ALL}"
        return formatted


def setup_colored_logging(
    log_file_path: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    use_colors: bool = True
):
    """
    设置统一的彩色日志系统
    
    Args:
        log_file_path: 日志文件路径（可选）
        console_level: 控制台日志级别
        file_level: 文件日志级别
        use_colors: 是否在控制台使用彩色输出
    
    Returns:
        配置好的 logger
    """
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        fmt="%(levelname)s - %(message)s",
        use_colors=use_colors
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)
    
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    
    try:
        from agno.utils.log import configure_agno_logging
        configure_agno_logging(custom_default_logger=logger)
    except ImportError:
        pass
    
    return logger


def add_file_handler(logger, log_file_path: str, level: int = logging.INFO):
    """为已存在的 logger 添加文件 handler"""
    import os
    
    log_file_path_abs = os.path.abspath(log_file_path)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(handler.baseFilename) == log_file_path_abs:
                return
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

