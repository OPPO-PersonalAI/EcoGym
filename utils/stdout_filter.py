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

import sys
from typing import TextIO


class FilteredStdout:
    """过滤掉特定内容的 stdout 包装器"""
    
    def __init__(self, original_stdout: TextIO, filters: list = None):
        """
        初始化过滤器
        
        Args:
            original_stdout: 原始的 stdout
            filters: 过滤条件列表，如果输出包含这些字符串则被过滤
        """
        self.original_stdout = original_stdout
        self.filters = filters or [
            "<class 'agno.models.message.Message'>",
        ]
        self.buffer = ""
        self.filtering = False
        self.newline_count = 0
    
    def write(self, text: str) -> int:
        """写入时过滤特定内容"""
        should_filter = False
        for filter_str in self.filters:
            if filter_str in text:
                should_filter = True
                self.filtering = True
                self.newline_count = 0
                break
        
        if self.filtering:
            if "\n" in text:
                self.newline_count += text.count("\n")
                if self.newline_count >= 2:
                    self.filtering = False
                    self.newline_count = 0
                    return len(text)
            return len(text)
        
        if should_filter:
            return len(text)
        
        return self.original_stdout.write(text)
    
    def flush(self):
        """刷新缓冲区"""
        self.original_stdout.flush()
    
    def __getattr__(self, name):
        """代理其他属性到原始 stdout"""
        return getattr(self.original_stdout, name)


def install_stdout_filter():
    """安装 stdout 过滤器"""
    if not isinstance(sys.stdout, FilteredStdout):
        sys.stdout = FilteredStdout(sys.stdout)


def uninstall_stdout_filter():
    """卸载 stdout 过滤器，恢复原始 stdout"""
    if isinstance(sys.stdout, FilteredStdout):
        sys.stdout = sys.stdout.original_stdout

