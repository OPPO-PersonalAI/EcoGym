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

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class SessionManager:
    """
    管理工作流会话的持久化和恢复
    使用 JSON/JSONL 格式存储会话数据
    """
    
    def __init__(self, session_id: str, base_dir: str = "logs/sessions"):
        """
        初始化会话管理器
        
        Args:
            session_id: 会话唯一标识符
            base_dir: 会话数据存储的基础目录
        """
        self.logger = logging.getLogger("agno_stimulation")
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / session_id
        
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.session_dir / "metadata.json"
        self.steps_file = self.session_dir / "steps.jsonl"
        self.state_file = self.session_dir / "state.json"
        
        self.logger.info(f"Session manager initialized - Session ID: {session_id}")
        self.logger.info(f"Session directory: {self.session_dir}")
    
    def init_session(self, config: Dict[str, Any], initial_state: Dict[str, Any]) -> None:
        """
        初始化新会话的元数据
        
        Args:
            config: 配置信息
            initial_state: 初始状态
        """
        metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "last_step": 0,
            "total_steps": 0,
            "status": "running",
            "config": config,
            "initial_state": initial_state
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.save_state(initial_state, step=0)
        
        self.logger.info(f"Session initialized - metadata saved to: {self.metadata_file}")
    
    def save_step(self, step_data: Dict[str, Any]) -> None:
        """
        保存单个 step 的数据到 JSONL 文件
        
        Args:
            step_data: 包含 step 所有信息的字典
        """
        step_data['timestamp'] = datetime.now().isoformat()
        
        with open(self.steps_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(step_data, ensure_ascii=False) + '\n')
        
        self._update_metadata(step_data['step'], step_data.get('is_finished', False))
        
        self.logger.debug(f"Step {step_data['step']} data saved")
    
    def save_state(self, state: Dict[str, Any], step: int) -> None:
        """
        保存当前状态到状态文件（便于快速恢复）
        
        Args:
            state: 当前状态
            step: 当前步骤编号
        """
        state_snapshot = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "state": state
        }
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state_snapshot, f, ensure_ascii=False, indent=2)
        
        self.logger.debug(f"State saved - Step {step}")
    
    def _update_metadata(self, step: int, is_finished: bool = False) -> None:
        """
        更新元数据文件
        
        Args:
            step: 当前步骤编号
            is_finished: 是否完成
        """
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        metadata['last_step'] = step
        metadata['total_steps'] = step
        metadata['last_update'] = datetime.now().isoformat()
        if is_finished:
            metadata['status'] = 'completed'
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def mark_interrupted(self) -> None:
        """标记会话为中断状态"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            metadata['status'] = 'interrupted'
            metadata['interrupted_time'] = datetime.now().isoformat()
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Session marked as interrupted")
    
    def mark_completed(self) -> None:
        """标记会话为完成状态"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            metadata['status'] = 'completed'
            metadata['completed_time'] = datetime.now().isoformat()
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Session marked as completed")
    
    def update_cost_info(
        self,
        total_input_tokens: int,
        total_output_tokens: int,
        total_cost: float,
        model_name: str,
        pricing: Optional[Dict[str, float]] = None
    ) -> None:
        """
        更新成本信息到元数据
        
        Args:
            total_input_tokens: 总输入 token 数
            total_output_tokens: 总输出 token 数
            total_cost: 总成本
            model_name: 模型名称
            pricing: 定价信息（可选），包含 input 和 output 价格
        """
        if not self.metadata_file.exists():
            return
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        metadata['cost_info'] = {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'total_cost': total_cost,
            'model_name': model_name,
        }
        
        if pricing:
            metadata['cost_info']['pricing'] = pricing
        
        metadata['last_update'] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.debug(f"Cost info updated in metadata: Total Cost = ${total_cost:.6f}")
    
    def update_final_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        更新最终评估指标到元数据
        
        Args:
            metrics: 最终评估指标字典
        """
        if not self.metadata_file.exists():
            return
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        metadata['final_metrics'] = metrics
        
        metadata['last_update'] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Final evaluation metrics saved to metadata")
        self.logger.debug(f"Final metrics: {metrics}")
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        加载会话元数据
        
        Returns:
            元数据字典，如果不存在则返回 None
        """
        if not self.metadata_file.exists():
            return None
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        加载最后保存的状态
        
        Returns:
            状态快照字典，如果不存在则返回 None
        """
        if not self.state_file.exists():
            return None
        
        with open(self.state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_all_steps(self) -> List[Dict[str, Any]]:
        """
        加载所有 step 的数据
        
        Returns:
            包含所有 step 数据的列表
        """
        if not self.steps_file.exists():
            return []
        
        steps = []
        with open(self.steps_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line))
        
        return steps
    
    def can_resume(self) -> bool:
        """
        检查会话是否可以恢复
        
        Returns:
            如果会话可以恢复返回 True
        """
        metadata = self.load_metadata()
        if not metadata:
            return False
        
        status = metadata.get('status', '')
        return status in ['interrupted', 'running']
    
    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """
        获取恢复所需的信息
        
        Returns:
            包含恢复信息的字典：last_step, state, metadata
        """
        if not self.can_resume():
            return None
        
        metadata = self.load_metadata()
        state_snapshot = self.load_state()
        
        if not metadata or not state_snapshot:
            return None
        
        actual_completed_step = state_snapshot.get('step', 0)
        metadata_last_step = metadata.get('last_step', 0)
        
        if actual_completed_step != metadata_last_step:
            self.logger.info(
                f"Resume: state.json step ({actual_completed_step}) differs from "
                f"metadata.json last_step ({metadata_last_step}). "
                f"Using state.json step as it represents the fully saved state."
            )
        
        return {
            'last_step': actual_completed_step,
            'state': state_snapshot['state'],
            'metadata': metadata,
            'session_id': self.session_id
        }
    
    @staticmethod
    def list_resumable_sessions(base_dir: str = "logs/sessions") -> List[Dict[str, Any]]:
        """
        列出所有可恢复的会话
        
        Args:
            base_dir: 会话数据存储的基础目录
        
        Returns:
            可恢复会话的列表，每个元素包含 session_id 和元数据
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        resumable = []
        for session_dir in base_path.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    status = metadata.get('status', '')
                    if status in ['interrupted', 'running']:
                        resumable.append({
                            'session_id': session_dir.name,
                            'metadata': metadata
                        })
        
        resumable.sort(key=lambda x: x['metadata'].get('last_update', ''), reverse=True)
        return resumable

