#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import psutil
import time
import threading
from typing import Dict, List, Optional, Callable
import logging

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("警告：GPUtil 未安裝，GPU 監控功能將不可用")

class MemoryMonitor:
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        self.history = []
        self.logger = logging.getLogger(__name__)
        
    def get_system_memory_info(self) -> Dict:
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'free': memory.free
        }
    
    def get_gpu_memory_info(self) -> List[Dict]:
        if not GPU_AVAILABLE:
            return []
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature,
                    'load': gpu.load * 100 if gpu.load else 0
                })
            
            return gpu_info
        except Exception as e:
            self.logger.error(f"獲取 GPU 資訊失敗: {e}")
            return []
    
    def get_memory_status(self) -> Dict:
        system_memory = self.get_system_memory_info()
        gpu_memory = self.get_gpu_memory_info()
        
        status = {
            'timestamp': time.time(),
            'system_memory': system_memory,
            'gpu_memory': gpu_memory,
            'warnings': [],
            'critical': False
        }
       # if system_memory['percent'] > self.critical_threshold * 100:
       #     status['critical'] = True
       #     status['warnings'].append(f"系統記憶體使用率過高: {system_memory['percent']:.1f}%")
       # elif system_memory['percent'] > self.warning_threshold * 100:
       #     status['warnings'].append(f"系統記憶體使用率較高: {system_memory['percent']:.1f}%")
       # for gpu in gpu_memory:
       #     if gpu['memory_percent'] > self.critical_threshold * 100:
       #         status['critical'] = True
       #         status['warnings'].append(f"GPU {gpu['id']} 記憶體使用率過高: {gpu['memory_percent']:.1f}%")
       #     elif gpu['memory_percent'] > self.warning_threshold * 100:
       #         status['warnings'].append(f"GPU {gpu['id']} 記憶體使用率較高: {gpu['memory_percent']:.1f}%")
       # 
        return status
    
    def add_callback(self, callback: Callable[[Dict], None]):
        self.callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 1.0):
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("記憶體監控已啟動")
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("記憶體監控已停止")
    
    def _monitor_loop(self, interval: float):
        while self.monitoring:
            try:
                status = self.get_memory_status()
                self.history.append(status)
                if len(self.history) > 1000:
                    self.history = self.history[-500:]
                for callback in self.callbacks:
                    try:
                        callback(status)
                    except Exception as e:
                        self.logger.error(f"回調函數執行失敗: {e}")
                if status['warnings']:
                    for warning in status['warnings']:
                        if status['critical']:
                            self.logger.critical(warning)
                        else:
                            self.logger.warning(warning)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"監控循環錯誤: {e}")
                time.sleep(interval)
    
    def get_memory_summary(self) -> Dict:
        if not self.history:
            return {}
        
        recent_history = self.history[-100:]
        
        system_memory_avg = sum(h['system_memory']['percent'] for h in recent_history) / len(recent_history)
        system_memory_max = max(h['system_memory']['percent'] for h in recent_history)
        
        summary = {
            'system_memory': {
                'average_percent': system_memory_avg,
                'max_percent': system_memory_max,
                'current_percent': self.history[-1]['system_memory']['percent']
            },
            'gpu_memory': {}
        }
        
        if self.history[-1]['gpu_memory']:
            for gpu in self.history[-1]['gpu_memory']:
                gpu_id = gpu['id']
                gpu_percentages = [h['gpu_memory'][i]['memory_percent'] 
                                 for h in recent_history 
                                 for i, g in enumerate(h['gpu_memory']) 
                                 if g['id'] == gpu_id]
                
                if gpu_percentages:
                    summary['gpu_memory'][gpu_id] = {
                        'name': gpu['name'],
                        'average_percent': sum(gpu_percentages) / len(gpu_percentages),
                        'max_percent': max(gpu_percentages),
                        'current_percent': gpu['memory_percent']
                    }
        
        return summary
    
    def is_memory_safe(self) -> bool:
        system_memory = self.get_system_memory_info()
        if system_memory['percent'] > self.critical_threshold * 100:
            self.logger.critical(f"系統記憶體使用率過高: {system_memory['percent']:.1f}%，暫停以策安全。")
            return False
        return True
    
    def get_available_memory(self) -> Dict:
        system_memory = self.get_system_memory_info()
        gpu_memory = self.get_gpu_memory_info()
        
        return {
            'system_available_gb': system_memory['available'] / (1024**3),
            'system_used_gb': system_memory['used'] / (1024**3),
            'gpu_available': [
                {
                    'id': gpu['id'],
                    'name': gpu['name'],
                    'available_mb': gpu['memory_free'],
                    'used_mb': gpu['memory_used']
                }
                for gpu in gpu_memory
            ]
        }
    
memory_monitor = MemoryMonitor()

def get_memory_monitor() -> MemoryMonitor:
    return memory_monitor
