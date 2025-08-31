#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

class CacheManager:
    def __init__(self, cache_dir: str = ".cache", max_age_hours: int = 24):
        """
        初始化緩存管理器
        
        Args:
            cache_dir: 緩存目錄
            max_age_hours: 緩存最大保存時間（小時）
        """
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)
        self._cleanup_expired_cache()
    
    def _generate_cache_key(self, model_name: str, parameters: Dict[str, Any]) -> str:
        param_str = json.dumps(parameters, sort_keys=True)
        hash_obj = hashlib.md5()
        hash_obj.update(f"{model_name}:{param_str}".encode('utf-8'))
        
        return hash_obj.hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _is_cache_valid(self, cache_data: Dict) -> bool:
        if 'timestamp' not in cache_data:
            return False
        
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        max_age = timedelta(hours=self.max_age_hours)
        
        return datetime.now() - cache_time < max_age
    
    def get(self, model_name: str, parameters: Dict[str, Any]) -> Optional[Dict]:
        """
        獲取緩存結果
        
        Args:
            model_name: 模型名稱
            parameters: 測試參數
            
        Returns:
            緩存的結果，如果不存在或已過期則返回 None
        """
        try:
            cache_key = self._generate_cache_key(model_name, parameters)
            cache_file = self._get_cache_file_path(cache_key)
            
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if not self._is_cache_valid(cache_data):
                os.remove(cache_file)
                return None
            
            self.logger.info(f"從緩存獲取結果: {model_name}")
            return cache_data.get('result')
            
        except Exception as e:
            self.logger.error(f"讀取緩存失敗: {e}")
            return None
    
    def set(self, model_name: str, parameters: Dict[str, Any], result: Dict[str, Any]):
        """
        設置緩存結果
        
        Args:
            model_name: 模型名稱
            parameters: 測試參數
            result: 測試結果
        """
        try:
            cache_key = self._generate_cache_key(model_name, parameters)
            cache_file = self._get_cache_file_path(cache_key)
            
            cache_data = {
                'model_name': model_name,
                'parameters': parameters,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"緩存結果已保存: {model_name}")
            
        except Exception as e:
            self.logger.error(f"保存緩存失敗: {e}")
    
    def clear(self, model_name: Optional[str] = None):
        """
        清理緩存
        
        Args:
            model_name: 如果指定，只清理該模型的緩存
        """
        try:
            if not os.path.exists(self.cache_dir):
                return
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    if model_name and cache_data.get('model_name') != model_name:
                        continue
                    
                    os.remove(file_path)
                    self.logger.info(f"已清理緩存: {filename}")
                    
                except Exception as e:
                    self.logger.error(f"清理緩存檔案失敗 {filename}: {e}")
                    
        except Exception as e:
            self.logger.error(f"清理緩存失敗: {e}")
    
    def _cleanup_expired_cache(self):
        try:
            if not os.path.exists(self.cache_dir):
                return
            
            expired_count = 0
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if not self._is_cache_valid(cache_data):
                        os.remove(file_path)
                        expired_count += 1
                        
                except Exception:
                    # 如果檔案損壞，直接刪除
                    os.remove(file_path)
                    expired_count += 1
            
            if expired_count > 0:
                self.logger.info(f"已清理 {expired_count} 個過期緩存檔案")
                
        except Exception as e:
            self.logger.error(f"清理過期緩存失敗: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.cache_dir):
                return {'total_files': 0, 'total_size_mb': 0, 'models': {}}
            
            total_size = 0
            model_stats = {}
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    model_name = cache_data.get('model_name', 'unknown')
                    if model_name not in model_stats:
                        model_stats[model_name] = {'count': 0, 'size': 0}
                    
                    model_stats[model_name]['count'] += 1
                    model_stats[model_name]['size'] += file_size
                    
                except Exception:
                    pass
            
            return {
                'total_files': len([f for f in os.listdir(self.cache_dir) if f.endswith('.json')]),
                'total_size_mb': total_size / (1024 * 1024),
                'models': model_stats
            }
            
        except Exception as e:
            self.logger.error(f"獲取緩存統計失敗: {e}")
            return {'error': str(e)}
    
    def get_similar_results(self, model_name: str, parameters: Dict[str, Any], 
                          tolerance: float = 0.1) -> List[Dict]:
        """
        獲取相似參數的結果
        
        Args:
            model_name: 模型名稱
            parameters: 目標參數
            tolerance: 參數差異容忍度
            
        Returns:
            相似參數的結果列表
        """
        similar_results = []
        
        try:
            if not os.path.exists(self.cache_dir):
                return similar_results
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if cache_data.get('model_name') != model_name:
                        continue
                    
                    cached_params = cache_data.get('parameters', {})

                    if self._are_parameters_similar(parameters, cached_params, tolerance):
                        similar_results.append({
                            'parameters': cached_params,
                            'result': cache_data.get('result'),
                            'timestamp': cache_data.get('timestamp')
                        })
                        
                except Exception:
                    continue
            
            similar_results.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"獲取相似結果失敗: {e}")
        
        return similar_results
    
    def _are_parameters_similar(self, params1: Dict, params2: Dict, tolerance: float) -> bool:
        if set(params1.keys()) != set(params2.keys()):
            return False
        
        for key in params1:
            val1 = params1[key]
            val2 = params2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > tolerance * max(abs(val1), abs(val2)):
                    return False
            elif val1 != val2:
                return False
        
        return True
cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    return cache_manager
