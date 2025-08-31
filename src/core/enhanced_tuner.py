#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ollama
import time
import multiprocessing
import sys
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from queue import Empty

from src.utils.memory_monitor import get_memory_monitor
from src.utils.cache_manager import get_cache_manager
from src.models.bayesian_optimizer import create_optimizer_for_quality_tuning, AdaptiveBayesianOptimizer
from src.core.enhanced_evaluator import get_enhanced_evaluator, get_evaluation_dataset
from evaluation_dataset import HALLUCINATION_EVAL_SET, LONG_CONTEXT_PERFORMANCE_PROMPT

class EnhancedOllamaTuner:  
    def __init__(self, model_name: str, constraints: Optional[Dict] = None):
        """
        初始化增強調校器
        
        Args:
            model_name: 模型名稱
            constraints: 約束條件
        """
        self.model_name = model_name
        self.constraints = constraints or {
            "time_limit_s": 60.0,
            "ttft_limit_s": 5.0,
            "hallucination_threshold": 0.95,
            "num_predict": 256
        }
        self.memory_monitor = get_memory_monitor()
        self.cache_manager = get_cache_manager()
        self.evaluator = get_enhanced_evaluator()
        self.evaluation_dataset = get_evaluation_dataset()
        self.optimizer = create_optimizer_for_quality_tuning()
        self.best_settings = {}
        self.current_process = None
        self.logger = logging.getLogger(__name__)
        self.memory_monitor.start_monitoring(interval=2.0)
        
        self.logger.info(f"增強調校器已初始化: {model_name}")
    
    def _safe_generate(self, prompt: str, settings: Dict, timeout: float) -> Tuple[Any, str]:
        try:
            clean_settings = {}
            for key, value in settings.items():
                if key not in ['stream'] and isinstance(value, (int, float, str, bool)):
                    clean_settings[key] = value
            if settings.get('stream', False):
                start_time = time.time()
                stream = ollama.generate(model=self.model_name, prompt=prompt, options=clean_settings, stream=True)
                first_token_time = None
                token_count = 0
                
                for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                    token_count += 1
                
                end_time = time.time()
                if first_token_time is None:
                    result = (float('inf'), 0, float('inf'))
                else:
                    ttft = first_token_time - start_time
                    total_duration = end_time - start_time
                    tps = (token_count - 1) / (total_duration - ttft) if (total_duration - ttft) > 0 else 0
                    result = (ttft, tps, total_duration)
                return result, "success"
            else:
                response = ollama.generate(model=self.model_name, prompt=prompt, options=clean_settings, stream=False)
                return response['response'], "success"
                
        except Exception as e:
            return None, str(e)
    
    def _check_memory_safety(self) -> bool:
        if not self.memory_monitor.is_memory_safe():
            self.logger.warning("記憶體使用率過高，暫停測試")
            return False
        return True
    
    def _get_cached_result(self, parameters: Dict) -> Optional[Dict]:
        return self.cache_manager.get(self.model_name, parameters)
    
    def _cache_result(self, parameters: Dict, result: Dict):
        self.cache_manager.set(self.model_name, parameters, result)
    
    def _evaluate_quality_comprehensive(self, settings: Dict) -> Dict[str, float]:
        scores = []
        evaluations = []
        test_prompt = "請回答：1+1等於多少？"
        test_settings = settings.copy()
        test_settings['stream'] = False
        
        answer, status = self._safe_generate(test_prompt, test_settings, timeout=30.0)
        
        if status != "success" or "does not support generate" in str(answer):
            return {"overall": 0.0, "hallucination": 0.0, "error": "incompatible"}
        for item in HALLUCINATION_EVAL_SET:
            prompt = f"請參考以下資訊來回答問題。\n\n上下文：{item['context']}\n\n問題：{item['question']}"
            
            answer, status = self._safe_generate(prompt, test_settings, timeout=60.0)
            
            if status == "success" and isinstance(answer, str):
                evaluation = self.evaluator.comprehensive_evaluation(
                    context=item['context'],
                    question=item['question'],
                    answer=answer,
                    ground_truth_keywords=item['ground_truth_keywords']
                )
                evaluations.append(evaluation)
                scores.append(evaluation['overall'])
            else:
                self.logger.warning(f"評估項目 '{item['id']}' 失敗 ({status})")
                scores.append(0.0)
        
        if not scores:
            return {"overall": 0.0, "hallucination": 0.0, "error": "no_valid_tests"}
        avg_score = sum(scores) / len(scores)
        avg_evaluation = {}
        if evaluations:
            for key in evaluations[0].keys():
                if key != 'overall':
                    avg_evaluation[key] = sum(e[key] for e in evaluations) / len(evaluations)
        
        result = {
            "overall": avg_score,
            "hallucination": avg_evaluation.get('hallucination', 0.0),
            "relevance": avg_evaluation.get('relevance', 0.0),
            "logical_consistency": avg_evaluation.get('logical_consistency', 0.0),
            "factual_accuracy": avg_evaluation.get('factual_accuracy', 0.0),
            "creativity": avg_evaluation.get('creativity', 0.0),
            "multilingual_support": avg_evaluation.get('multilingual_support', 0.0),
            "completeness": avg_evaluation.get('completeness', 0.0),
            "fluency": avg_evaluation.get('fluency', 0.0)
        }
        
        return result
    
    def tune_quality_bayesian(self) -> str:
        self.logger.info("開始貝葉斯優化品質調校")
        
        iteration = 0
        max_iterations = 25
        
        while iteration < max_iterations:
            if not self._check_memory_safety():
                time.sleep(5)
                continue
            next_params = self.optimizer.suggest_next_point()
            if next_params is None:
                self.logger.info("貝葉斯優化建議早停")
                break
            
            iteration += 1
            self.logger.info(f"迭代 {iteration}/{max_iterations}: 測試參數 {next_params}")
            cached_result = self._get_cached_result(next_params)
            if cached_result:
                self.logger.info(f"使用緩存結果: {cached_result['overall']:.4f}")
                self.optimizer.update(next_params, cached_result['overall'])
                continue
            evaluation_result = self._evaluate_quality_comprehensive(next_params)
            
            if 'error' in evaluation_result:
                self.logger.error(f"評估失敗: {evaluation_result['error']}")
                if evaluation_result['error'] == 'incompatible':
                    return "incompatible"
                continue
            
            score = evaluation_result['overall']
            self._cache_result(next_params, evaluation_result)
            self.optimizer.update(next_params, score)          
            self.logger.info(f"評分: {score:.4f}")
            if self.optimizer.should_stop_early():
                self.logger.info("觸發早停條件")
                break
        best_params, best_score = self.optimizer.get_best_result()
        
        if best_params:
            self.best_settings.update(best_params)
            self.best_settings['hallucination_score'] = round(best_score, 4)
            detailed_eval = self._evaluate_quality_comprehensive(best_params)
            self.best_settings['detailed_evaluation'] = detailed_eval
            
            self.logger.info(f"最佳品質設定: {best_params} (評分: {best_score:.4f})")
        
        return "continue"
    
    def tune_context_window(self, ctx_options: List[int] = None) -> bool:
        if ctx_options is None:
            ctx_options = [8192, 4096, 2048, 1024]
        
        self.logger.info("開始上下文窗口調校")
        
        for ctx_size in ctx_options:
            if not self._check_memory_safety():
                time.sleep(5)
                continue
            
            self.logger.info(f"測試 num_ctx = {ctx_size}")
            
            settings = self.best_settings.copy()
            settings.update({'num_ctx': ctx_size, 'num_predict': self.constraints['num_predict']})
            cached_result = self._get_cached_result(settings)
            if cached_result and 'performance' in cached_result:
                ttft, tps, duration = cached_result['performance']
                self.logger.info(f"使用緩存性能結果: TTFT={ttft*1000:.0f}ms, TPS={tps:.2f}")
            else:
                ttft, tps, duration = self._measure_speed(LONG_CONTEXT_PERFORMANCE_PROMPT, settings)
                self._cache_result(settings, {'performance': (ttft, tps, duration)})
            
            self.logger.info(f"性能: 總時間={duration:.2f}s, TTFT={ttft*1000:.0f}ms, TPS={tps:.2f}")
            
            if duration < self.constraints['time_limit_s'] and ttft < self.constraints['ttft_limit_s']:
                self.best_settings['num_ctx'] = ctx_size
                self.logger.info(f"找到可接受的 num_ctx: {ctx_size}")
                return True
        self.best_settings['num_ctx'] = ctx_options[-1]
        self.logger.warning(f"使用最小 num_ctx: {ctx_options[-1]}")
        return False
    
    def tune_gpu_layers(self) -> Tuple[bool, Dict]:
        self.logger.info("開始 GPU 層數調校")
        
        low, high = 0, 101
        best_working_gpu = 0
        
        while low < high:
            if not self._check_memory_safety():
                time.sleep(5)
                continue
            
            mid = low + (high - low) // 2
            if mid == low:
                mid = high
            if mid == high and mid == best_working_gpu:
                break
            
            self.logger.info(f"二分搜尋測試 num_gpu = {mid}")
            
            settings = self.best_settings.copy()
            settings.update({'num_gpu': mid, 'num_predict': self.constraints['num_predict']})
            cached_result = self._get_cached_result(settings)

            if cached_result and 'performance' in cached_result:
                ttft, tps, duration = cached_result['performance']
            else:
                ttft, tps, duration = self._measure_speed(LONG_CONTEXT_PERFORMANCE_PROMPT, settings)
                self._cache_result(settings, {'performance': (ttft, tps, duration)})
            
            self.logger.info(f"性能: 總時間={duration:.2f}s, TTFT={ttft*1000:.0f}ms, TPS={tps:.2f}")
            
            if duration < self.constraints['time_limit_s'] and ttft < self.constraints['ttft_limit_s']:
                best_working_gpu = mid
                low = mid
                self.logger.info(f"成功，嘗試更高值")
            else:
                high = mid - 1
                self.logger.info(f"失敗，嘗試更低值")
        
        self.best_settings['num_gpu'] = best_working_gpu
        settings = self.best_settings.copy()
        settings['num_predict'] = self.constraints['num_predict']
        
        ttft, tps, duration = self._measure_speed(LONG_CONTEXT_PERFORMANCE_PROMPT, settings)
        final_performance = {'ttft': ttft, 'tps': tps, 'duration': duration}
        
        if duration < self.constraints['time_limit_s'] and ttft < self.constraints['ttft_limit_s']:
            self.logger.info(f"最終確認成功: num_gpu = {best_working_gpu}")
            return True, final_performance
        else:
            self.logger.warning("最終確認失敗，使用 CPU")
            self.best_settings['num_gpu'] = 0
            settings['num_gpu'] = 0
            ttft, tps, duration = self._measure_speed(LONG_CONTEXT_PERFORMANCE_PROMPT, settings)
            final_performance = {'ttft': ttft, 'tps': tps, 'duration': duration}
            return False, final_performance
    
    def _measure_speed(self, prompt: str, settings: Dict) -> Tuple[float, float, float]:
        process_settings = settings.copy()
        process_settings['stream'] = True
        timeout = self.constraints['time_limit_s'] + 10.0
        
        result, status = self._safe_generate(prompt, process_settings, timeout=timeout)
        
        if status == "success":
            return result
        else:
            self.logger.warning(f"速度測試失敗: {status}")
            return float('inf'), 0, float('inf')
    
    def run(self) -> Optional[Dict]:
        try:
            self.logger.info(f"開始為模型 '{self.model_name}' 進行增強調校")
            quality_status = self.tune_quality_bayesian()
            if quality_status == "incompatible":
                return "incompatible"
            self.tune_context_window()
            success, final_performance = self.tune_gpu_layers()
            final_settings = self.best_settings.copy()
            final_settings['num_predict'] = self.constraints['num_predict']
            
            if 'num_gpu' in final_settings and final_settings['num_gpu'] == 101:
                final_settings['num_gpu'] = 100
            optimization_history = self.optimizer.get_optimization_history()
            parameter_importance = self.optimizer.get_parameter_importance()
            memory_summary = self.memory_monitor.get_memory_summary()
            cache_stats = self.cache_manager.get_cache_stats()
            
            result = {
                "model_name": self.model_name,
                "optimal_settings": final_settings,
                "constraints": self.constraints,
                "final_performance": final_performance,
                "optimization_history": optimization_history,
                "parameter_importance": parameter_importance,
                "memory_summary": memory_summary,
                "cache_stats": cache_stats
            }
            
            self.logger.info("增強調校完成")
            return result
            
        except KeyboardInterrupt:
            self.logger.info("接收到中斷信號，正在清理...")
            if self.current_process and self.current_process.is_alive():
                self.current_process.terminate()
                self.current_process.join()
            return None
        except Exception as e:
            self.logger.error(f"調校過程中發生錯誤: {e}")
            return None
        finally:
            self.memory_monitor.stop_monitoring()
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        return {
            "parameter_importance": self.optimizer.get_parameter_importance(),
            "optimization_history": self.optimizer.get_optimization_history(),
            "memory_usage": self.memory_monitor.get_memory_summary(),
            "cache_efficiency": self.cache_manager.get_cache_stats()
        }
