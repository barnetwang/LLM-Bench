#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.ollama_utils import get_model_size_in_billions, get_local_ollama_models
from core.enhanced_tuner import EnhancedOllamaTuner
from utils.memory_monitor import get_memory_monitor
from utils.cache_manager import get_cache_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ollama_tuner.log', encoding='utf-8')
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def select_constraints_by_size(model_data: Dict) -> Dict:
    model_name = model_data.get('name') or model_data.get('model')
    size_b = get_model_size_in_billions(model_data.get('details', {}))
    
    logger.info(f"偵測到模型 '{model_name}' 的大小約為 {size_b:.2f}B")
    
    if 0 < size_b < 10:
        return {
            "time_limit_s": 30.0,
            "ttft_limit_s": 2.0,
            "hallucination_threshold": 0.95,
            "num_predict": 1024
        }
    elif 10 <= size_b <= 18:
        return {
            "time_limit_s": 120.0,
            "ttft_limit_s": 6.0,
            "hallucination_threshold": 1.0,
            "num_predict": 2048
        }
    elif size_b > 18:
        return {
            "time_limit_s": 180.0,
            "ttft_limit_s": 15.0,
            "hallucination_threshold": 0.95,
            "num_predict": 8192
        }
    else:
        logger.warning("使用通用預設約束條件")
        return {
            "time_limit_s": 60.0,
            "ttft_limit_s": 5.0,
            "hallucination_threshold": 0.95,
            "num_predict": 256
        }

def generate_enhanced_html_report(results_data: List[Dict]):
    from jinja2 import Environment, FileSystemLoader
    
    report_filename = 'enhanced_ollama_tuner_report.html'
    
    try:
        if os.path.exists(report_filename):
            os.remove(report_filename)
            logger.info(f"已刪除舊的報告檔案：{report_filename}")
        
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('enhanced_report_template.html')
        
        report_data = {
            'results': results_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'memory_summary': get_memory_monitor().get_memory_summary(),
            'cache_stats': get_cache_manager().get_cache_stats()
        }
        
        html_content = template.render(report_data)
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"增強的 HTML 報告已生成：{report_filename}")
        
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(report_filename))
            logger.info("正在瀏覽器中打開報告...")
        except Exception as e:
            logger.error(f"無法自動打開瀏覽器：{e}")
    
    except Exception as e:
        logger.error(f"生成 HTML 報告時發生錯誤: {e}")

def print_optimization_summary(result: Dict):
    print(f"\n{'='*60}")
    print(f"模型: {result['model_name']}")
    print(f"{'='*60}")
    
    print("\n🎯 最佳設定:")
    optimal = result['optimal_settings']
    for key, value in optimal.items():
        if key != 'detailed_evaluation':
            print(f"  {key}: {value}")
    
    print("\n⚡ 性能指標:")
    perf = result['final_performance']
    print(f"  TTFT: {perf['ttft']*1000:.0f}ms")
    print(f"  TPS: {perf['tps']:.2f} tokens/s")
    print(f"  總時間: {perf['duration']:.2f}s")
    
    if 'parameter_importance' in result:
        print("\n📊 參數重要性:")
        importance = result['parameter_importance']
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {imp:.3f}")
    
    if 'optimization_history' in result:
        history = result['optimization_history']
        print(f"\n🔄 優化歷史 (共 {len(history)} 次迭代):")
        best_iteration = max(history, key=lambda x: x['score'])
        print(f"  最佳迭代: {best_iteration['iteration']} (評分: {best_iteration['score']:.4f})")
    
    if 'memory_summary' in result:
        print("\n💾 記憶體使用:")
        memory = result['memory_summary']
        if 'system_memory' in memory:
            sys_mem = memory['system_memory']
            print(f"  系統記憶體: {sys_mem['current_percent']:.1f}% (平均: {sys_mem['average_percent']:.1f}%)")
        
        if 'gpu_memory' in memory:
            for gpu_id, gpu_mem in memory['gpu_memory'].items():
                print(f"  GPU {gpu_id}: {gpu_mem['current_percent']:.1f}% (平均: {gpu_mem['average_percent']:.1f}%)")
    
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description="增強的 Ollama Auto-Tuner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=str, help="指定要進行調校的特定模型名稱。")
    parser.add_argument("--time-limit", type=float, help="自定義測試的時間限制（秒）。")
    parser.add_argument("--ttft-limit", type=float, help="自定義 TTFT 的限制（秒）。")
    parser.add_argument("--verbose", action="store_true", help="啟用詳細日誌輸出 (DEBUG level)。")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("已啟用詳細日誌輸出 (DEBUG level)。")

    print("🚀 增強的 Ollama Auto-Tuner")
    print("=" * 50)
    print("新功能:")
    print("✅ GPU/CPU 記憶體監控")
    print("✅ 智能緩存機制")
    print("✅ 貝葉斯優化算法")
    print("✅ 增強評估指標")
    print("✅ 早期停止機制")
    print("✅ 自適應參數範圍")
    print("=" * 50)
    
    # 檢查 Ollama 服務
    logger.info("正在檢查 Ollama 服務...")
    all_local_models = get_local_ollama_models()
    
    if not all_local_models:
        logger.error("未找到任何本地 Ollama 模型，或無法連接到 Ollama 服務")
        return
    
    models_to_run = all_local_models
    if args.model:
        models_to_run = [m for m in all_local_models if m.get('model') == args.model]
        if not models_to_run:
            logger.error(f"找不到指定的模型: {args.model}")
            available_models = [m.get('model') for m in all_local_models if m.get('model')]
            logger.info(f"可用的模型有: {available_models}")
            return

    model_names = [m.get('model') for m in models_to_run if m.get('model')]
    logger.info(f"將為 {len(models_to_run)} 個模型進行調校: {model_names}")
    
    memory_monitor = get_memory_monitor()
    cache_manager = get_cache_manager()
    
    memory_monitor.start_monitoring(interval=1.0)
    
    all_results = []
    start_time = time.time()
    
    try:
        for i, model_data in enumerate(models_to_run, 1):
            model_name = model_data.get('model')
            if not model_name:
                logger.warning(f"跳過沒有名稱的模型: {model_data}")
                continue
            
            print(f"\n{'='*20} 模型 {i}/{len(models_to_run)}: {model_name} {'='*20}")
            
            constraints = select_constraints_by_size(model_data)
            if args.time_limit:
                constraints['time_limit_s'] = args.time_limit
                logger.info(f"使用自定義時間限制: {args.time_limit}s")
            if args.ttft_limit:
                constraints['ttft_limit_s'] = args.ttft_limit
                logger.info(f"使用自定義 TTFT 限制: {args.ttft_limit}s")

            logger.info(f"使用約束條件: {constraints}")           
            tuner = EnhancedOllamaTuner(model_name=model_name, constraints=constraints)            
            result = tuner.run()
            
            if result and result != "incompatible":
                all_results.append(result)
                print_optimization_summary(result)
                        
            elif result == "incompatible":
                logger.warning(f"模型 '{model_name}' 不支援生成功能，已跳過")
            else:
                logger.error(f"模型 '{model_name}' 調校失敗")
            
            print(f"{'='*20} {model_name} 調校完成 {'='*20}")
        
        if all_results:
            total_time = time.time() - start_time
            logger.info(f"所有模型調校完成，總耗時: {total_time/60:.1f} 分鐘")
            
            generate_enhanced_html_report(all_results)

            cache_stats = cache_manager.get_cache_stats()
            logger.info(f"緩存統計: {cache_stats['total_files']} 個檔案, {cache_stats['total_size_mb']:.2f} MB")
            
        else:
            if args.model:
                 logger.warning(f"模型 '{args.model}' 未能成功完成調校。")
            else:
                 logger.warning("沒有任何相容的模型成功完成調校。")
    
    except KeyboardInterrupt:
        logger.info("接收到中斷信號，正在清理...")
    except Exception as e:
        logger.error(f"執行過程中發生錯誤: {e}", exc_info=args.verbose)
    finally:
        memory_monitor.stop_monitoring()
        logger.info("程式執行完成")

if __name__ == "__main__":
    main()
