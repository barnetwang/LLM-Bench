#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import logging
import time
import threading
import argparse
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.ollama_utils import get_model_size_in_billions, get_local_ollama_models
from src.core.enhanced_tuner import EnhancedOllamaTuner
from src.utils.memory_monitor import get_memory_monitor
from src.utils.cache_manager import get_cache_manager
from src.ui.web_interface import get_web_ui

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ollama_tuner_web.log', encoding='utf-8')
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

class WebIntegratedTuner:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.web_ui = get_web_ui()
        self.tuning_thread = None
        self.is_tuning = False
        self.stop_event = threading.Event()
        self.all_results = []
        
        self.memory_monitor = get_memory_monitor()
        self.cache_manager = get_cache_manager()
        
        self.memory_monitor.start_monitoring(interval=2.0)
        
        self.web_ui.socketio.on('start_tuning')(self.start_tuning_from_web)
        self.web_ui.socketio.on('stop_tuning')(self.stop_tuning_from_web)
        self.web_ui.socketio.on('generate_report')(self.generate_report_from_web)

    def start_web_ui(self):
        web_thread = threading.Thread(target=lambda: self.web_ui.run(host=self.host, port=self.port), daemon=True)
        web_thread.start()
        logger.info(f"🌐 Web UI 已啟動: http://{self.host}:{self.port}")

    def start_monitoring_thread(self):
        def monitor():
            while True:
                try:
                    memory_data = self.memory_monitor.get_memory_summary()
                    self.web_ui.update_memory_usage(memory_data)
                    cache_data = self.cache_manager.get_cache_stats()
                    self.web_ui.update_cache_stats(cache_data)
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"監控線程錯誤: {e}", exc_info=True)
                    time.sleep(5)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def generate_report_from_web(self):
        self.web_ui.add_log_message('info', "正在生成完整報告...")
        report_filename = self.generate_enhanced_html_report()
        if report_filename:
            self.web_ui.add_log_message('info', f"報告已生成: {report_filename}")
            self.web_ui.socketio.emit('report_generated', {'filename': report_filename})
        else:
            self.web_ui.add_log_message('error', "報告生成失敗")

    def generate_enhanced_html_report(self):
        from jinja2 import Environment, FileSystemLoader
        report_filename = 'enhanced_ollama_tuner_report.html'
        try:
            if not self.all_results:
                self.web_ui.add_log_message('warning', "沒有結果可供生成報告。")
                return None
            
            env = Environment(loader=FileSystemLoader('.'))
            template = env.get_template('enhanced_report_template.html')
            
            report_data = {
                'results': self.all_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'memory_summary': self.memory_monitor.get_memory_summary(),
                'cache_stats': self.cache_manager.get_cache_stats()
            }
            html_content = template.render(report_data)
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return report_filename
        except Exception as e:
            logger.error(f"生成 HTML 報告時發生錯誤: {e}", exc_info=True)
            return None

    def select_constraints_by_size(self, model_data: Dict) -> Dict:
        model_name = model_data.get('model')
        size_b = get_model_size_in_billions(model_data.get('details', {}))
        self.web_ui.add_log_message('info', f"偵測到模型 '{model_name}' 的大小約為 {size_b:.2f}B")
        if 0 < size_b < 10: return {"time_limit_s": 30.0, "ttft_limit_s": 2.0, "num_predict": 1024}
        elif 10 <= size_b <= 18: return {"time_limit_s": 120.0, "ttft_limit_s": 6.0, "num_predict": 2048}
        elif size_b > 18: return {"time_limit_s": 180.0, "ttft_limit_s": 15.0, "num_predict": 8192}
        else: return {"time_limit_s": 60.0, "ttft_limit_s": 5.0, "num_predict": 256}

    def start_tuning_from_web(self, data):
        if self.is_tuning: return self.web_ui.add_log_message('warning', '調校已在運行中')
        self.all_results = []
        self.stop_event.clear()
        model_to_run = data.get('model_name') if data and data.get('model_name') else None
        self.is_tuning = True
        self.tuning_thread = threading.Thread(target=self.run_tuning_logic, args=(model_to_run,), daemon=True)
        self.tuning_thread.start()

    def stop_tuning_from_web(self):
        if self.is_tuning:
            self.stop_event.set()
            self.is_tuning = False
            self.web_ui.set_status('stopped', '正在停止調校...'); 
            logger.info("收到停止調校信號")

    def run_tuning_logic(self, target_model: str = None):
        try:
            self.web_ui.set_status('running', "正在檢查 Ollama 服務...")
            all_models = get_local_ollama_models()
            if not all_models: return self.web_ui.set_status('error', "未找到任何本地 Ollama 模型")

            models_to_run = all_models
            if target_model: models_to_run = [m for m in all_models if m.get('model') == target_model]
            if not models_to_run: return self.web_ui.set_status('error', f"找不到指定的模型: {target_model}")

            self.web_ui.add_log_message('info', f"將為 {len(models_to_run)} 個模型進行調校: {[m.get('model') for m in models_to_run]}")
            
            for i, model_data in enumerate(models_to_run):
                if not self.is_tuning: self.web_ui.add_log_message('info', "調校已手動停止。"); break
                model_name = model_data.get('model')
                self.web_ui.set_status('running', f"正在調校模型 {i+1}/{len(models_to_run)}: {model_name}")
                
                tuner = EnhancedOllamaTuner(
                    model_name=model_name, 
                    constraints=self.select_constraints_by_size(model_data),
                    stop_event=self.stop_event
                )
                
                result = tuner.run()

                if self.stop_event.is_set():
                    self.web_ui.add_log_message('info', f"模型 '{model_name}' 的調校已中斷。")
                    break

                if result and result != "incompatible":
                    # 修正: 在發送到前端前，轉換 NumPy 型別
                    cleaned_result = convert_numpy_types(result)
                    self.all_results.append(cleaned_result)
                    self.web_ui.add_tuning_result(cleaned_result)
                elif result == "incompatible": 
                    self.web_ui.add_log_message('warning', f"模型 '{model_name}' 不支援生成功能")
                else: 
                    self.web_ui.add_log_message('error', f"模型 '{model_name}' 調校失敗")

            if self.is_tuning: self.web_ui.set_status('completed', "所有模型調校完成")
            else: self.web_ui.set_status('stopped', "調校已停止")
        except Exception as e:
            self.web_ui.set_status('error', f"調校過程中發生嚴重錯誤: {e}")
            logger.error("調校過程中發生嚴重錯誤", exc_info=True)
        finally: self.is_tuning = False

    def run(self):
        print("🚀 增強的 Ollama Auto-Tuner (Web UI 版本)")
        print("=" * 50)      
        self.start_web_ui()
        self.start_monitoring_thread()
        time.sleep(1)
        self.web_ui.add_log_message('info', "正在獲取本地模型列表...")
        all_local_models = get_local_ollama_models()
        if all_local_models:
            model_names = [m.get('model') for m in all_local_models if m.get('model')]
            self.web_ui.set_available_models(model_names)
            self.web_ui.add_log_message('info', f"偵測到 {len(model_names)} 個模型。")
        else:
            self.web_ui.add_log_message('error', "無法獲取本地模型列表，請檢查 Ollama 服務。")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到中斷信號，正在關閉...")
            if self.memory_monitor:
                self.memory_monitor.stop_monitoring()
            logger.info("程式執行完成")

def main():
    parser = argparse.ArgumentParser(description='增強的 Ollama Auto-Tuner (Web UI 版本)')
    parser.add_argument('--port', type=int, default=5000, help='Web UI 端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web UI 主機')
    args = parser.parse_args()
    tuner = WebIntegratedTuner(host=args.host, port=args.port)
    tuner.run()

if __name__ == "__main__":
    main()