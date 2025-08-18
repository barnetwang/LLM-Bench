import ollama
import time
import pandas as pd
import itertools
import multiprocessing
import sys
import os
import webbrowser

from queue import Empty
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from evaluation_dataset import HALLUCINATION_EVAL_SET, LONG_CONTEXT_PERFORMANCE_PROMPT
from utils import get_model_size_in_billions, get_local_ollama_models


def ollama_generate_worker(queue, model_name, prompt, options):
    try:
        if options.get('stream', False):
            start_time = time.time()
            stream = ollama.generate(model=model_name, prompt=prompt, options=options, stream=True)
            first_token_time = None; token_count = 0
            for chunk in stream:
                if first_token_time is None: first_token_time = time.time()
                token_count += 1
            end_time = time.time()
            if first_token_time is None:
                result = (float('inf'), 0, float('inf'))
            else:
                ttft = first_token_time - start_time
                total_duration = end_time - start_time
                tps = (token_count - 1) / (total_duration - ttft) if (total_duration - ttft) > 0 else 0
                result = (ttft, tps, total_duration)
        else:
            response = ollama.generate(model=model_name, prompt=prompt, options=options, stream=False)
            result = response['response']
        queue.put(result)
    except Exception as e:
        queue.put(e)

def select_constraints_by_size(model_name: str):
    model_name = model_data.get('name') or model_data.get('model')
    size_b = get_model_size_in_billions(model_data['details'])
    print(f"偵測到模型 '{model_name}' 的大小約為 {size_b:.2f}B。")
    chat_constraints = {"name": "小型模型 / 聊天場景", "time_limit_s": 30.0, "ttft_limit_s": 2.0, "hallucination_threshold": 0.95, "num_predict": 128}
    summary_constraints = {"name": "中型模型 / 摘要場景", "time_limit_s": 120.0, "ttft_limit_s": 6.0, "num_predict": 512, "hallucination_threshold": 1.0}
    large_model_constraints = {"name": "大型模型 / 複雜任務", "time_limit_s": 180.0, "ttft_limit_s": 15.0, "hallucination_threshold": 0.95, "num_predict": 256}
    if 0 < size_b < 10: return chat_constraints
    elif 10 <= size_b <= 18: return summary_constraints
    elif size_b > 18: return large_model_constraints
    else:
        print("使用通用預設約束條件。")
        return chat_constraints

def generate_html_report(results_data):
    report_filename = 'ollama_tuner_report.html'   
    try:
        if os.path.exists(report_filename):
            os.remove(report_filename)
            print(f"ⓘ 已刪除舊的報告檔案：{report_filename}")

        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('report_template.html')
        report_data = {'results': results_data, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        html_content = template.render(report_data)
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)           
        print(f"\n✅ HTML 報告已成功生成：{report_filename}")
        
        try:
            webbrowser.open('file://' + os.path.abspath(report_filename))
            print("ⓘ 正在你的預設瀏覽器中打開報告...")
        except Exception as e:
            print(f"❌ 無法自動打開瀏覽器，請手動打開檔案。錯誤: {e}")

    except Exception as e:
        print(f"\n❌ 生成 HTML 報告時發生錯誤: {e}")

class OllamaAutoTuner:
    def __init__(self, model_name, constraints=None):
        self.model_name = model_name
        self.constraints = {
            "time_limit_s": 60.0, "ttft_limit_s": 5.0,
            "hallucination_threshold": 0.95, "num_predict": 256
        }
        if constraints: self.constraints.update(constraints)
        self.best_settings = {}
        self.current_process = None
        print("--- 自動調校器已初始化 ---")
        print("模型:", self.model_name)
        print("約束條件:", self.constraints)

    def _safe_generate(self, prompt, settings, timeout):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=ollama_generate_worker, args=(q, self.model_name, prompt, settings))
        try:
            self.current_process = p
            p.start()
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                return None, "timeout"
            result = q.get_nowait()
            if isinstance(result, Exception):
                return None, str(result)
            return result, "success"
        except Empty:
            return None, "empty_queue"
        finally:
            self.current_process = None

    def _evaluate_hallucination(self, settings):
        scores = []
        is_incompatible = False
        for item in HALLUCINATION_EVAL_SET:
            prompt = f"請參考以下資訊來回答問題。\n\n上下文：{item['context']}\n\n問題：{item['question']}"
            process_settings = settings.copy(); process_settings['stream'] = False
            answer, status = self._safe_generate(prompt, process_settings, timeout=60.0)
            
            if status != "success" and "does not support generate" in status:
                is_incompatible = True
                break

            if status == "success" and isinstance(answer, str):
                if any(keyword in answer.lower() for keyword in item['ground_truth_keywords']):
                    scores.append(1.0)
                else: scores.append(0.0)
            else:
                print(f"  -> 警告：幻覺檢測 item '{item['id']}' 失敗 ({status})，計為 0 分。")
                scores.append(0.0)
        
        if is_incompatible:
            return None          
        return sum(scores) / len(scores) if scores else 0.0

    def _measure_speed(self, prompt, settings):
        process_settings = settings.copy(); process_settings['stream'] = True
        timeout = self.constraints['time_limit_s'] + 10.0
        result, status = self._safe_generate(prompt, process_settings, timeout=timeout)
        if status == "success":
            return result
        else:
            print(f"  -> 警告：速度測試失敗 ({status})。")
            return float('inf'), 0, float('inf')

    def tune_quality(self, temp_options=[1.0, 0.7, 0.5, 0.3], top_p_options=[1.0, 0.9, 0.95]):
        print("\n--- 1. 開始進行品質 (幻覺) 調校 ---")
        best_score = -1.0; best_quality_params = {}
        for temp, top_p in itertools.product(temp_options, top_p_options):
            print(f"測試組合: temperature={temp}, top_p={top_p}")
            score = self._evaluate_hallucination({"temperature": temp, "top_p": top_p, "seed": 42})
            
            if score is None:
                print(f"❌ 模型 '{self.model_name}' 不支援生成功能，已跳過。")
                return "incompatible"

            print(f"幻覺評分: {score:.2f}")
            if score > best_score:
                best_score = score
                best_quality_params = {"temperature": temp, "top_p": top_p}

        print(f"ⓘ 找到最佳品質設定: {best_quality_params} (最高分: {best_score:.2f})")
        self.best_settings.update(best_quality_params)
        self.best_settings['hallucination_score'] = round(best_score, 2)
        return "continue"

    def tune_context_window(self, ctx_options=[8192, 4096, 2048, 1024]):
        print("\n--- 2. 開始進行上下文長度 (num_ctx) 調校 ---")
        for ctx_size in ctx_options:
            print(f"正在測試 num_ctx = {ctx_size}...")
            settings = self.best_settings.copy()
            settings.update({'num_ctx': ctx_size, 'num_predict': self.constraints['num_predict']})
            ttft, tps, duration = self._measure_speed(LONG_CONTEXT_PERFORMANCE_PROMPT, settings)
            print(f"  總時間: {duration:.2f}s (上限 {self.constraints['time_limit_s']:.1f}s), 首字元時間(TTFT): {ttft*1000:.0f}ms (上限 {self.constraints['ttft_limit_s']*1000:.0f}ms)")
            if duration < self.constraints['time_limit_s'] and ttft < self.constraints['ttft_limit_s']:
                print(f"✅ 找到可接受的 num_ctx 設定: {ctx_size}")
                self.best_settings['num_ctx'] = ctx_size
                return True
        print(f"❌ 未能找到滿足時間要求的 num_ctx 設定。將使用最小值 {ctx_options[-1]}。")
        self.best_settings['num_ctx'] = ctx_options[-1]
        return False

    def tune_gpu_layers(self):
        print("\n--- 3. 開始進行 GPU 加速 (num_gpu) 調校 (高效模式) ---")
        low, high = 0, 101; best_working_gpu = 0
        while low < high:
            mid = low + (high - low) // 2
            if mid == low: mid = high
            if mid == high and mid == best_working_gpu: break
            print(f"正在二分搜尋測試 num_gpu = {mid}...")
            settings = self.best_settings.copy()
            settings.update({'num_gpu': mid, 'num_predict': self.constraints['num_predict']})
            ttft, tps, duration = self._measure_speed(LONG_CONTEXT_PERFORMANCE_PROMPT, settings)
            print(f"  總時間: {duration:.2f}s, TTFT: {ttft*1000:.0f}ms, 速度: {tps:.2f} tokens/s")
            if duration < self.constraints['time_limit_s'] and ttft < self.constraints['ttft_limit_s']:
                print(f"  -> 成功。嘗試在 [{mid}, {high-1}] 範圍內尋找更高值。")
                best_working_gpu = mid; low = mid
            else:
                print(f"  -> 失敗。在 [{low}, {mid-1}] 範圍內尋找。")
                high = mid - 1
        print(f"\n二分搜尋完成，找到的最佳候選值為 {best_working_gpu}。進行最終確認...")
        self.best_settings['num_gpu'] = best_working_gpu
        settings = self.best_settings.copy(); settings['num_predict'] = self.constraints['num_predict']
        ttft, tps, duration = self._measure_speed(LONG_CONTEXT_PERFORMANCE_PROMPT, settings)
        final_performance = {'ttft': ttft, 'tps': tps, 'duration': duration}
        if duration < self.constraints['time_limit_s'] and ttft < self.constraints['ttft_limit_s']:
             print(f"✅ 最終確認成功！最佳 num_gpu 設定為: {best_working_gpu}")
             return True, final_performance
        else:
             print(f"❌ 最終確認失敗。可能系統負載不穩定。將使用 CPU (num_gpu=0)。")
             self.best_settings['num_gpu'] = 0; settings['num_gpu'] = 0
             ttft, tps, duration = self._measure_speed(LONG_CONTEXT_PERFORMANCE_PROMPT, settings)
             final_performance = {'ttft': ttft, 'tps': tps, 'duration': duration}
             return False, final_performance

    def run(self):
        try:
            quality_status = self.tune_quality()
            if quality_status == "incompatible":
                return "incompatible"
            self.tune_context_window()
            success, final_performance = self.tune_gpu_layers()
            
            print("\n--- 最終最佳化設定 ---")
            final_settings = self.best_settings.copy()
            final_settings['num_predict'] = self.constraints['num_predict']
            print(pd.Series(final_settings).to_string())
            return {"model_name": self.model_name, "optimal_settings": final_settings,
                    "constraints": self.constraints, "final_performance": final_performance}
        except KeyboardInterrupt:
            print("\n\n🛑 接收到中斷訊號 (Ctrl+C)，正在清理並結束...")
            if self.current_process and self.current_process.is_alive():
                print(f"  -> 正在終止正在運行的子進程 (PID: {self.current_process.pid})...")
                self.current_process.terminate()
                self.current_process.join()
                print("  -> 子進程已清理。")
            print("程式已安全停止。")
            sys.exit(0)

if __name__ == "__main__":
    print("正在從本地 Ollama 偵測已安裝的模型...")
    MODELS_TO_TUNE = get_local_ollama_models()
    
    if not MODELS_TO_TUNE:
        print("\n未找到任何本地 Ollama 模型，或無法連接到 Ollama 服務。腳本已停止。")
    else:
        detected_model_names = [m.get('name') or m.get('model') for m in MODELS_TO_TUNE]
        print(f"偵測成功！將對以下模型進行調校：{detected_model_names}")
        
        all_results = []
        for model_data in MODELS_TO_TUNE:
            model_name = model_data.get('name') or model_data.get('model')
            if not model_name:
                print(f"警告：跳過一個沒有名稱的模型條目：{model_data}")
                continue
            print(f"\n{'='*20} 開始為 '{model_name}' 進行調校 {'='*20}")

            selected_constraints = select_constraints_by_size(model_data)
            
            print(f"自動選擇了 '{selected_constraints['name']}' 的約束條件。")
            tuner_constraints = selected_constraints.copy()
            del tuner_constraints['name']
            tuner = OllamaAutoTuner(model_name=model_name, constraints=tuner_constraints)
            result = tuner.run()
            
            if result and result != "incompatible":
                all_results.append(result)
            print(f"\n{'='*20} '{model_name}' 調校完成 {'='*20}\n")
            
        if all_results:
            generate_html_report(all_results)
        else:
            print("沒有任何相容的模型成功完成調校，無法生成報告。")
