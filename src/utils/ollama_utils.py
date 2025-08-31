import re
import ollama
import logging

def get_model_size_in_billions(model_details: dict) -> float:
    try:
        size_str = model_details.get('parameter_size', '').upper()
        if 'B' in size_str:
            return float(re.findall(r"(\d+\.?\d*)", size_str)[0])
        elif 'M' in size_str:
            return float(re.findall(r"(\d+\.?\d*)", size_str)[0]) / 1000.0
    except (IndexError, ValueError):
        pass
    
    model_name = model_details.get('model', '')
    model_name = model_name.lower()
    match = re.search(r'(\d+)b', model_name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
            
    print(f"警告：無法從模型 '{model_name}' 中自動解析參數大小。")
    return 0.0
    
def get_local_ollama_models() -> list[dict]:
    logger = logging.getLogger(__name__)
    logger.info("Attempting to fetch local models from Ollama service...")
    try:
        response = ollama.list()
        logger.info(f"Received response from ollama.list(): {response}")
        
        models = response.get('models', [])
        if not models:
            logger.warning("ollama.list() returned an empty model list or no 'models' key.")
        
        return models

    except ollama.ResponseError as e:
        logger.error(f"Error response from Ollama service when listing models. Status code: {e.status_code}")
        try:
            logger.error(f"Response body: {e.response.text}")
        except Exception:
            pass
        print(f"錯誤：無法從 Ollama 獲取模型列表。錯誤碼: {e.status_code}")
        print("請確認 Ollama 服務正在運行並且可以訪問。")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred when calling ollama.list(): {e}", exc_info=True)
        print(f"發生未知錯誤：{e}")
        print("請檢查你的網路連線以及 Ollama 服務狀態。")
        return []
