import re
import ollama

def get_model_size_in_billions(model_details: dict) -> float:
    try:
        size_str = model_details.get('parameter_size', '').upper()
        if 'B' in size_str:
            return float(re.findall(r"(\d+\.?\d*)", size_str)[0])
        elif 'M' in size_str:
            return float(re.findall(r"(\d+\.?\d*)", size_str)[0]) / 1000.0
    except (IndexError, ValueError):
        pass
    
    model_name = model_details.get('name') or model_details.get('model', '')
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
    try:
        response = ollama.list()
        return response.get('models', [])
    except Exception as e:
        print(f"錯誤：無法連接到 Ollama 服務或解析模型列表: {e}")
        return []
        
    except ollama.ResponseError as e:
        print(f"錯誤：無法從 Ollama 獲取模型列表。錯誤碼: {e.status_code}")
        print("請確認 Ollama 服務正在運行並且可以訪問。")
        return []
    except Exception as e:
        print(f"發生未知錯誤：{e}")
        print("請檢查你的網路連線以及 Ollama 服務狀態。")
        return []
        
