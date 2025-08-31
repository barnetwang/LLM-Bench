# 🚀 增強的 Ollama Auto-Tuner - 本地 LLM 智能優化與監控工具

增強的 Ollama Auto-Tuner 是一個使用 Python 開發的工具，旨在自動化地為您本地運行的 Ollama 大型語言模型（LLM）尋找最佳的性能與品質設定。本工具採用模組化架構，整合了多項先進技術，並提供了一個互動式的 Web UI 來即時監控調校過程。

[English Version](#-enhanced-ollama-auto-tuner---intelligent-optimization--monitoring-tool-for-local-llms)

## ✨ 核心功能

- **互動式 Web UI**：透過網頁介面即時監控系統資源、調校進度、日誌和結果，並可遠程啟動/停止調校任務。
- **智能優化算法**：使用貝葉斯優化，用更少的測試次數找到更優的參數組合。
- **早期停止機制**：在優化停滯時自動結束，節省時間和資源。
- **全方位性能優化**：包含 GPU/CPU 記憶體監控、智能緩存，確保測試流程穩定高效。
- **增強評估系統**：從回答相關性、邏輯一致性、事實準確性等多個維度進行綜合品質評分。
- **詳細視覺化報告**：測試完成後，可生成包含動態圖表的互動式 HTML 報告。

### 🏗️ 模組化架構
```
src/
├── core/               # 核心功能模組
│   ├── enhanced_tuner.py
│   └── enhanced_evaluator.py
├── models/             # 優化算法模組
│   └── bayesian_optimizer.py
├── ui/                 # Web UI 模組
│   ├── web_interface.py
│   └── templates/
│       └── index.html
└── utils/              # 工具模組
    ├── memory_monitor.py
    ├── cache_manager.py
    └── ollama_utils.py
```

## 🛠️ 安裝指南

1.  **確保 Ollama 已安裝並運行**
    請根據 [Ollama 官方網站](https://ollama.com/) 的指引安裝 Ollama，並確保其服務正在背景運行。

2.  **拉取您想測試的模型**
    ```bash
    ollama pull llama3:8b
    ```

3.  **克隆本專案** (如果您尚未操作)
    ```bash
    git clone https://github.com/your-username/ollama-auto-tuner.git
    cd ollama-auto-tuner
    ```

4.  **安裝 Python 依賴項**
    建議在虛擬環境中進行安裝：
    ```bash
    python -m venv venv
    source venv/bin/activate  # 在 Windows 上使用 `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## 🚀 如何使用

本工具提供兩種使用模式：互動式 Web UI (推薦) 和傳統的命令列介面 (CLI)。

### 方法一：互動式 Web UI (推薦)

這是最方便的使用方式，可以即時監控所有狀態。

1.  **啟動 Web 服務**：
    ```bash
    python web_ollama_autotuner.py
    ```

2.  **打開瀏覽器**：
    在瀏覽器中訪問 `http://127.0.0.1:5000` (或日誌中顯示的您的 IP 地址)。

3.  **開始調校**：
    在網頁上，您可以從下拉選單中選擇特定模型或所有模型，然後點擊「開始調校」。調校過程中的日誌、系統資源和即時結果都會顯示在頁面上。

4.  **查看報告**：
    當所有任務完成後，「產生完整報告」按鈕會被啟用。點擊它，程式會自動為您生成詳細的 HTML 報告並在新分頁中開啟。

### 方法二：命令列介面 (CLI)

適合自動化或不依賴圖形介面的場景。

1.  **基本使用** (測試所有模型):
    ```bash
    python enhanced_ollama_autotuner.py
    ```

2.  **進階選項**:
    ```bash
    # 指定單一模型進行測試
    python enhanced_ollama_autotuner.py --model llama3:8b

    # 自定義時間和 TTFT 限制
    python enhanced_ollama_autotuner.py --time-limit 120 --ttft-limit 5

    # 啟用更詳細的日誌輸出
    python enhanced_ollama_autotuner.py --verbose
    ```

## 📋 依賴套件

- **核心依賴**: `ollama`, `pandas`
- **報告生成**: `Jinja2`
- **系統監控**: `psutil`, `GPUtil`
- **機器學習與優化**: `scikit-learn`, `scipy`, `numpy`
- **Web UI (可選)**: `Flask`, `Flask-SocketIO`
- **圖表生成 (可選)**: `matplotlib`, `seaborn`

## 📝 更新日誌

### v3.0.0 (最新)
- ✨ **新增功能**：建立了功能完善的互動式 Web UI，用於即時監控和控制。
- 🐛 **錯誤修復**：修復了命令列參數無法正常工作的問題。
- 🐛 **錯誤修復**：修正了因 `ollama` 函式庫 API 變更導致無法正確讀取模型列表的問題。
- 🐛 **錯誤修復**：修復了 Web UI 中的多個 JavaScript 錯誤和時序問題。
- 🎨 **樣式改進**：統一了 HTML 報告中的背景樣式，使其能正確響應主題切換。
- 🏗️ **架構重構**：重構了 Web UI 的啟動和資料處理流程，使其更穩定。

### v2.0.0
- ✨ 新增貝葉斯優化算法
- 🧠 實時記憶體監控
- 💾 智能緩存系統
- 📊 增強評估指標
- 🎨 互動式報告
- 🏗️ 模組化架構重構

---

# 🚀 Enhanced Ollama Auto-Tuner - Intelligent Optimization & Monitoring Tool for Local LLMs

The Enhanced Ollama Auto-Tuner is a Python tool designed to automatically find optimal performance and quality settings for your locally running Ollama Large Language Models (LLMs). This tool adopts a modular architecture, integrates multiple advanced technologies, and provides an interactive Web UI to monitor the tuning process in real-time.

## ✨ Core Features

- **Interactive Web UI**: Monitor system resources, tuning progress, logs, and results in real-time via a web interface, with remote control to start/stop tuning tasks.
- **Intelligent Optimization**: Uses Bayesian Optimization to find better parameter combinations with fewer tests.
- **Early Stopping**: Automatically ends the process when optimization stalls, saving time and resources.
- **Comprehensive Performance Optimization**: Includes GPU/CPU memory monitoring and smart caching to ensure a stable and efficient testing process.
- **Enhanced Evaluation System**: Provides a comprehensive quality score from multiple dimensions like relevance, logical consistency, and factual accuracy.
- **Detailed Visual Reports**: Generates interactive HTML reports with dynamic charts upon completion.

## 🚀 How to Use

This tool offers two modes of operation: the interactive Web UI (recommended) and the traditional Command-Line Interface (CLI).

### Method 1: Interactive Web UI (Recommended)

1.  **Launch the Web Service**:
    ```bash
    python web_ollama_autotuner.py
    ```

2.  **Open Your Browser**:
    Navigate to `http://127.0.0.1:5000`.

3.  **Start Tuning**:
    On the web page, you can select a specific model or all models from the dropdown and click "Start Tuning".

4.  **View Report**:
    Once all tasks are complete, the "Generate Full Report" button will be enabled. Clicking it will generate a detailed HTML report and open it in a new tab.

### Method 2: Command-Line Interface (CLI)

1.  **Basic Usage** (test all models):
    ```bash
    python enhanced_ollama_autotuner.py
    ```

2.  **Advanced Options**:
    ```bash
    # Specify a single model to test
    python enhanced_ollama_autotuner.py --model llama3:8b
    ```