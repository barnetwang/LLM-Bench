# 🚀 Ollama Auto-Tuner - Automated Optimization Tool for Local LLMs

Ollama Auto-Tuner is a powerful tool developed in Python, designed to automatically find the optimal performance and quality settings for your locally running Ollama Large Language Models (LLMs).

[中文版說明](#-ollama-auto-tuner---本地-llm-自動化最佳化工具)

Have you ever been puzzled by questions like:
*   How many layers of my model should I offload to the GPU (`num_gpu`) to run efficiently without sacrificing too much speed?
*   What should I set `temperature` and `top_p` to, for more accurate, less hallucinatory responses?
*   Which of my models runs fastest on my hardware while also providing the best quality?

This tool automatically answers these questions for you and generates a clear, professional HTML report.

## ✨ Core Features

*   **Fully Automated Workflow**: Automatically detects all your locally installed Ollama models and tests them one by one.
*   **Intelligent Constraint Selection**: Automatically chooses a reasonable set of performance constraints based on the model's size (e.g., 7B, 20B).
*   **Quality-First Tuning**: Finds the best `temperature` and `top_p` combination for a model by running it against a carefully designed "hallucination evaluation" benchmark to maximize response accuracy.
*   **Efficient Performance Exploration**: Uses a binary search algorithm to efficiently determine the maximum GPU layers (`num_gpu`) and context window (`num_ctx`) that can be used within your time limits.
*   **Robust Execution Engine**:
    *   Uses multiprocessing to ensure that a single stalled or timed-out test won't crash the entire process.
    *   Supports safe interruption with `Ctrl+C`, allowing you to stop the tests at any time and clean up background processes.
    *   Automatically skips incompatible models (e.g., embedding models).
*   **Professional Report Generation**:
    *   Automatically generates a beautiful HTML report (`ollama_tuner_report.html`) after all tests are complete.
    *   The report includes a summary comparison table, detailed performance metrics (TTFT, TPS), optimized settings, and quality scores for each model.
    *   Automatically opens the report in your default web browser upon completion.

## 📊 Sample Report

*(Hint: You can take a screenshot of your report and replace this link)*

## 🛠️ Installation Guide

1.  **Ensure Ollama is Installed and Running**
    Please follow the instructions on the [Ollama Official Website](https://ollama.com/) to install Ollama and make sure its service is running in the background.

2.  **Pull the Models You Want to Test**
    In your terminal, use the `ollama pull` command to download the models you are interested in. For example:
    ```bash
    ollama pull llama3:8b
    ollama pull qwen:14b
    ```

3.  **Clone This Project**
    ```bash
    git clone https://github.com/your-username/ollama-auto-tuner.git
    cd ollama-auto-tuner
    ```

4.  **Install Python Dependencies**
    It is recommended to install in a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## 🚀 How to Use

Running the Auto-Tuner is simple. Just execute the main script:

```bash
python ollama_autotuner.py
```

The script will automatically perform the following steps:
1.  Connect to your local Ollama service and detect all installed models.
2.  Select appropriate testing criteria for each model.
3.  Perform quality and performance tuning for each model sequentially.
4.  Print a detailed log of the testing process and final settings in the terminal.
5.  After all tests are complete, generate `ollama_tuner_report.html` and open it in your browser.

## 🧬 How to Extend and Customize

You can easily customize this tool to meet your specific needs:

*   **Adjust Tuning Parameters**: In the `tune_quality` and `tune_context_window` methods of `ollama_autotuner.py`, you can modify the lists of options for `temperature`, `top_p`, and `num_ctx`.

*   **Enhance Hallucination Tests**: Edit the `evaluation_dataset.py` file to add more complex and challenging test cases to better evaluate model quality.

*   **Modify Constraints**: In the `select_constraints_by_size` function in `ollama_autotuner.py`, you can change the performance constraints (like `time_limit_s`, `ttft_limit_s`) for different model sizes.

*   **Beautify the Report**: Directly edit the `report_template.html` file. You can freely change the style, layout, or add charts (e.g., using [Chart.js](https://www.chartjs.org/)).

## 🙏 Acknowledgements

This project was born from an idea and gradually refined through numerous iterations and debugging sessions with an AI assistant. Thanks to all open-source community contributors, especially the Ollama team.

---

# 🚀 Ollama Auto-Tuner - 本地 LLM 自動化最佳化工具

Ollama Auto-Tuner 是一個使用 Python 開發的強大工具，旨在自動化地為您本地運行的 Ollama 大型語言模型（LLM）尋找最佳的性能與品質設定。

您是否曾困擾於以下問題：
*   我的模型應該分配多少層到 GPU (`num_gpu`) 才能在不犧牲太多速度的情況下運行？
*   `temperature` 和 `top_p` 應該設為多少，才能讓模型回答得更準確，減少幻覺？
*   哪個模型在我的硬體上跑得最快，同時品質也最好？

這個工具將為您自動解答這些問題，並生成一份清晰、專業的 HTML 報告。

## ✨ 核心功能

*   **全自動化流程**：自動偵測您本地已安裝的所有 Ollama 模型，並逐一進行測試。
*   **智慧約束選擇**：根據模型的大小（例如 7B, 20B）自動選擇一套合理的性能測試約束條件。
*   **品質優先調校**：透過一套精心設計的「幻覺評估」基準測試，為模型找到最佳的 `temperature` 和 `top_p` 組合，以最大限度地提高回答的準確性。
*   **高效性能探索**：使用二分搜尋法高效地確定在滿足時間限制的前提下，可以分配到 GPU 的最大層數 (`num_gpu`) 和可用的最大上下文窗口 (`num_ctx`)。
*   **健壯的執行機制**：
    *   使用多進程（multiprocessing）確保任何單一測試的卡死或超時都不會影響整個流程。
    *   支援 `Ctrl+C` 安全中斷，可以隨時停止測試並清理背景進程。
    *   自動跳過不相容的模型（如 Embedding 模型）。
*   **專業報告生成**：
    *   在所有測試結束後，自動生成一份精美的 HTML 報告 (`ollama_tuner_report.html`)。
    *   報告包含總結對比表格、每個模型的詳細性能指標（TTFT, TPS）、最佳化設定和品質評分。
    *   完成後自動在您的預設瀏覽器中打開報告。

## 📊 報告範例

*(提示: 您可以將您的報告截圖並替換此處的連結)*

## 🛠️ 安裝指南

1.  **確保 Ollama 已安裝並運行**
    請根據 [Ollama 官方網站](https://ollama.com/) 的指引安裝 Ollama，並確保其服務正在背景運行。

2.  **拉取您想測試的模型**
    在終端機中，使用 `ollama pull` 命令下載您感興趣的模型。例如：
    ```bash
    ollama pull llama3:8b
    ollama pull qwen:14b
    ```

3.  **克隆本專案**
    ```bash
    git clone https://github.com/your-username/ollama-auto-tuner.git
    cd ollama-auto-tuner
    ```

4.  **安裝 Python 依賴項**
    建議在虛擬環境中進行安裝。
    ```bash
    python -m venv venv
    source venv/bin/activate  # 在 Windows 上使用 `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## 🚀 如何使用

運行 Auto-Tuner 非常簡單，只需要執行主腳本即可：

```bash
python ollama_autotuner.py
```

程式將會自動執行以下步驟：
1.  連接到您的本地 Ollama 服務並偵測所有已安裝的模型。
2.  為每個模型選擇合適的測試標準。
3.  逐一對每個模型進行品質和性能調校。
4.  在終端機中打印詳細的測試過程和最終設定。
5.  所有測試完成後，生成 `ollama_tuner_report.html` 並自動在瀏覽器中打開。

## 🧬 如何擴展與客製化

您可以輕鬆地客製化此工具以滿足您的特定需求：

*   **調整測試參數**：在 `ollama_autotuner.py` 的 `tune_quality` 和 `tune_context_window` 方法中，您可以修改要測試的 `temperature`, `top_p` 和 `num_ctx` 的選項列表。

*   **增強幻覺測試**：編輯 `evaluation_dataset.py` 檔案，加入更多、更刁鑽的測試題目，以更好地評估模型的品質。

*   **修改約束條件**：在 `ollama_autotuner.py` 的 `select_constraints_by_size` 函式中，您可以修改不同大小模型的性能約束條件（如 `time_limit_s`, `ttft_limit_s`）。

*   **美化報告**：直接編輯 `report_template.html` 檔案，您可以自由地修改報告的樣式、佈局或新增圖表（例如使用 [Chart.js](https://www.chartjs.org/)）。

## ⚖️ 授權與感謝 / License & Acknowledgements

此專案為開源項目，使用了多個第三方函式庫。
This project is open-source and uses various third-party libraries.
請仔細閱讀其授權條款，在商業用途前務必審視清楚。
Please review their licenses carefully before using this code for commercial purposes.