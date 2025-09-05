# evaluation_dataset.py

# --- Task Type 1: Hallucination Detection ---
# Goal: Check if the model makes up information not present in the context.
HALLUCINATION_EVAL_SET = [
    {
        "id": "fictional_company",
        "task_type": "hallucination",
        "context": "根據內部文件，'輝光科技(GlowTech)' 是一家成立於2035年、專門製造量子電池的公司。它的總部位於火星的奧林帕斯城。",
        "question": "'輝光科技'的執行長是誰？",
        "ground_truth_keywords": ["不知道", "未提及", "沒有提到", "無法回答"], 
        "evaluation_logic": "must_contain_any"
    },
    {
        "id": "made_up_animal",
        "task_type": "hallucination",
        "context": "『晶翼獸』是一種生活在潘朵拉星球深海的虛構生物，牠們的身體由發光的矽晶體構成，以吸收海底熱泉的能量維生。",
        "question": "『晶翼獸』的天敵是什麼？",
        "ground_truth_keywords": ["不知道", "未提及", "沒有提到", "無法回答"],
        "evaluation_logic": "must_contain_any"
    },
    {
        "id": "specific_fact_check",
        "task_type": "hallucination",
        "context": "這份報告指出，A專案的預算為500萬美元。",
        "question": "A專案的預算是多少？",
        "ground_truth_keywords": ["500萬", "五百萬", "5,000,000"],
        "evaluation_logic": "must_contain_any"
    },
    {
        "id": "leading_question",
        "task_type": "hallucination",
        "context": "輝光科技是一家位於火星的量子電池公司，其創新的固態電解質技術引領了行業革命。",
        "question": "輝光科技的創辦人，像伊隆·馬斯克一樣，對太空探索有著怎樣的願景？",
        "ground_truth_keywords": ["不知道", "未提及", "沒有提到", "無法回答"],
        "evaluation_logic": "must_contain_any"
    },
    {
        "id": "subtle_misinformation",
        "task_type": "hallucination",
        "context": "報告顯示，Alpha專案在2023年完成，而Beta專案的啟動會議則在2024年1月舉行。",
        "question": "哪個專案是在2023年啟動的？",
        "ground_truth_keywords": ["不知道", "未提及", "沒有提到", "沒有專案"],
        "evaluation_logic": "must_contain_any"
    },
    {
        "id": "inference_negation",
        "task_type": "hallucination",
        "context": "除了財務部門，所有部門都參加了週五的全員會議。",
        "question": "財務部門的主管參加了週五的會議嗎？",
        "ground_truth_keywords": ["沒有", "未參加", "否"],
        "evaluation_logic": "must_contain_any"
    }
]

# --- Task Type 2: Summarization ---
# Goal: Evaluate the quality of a summary using ROUGE and BLEU scores.
SUMMARIZATION_EVAL_SET = [
    {
        "id": "tech_article_summary",
        "task_type": "summarization",
        "source_text": "蘋果公司今天發布了其最新的M4晶片，該晶片採用了第二代3奈米技術，擁有一個速度更快的10核心CPU和一個新的16核心神經網路引擎，能夠每秒執行高達38兆次操作。蘋果聲稱，與M2晶片相比，M4的CPU性能提升了50%，而GPU性能則提升了高達4倍，這使其在處理複雜的機器學習任務和高畫質圖形渲染方面表現出色。新的神經網路引擎也為AI應用帶來了顯著的加速，例如在Final Cut Pro中實現即時的場景移除遮罩功能。",
        "question": "請總結以上關於蘋果M4晶片的文章。",
        "reference_summary": "蘋果發布了採用3奈米技術的M4晶片，其CPU和GPU性能相較M2有大幅提升，並配備了更快的神經網路引擎以加速AI任務。"
    }
]

# Note on Coding Evaluation:
# The coding evaluation uses the standard HumanEval dataset, which is loaded
# directly by the `human-eval` library. Therefore, no specific dataset is defined here.
# The tuner will automatically select the first problem from the HumanEval dataset for evaluation.

# --- Performance Test Prompt ---
LONG_CONTEXT_PERFORMANCE_PROMPT = '''
一個細雨濛濛的下午，我坐在窗邊的咖啡館裡，看著行人撐著傘，匆匆走過被雨水洗淨的街道。空氣中瀰漫著濕潤的泥土氣息和淡淡的咖啡香，讓人感到一種難得的寧靜。這家咖啡館不大，裝潢簡單而溫馨，木質的桌椅在柔和的燈光下顯得格外有質感。我手裡捧著一杯熱拿鐵，暖意從指尖傳遍全身，耳機裡播放著輕柔的爵士樂，伴隨著窗外的雨聲，構成一曲獨特的交響。
我思緒飄遠，想起前幾天在書店偶然看到的一本書——《時間的縫隙》。這本書的作者是一位不知名的哲學家，他沒有華麗的詞藻，只是用最樸實的語言探討時間的本質。他寫道，我們總以為時間是一條筆直的長河，從過去流向未來，但事實上，時間更像是一張由無數細小「縫隙」編織而成的網。這些縫隙，就是我們生活中的每一個瞬間，每一個選擇，每一個錯過。
書中提到一個很有趣的觀點：我們無法回到過去，但我們可以透過「記憶」在時間的縫隙中穿梭。記憶不是單純的錄影帶，而是我們不斷重新詮釋和賦予意義的過程。每一次回憶，都是一次新的創造。作者認為，生命的意義並不在於累積多少成就或財富，而在於我們如何在這些時間的縫隙中，找到並珍視那些讓我們心靈富足的瞬間。
我閉上眼睛，試著在我的記憶網中尋找那些微小的光點。我看到了多年前一個夏日午後，與家人在公園野餐的場景。陽光穿過樹葉的縫隙灑在草地上，孩子們追逐嬉鬧的笑聲清脆悅耳。那時，我正忙於一個重要的專案，壓力很大，但那一刻，所有的煩惱都煙消雲散，我只感到一種純粹的快樂和滿足。我也回想起幾年前，為了處理一個 BIOS 方面的技術難題，我連續熬夜好幾天。當最終成功解決問題時，那種成就感至今仍讓我感到振奮。這些瞬間，無論大小，都構成了我生命中不可或缺的部分。
雨勢漸小，天空開始放晴，一縷陽光從雲層後探出頭來，灑在濕漉漉的街道上，泛著金色的光芒。我意識到，生活中的美好，常常在最意想不到的時刻出現。我們不必刻意去追尋，只需放慢腳步，用心去感受每一個當下。也許，真正的幸福，就是這樣一種能力——能夠在時間的縫隙中，發現那些微小而珍貴的光芒，並將它們永遠珍藏在心底。
我拿起手機，拍下了窗外放晴的景色。我想把這個瞬間記錄下來，放進我記憶的網中。或許在未來的某個雨天，當我再次感到迷茫時，這張照片能提醒我，即使在最晦暗的時刻，也總會有陽光灑落。
付完帳後，我走出咖啡館，深吸了一口氣。雨後的空氣格外清新，世界彷彿被重新洗滌了一遍。我走在回家的路上，心情輕鬆而愉悅。我知道，生命中還有很多未知的「縫隙」等著我去探索，而我已準備好，去迎接每一個新的瞬間。
根據以上全文，請總結這篇文章的核心觀點，並列出三個關鍵的數據點。
'''