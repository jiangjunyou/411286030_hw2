# HW2 - Text Analysis (Traditional vs AI Methods)

本作業比較傳統 NLP 方法（TF-IDF、規則式分類、統計式摘要）與 AI 方法（OpenAI GPT-4o）在文本相似度、分類與摘要任務的表現。

---

## 專案結構
411286030_hw2/
├── traditional_methods.py # Part A: 傳統方法（TF-IDF、規則式分類、自動摘要）
├── modern_methods.py # Part B: AI 方法（GPT-4o 語意相似度、分類、摘要）
├── comparison.py # Part C: 傳統 vs AI 整合比較
├── report.md # 分析報告（質性與量化分析）
├── requirements.txt # 套件需求（pip install -r requirements.txt）
├── results/ # 輸出結果資料夾
│ ├── tfidf_similarity_matrix.png
│ ├── classification_results.csv
│ ├── summarization_comparison.txt
│ └── performance_metrics.json
└── README.md # 執行說明（本文件）

---

## 環境需求
請先安裝 Python 3.10 以上版本，並安裝所需套件：

pip install -r requirements.txt

主要套件
numpy
pandas
matplotlib
seaborn
scikit-learn
jieba
openai
ipython

API Key 設定
若要執行 modern_methods.py 或 comparison.py（使用 GPT-4o 模型），
請先在環境中設定 OpenAI API 金鑰：

Windows PowerShell :
setx OPENAI_API_KEY "你的_API_KEY"

macOS / Linux (bash) :
export OPENAI_API_KEY="你的_API_KEY"

若未設定 API key，系統將自動切換為「無 AI 模式」，部分結果（AI 相似度 / 摘要）將不生成。

執行方式
1. 傳統方法（TF-IDF、規則分類、自動摘要）
python traditional_methods.py

2. AI 方法（GPT-4o）
python modern_methods.py

3. 傳統 vs AI 比較
python comparison.py

輸出結果（Results）
執行完程式後，結果會自動輸出至 results/ 資料夾：

tfidf_similarity_matrix.png：TF-IDF 相似度熱圖

classification_results.csv：規則分類結果表

summarization_comparison.txt：傳統 vs AI 摘要比較

performance_metrics.json：執行效能與摘要壓縮率記錄

作者資訊
Student ID: 411286030
Course: AI Text Processing
Instructor: —
Date: November 2025