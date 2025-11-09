# comparison.py  — 可交付安全版（保留原本輸出，只寫 results/performance_metrics.json）
from IPython.display import display

# ---- 先載入 Part A 的傳統法 ----
from traditional_methods import (
    RuleBasedSentimentClassifier,
    TopicClassifier,
    StatisticalSummarizer,
)

# ================== 通用匯入 ==================
import os, re, json, time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

# 只在有 ASCII 的 OPENAI_API_KEY 才初始化 OpenAI client，否則 client=None
def _get_openai_client_if_available():
    try:
        from openai import OpenAI  # 若未安裝 openai，這裡會丟例外；我們就走無 AI 模式
    except Exception:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        api_key.encode("ascii")  # Windows cmd/PowerShell 有時會混入全形/不可見字元
    except UnicodeEncodeError:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

CLIENT = _get_openai_client_if_available()
AI_ENABLED = CLIENT is not None
JUDGE_MODEL = "gpt-4o"

# ================== A. 相似度（AI 版本，維持你原本輸出） ==================
SYSTEM_MSG = (
    "你是嚴謹的語意相似度評分器。"
    "請只輸出 JSON：{\"score\": 浮點數}，分數範圍 0~1，保留兩位小數，不得輸出其他文字。"
    "評分尺標：1.00 幾乎同義或同一事件；0.70~0.90 主題/語意明顯相關；"
    "0.30~0.60 主題有一定關聯或同一大類；0.05~0.25 只有很弱的關聯（語氣/活動類似）；"
    "0.00 完全無關。請忽略字面重複，著重語意。"
)

def ai_similarity_01(a: str, b: str) -> float:
    """回傳 0~1；沒有 API 就直接回 0.00，程式不中斷。"""
    if not AI_ENABLED:
        return 0.00

    user_msg = (
        f"文本A：{a}\n文本B：{b}\n"
        "請只回傳 JSON：{\"score\": 介於 0~1 的浮點數（保留兩位小數）}。"
    )
    try:
        resp = CLIENT.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
    except Exception:
        try:
            resp = CLIENT.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content.strip()
        except Exception:
            return 0.00

    # 解析 JSON → score
    try:
        obj = json.loads(raw)
        score = float(obj["score"])
        return float(f"{max(0.0, min(1.0, score)):.2f}")
    except Exception:
        pass

    # 後備：拿最後一個數字當分數
    cleaned = re.sub(r"0\s*[–\-~]\s*1", "", raw)
    nums = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if nums:
        score = float(nums[-1])
        return float(f"{max(0.0, min(1.0, score)):.2f}")

    return 0.00

# --- 你原本的 texts（保留） ---
texts = [
    "人工智慧正在改變世界，機器學習是其核心技術",
    "深度學習推動了人工智慧的發展，特別是在圖像識別領域",
    "今天天氣很好，適合出去運動",
    "機器學習和深度學習都是人工智慧的重要分支",
    "運動有益健康，每天都應該保持運動習慣"
]

def print_ai_similarity_matrix_and_time(texts) -> Dict[str, Any]:
    """照你原本的作法輸出 df_sim；同時量測 AI 相似度總時間。"""
    n = len(texts)
    mat = np.zeros((n, n), dtype=float)

    t0 = time.perf_counter()
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.00
            elif j < i:
                mat[i, j] = mat[j, i]
            else:
                mat[i, j] = ai_similarity_01(texts[i], texts[j])
    t1 = time.perf_counter()

    labels = [f"Doc{i+1}" for i in range(n)]
    df_sim = pd.DataFrame(mat, index=labels, columns=labels)
    print("=== AI 相似度矩陣（0~1）===")
    display(df_sim)

    return {
        "matrix_size": n,
        "ai_time_sec": round(t1 - t0, 4) if AI_ENABLED else None,
    }

# ================== B. 分類（Rule-based vs AI） ==================
def ai_classify(text: str) -> Dict[str, Any]:
    """沒有 API 時回預設值（不中斷）。"""
    if not AI_ENABLED:
        return {"sentiment": "中性", "topic": "其他", "confidence": 0.0}
    system_msg = (
        "你是專業中文文本分類器，需只輸出 JSON，無其他解釋。"
        "欄位: sentiment(正面/負面/中性), topic(科技/運動/美食/旅遊/娛樂/其他), confidence(0~1)。"
        "請根據語意，不用逐字比對。"
    )
    user_msg = f"請為下列文本分類並只輸出 JSON：\n文本：{text}"
    try:
        resp = CLIENT.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.S)
        js = json.loads(m.group(0) if m else raw)
        sentiment = js.get("sentiment", "中性")
        topic = js.get("topic", "其他")
        conf = float(js.get("confidence", 0.0))
        return {"sentiment": sentiment, "topic": topic, "confidence": max(0.0, min(1.0, conf))}
    except Exception:
        return {"sentiment": "中性", "topic": "其他", "confidence": 0.0}

def print_classification_compare_and_time(test_texts) -> Dict[str, Any]:
    rb_s = RuleBasedSentimentClassifier()
    rb_t = TopicClassifier()

    # 傳統法
    t0 = time.perf_counter()
    rb_rows = []
    for txt in test_texts:
        rb_rows.append({"sentiment": rb_s.classify(txt), "topic": rb_t.classify(txt)})
    t1 = time.perf_counter()

    # AI 法
    ai_rows = []
    t2 = time.perf_counter()
    for txt in test_texts:
        ai_rows.append(ai_classify(txt))
    t3 = time.perf_counter()

    # 建表（保留你的輸出）
    rows = []
    for txt, r1, r2 in zip(test_texts, rb_rows, ai_rows):
        rows.append({
            "Text": txt,
            "Sentiment (Rule-based)": r1["sentiment"],
            "Topic (Rule-based)": r1["topic"],
            "Sentiment (AI)": r2["sentiment"],
            "Topic (AI)": r2["topic"],
            "Confidence": round(r2["confidence"], 3)
        })
    df_compare = pd.DataFrame(rows, columns=[
        "Text", "Sentiment (Rule-based)", "Topic (Rule-based)",
        "Sentiment (AI)", "Topic (AI)", "Confidence"
    ])
    print("=== Rule-based vs GPT-4o Classification ===")
    display(df_compare)

    # 一致率（僅參考）
    agree = sum(
        (a["sentiment"] == b["sentiment"]) and (a["topic"] == b["topic"])
        for a, b in zip(rb_rows, ai_rows)
    )
    agree_rate = round(agree / len(test_texts), 3) if len(test_texts) else None

    return {
        "num_texts": len(test_texts),
        "traditional_time_sec": round(t1 - t0, 4),
        "ai_time_sec": round(t3 - t2, 4) if AI_ENABLED else None,
        "agreement_rate_between_rule_and_ai": agree_rate if AI_ENABLED else None,
    }

# ================== C. 摘要（傳統 vs AI） ==================
def ai_summarize(text: str, ratio: float = 0.2) -> str:
    if not AI_ENABLED:
        return "[AI 未啟用] 無摘要"
    system_msg = (
        "你是一位專業的中文摘要生成助手，請以清楚、流暢的語句摘要以下文章。"
        "請保留主要資訊與邏輯脈絡，刪除冗餘內容。"
    )
    user_msg = f"請將以下文章縮短成原文約 {int(len(text)*ratio)} 字的摘要：\n\n{text}"
    try:
        resp = CLIENT.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI 摘要錯誤]: {e}"

def traditional_summary(text: str, top_n: int = 3) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sentences = re.split(r"[。！？\n]", text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return ""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf, tfidf)
    scores = np.sum(sim_matrix, axis=1)
    top_idx = np.argsort(scores)[-top_n:]
    summary = "。".join([sentences[i] for i in sorted(top_idx)]) + "。"
    return summary.strip()

def print_summary_compare_and_time(article: str) -> Dict[str, Any]:
    # 傳統
    t0 = time.perf_counter()
    trad_summary = traditional_summary(article, top_n=3)
    t1 = time.perf_counter()

    # AI
    t2 = time.perf_counter()
    ai_summary_text = ai_summarize(article, ratio=0.2)
    t3 = time.perf_counter()

    data = [
        {"方法": "傳統方法 (TF-IDF)", "原文長度": len(article),
         "摘要長度": len(trad_summary), "摘要內容": trad_summary},
        {"方法": "AI 方法 (GPT-4o)", "原文長度": len(article),
         "摘要長度": len(ai_summary_text), "摘要內容": ai_summary_text},
    ]
    df_summary = pd.DataFrame(data)
    print("=== 摘要比較（傳統 vs AI）===")
    display(df_summary)

    return {
        "orig_len": len(article),
        "traditional": {
            "summary_len": len(trad_summary),
            "time_sec": round(t1 - t0, 4),
            "compression_ratio": round(len(trad_summary) / max(1, len(article)), 3),
        },
        "ai": {
            "summary_len": len(ai_summary_text) if AI_ENABLED else None,
            "time_sec": round(t3 - t2, 4) if AI_ENABLED else None,
            "compression_ratio": (
                round(len(ai_summary_text) / max(1, len(article)), 3) if AI_ENABLED else None
            ),
        },
    }

# ================== 測試資料（和你原本一致） ==================
test_texts = [
    "這家餐廳的牛肉麵真的太好吃了，湯頭濃郁，麵條Q彈，下次一定再來！",
    "最新的AI技術突破讓人驚豔，深度學習模型的表現越來越好",
    "這部電影劇情空洞，演技糟糕，完全是浪費時間",
    "每天慢跑5公里，配合適當的重訓，體能進步很多",
]

article = (
    "人工智慧（AI）的發展正深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。\n"
    "在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。\n"
    "教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。\n"
    "然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。\n"
    "面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。\n"
)

# ================== 主程式：保留輸出 + 只寫 JSON ==================
def main():
    # 1) AI 相似度矩陣（保留你的輸出）
    sim_metrics = print_ai_similarity_matrix_and_time(texts)

    # 2) 分類比較（保留你的輸出）
    cls_metrics = print_classification_compare_and_time(test_texts)

    # 3) 摘要比較（保留你的輸出）
    sum_metrics = print_summary_compare_and_time(article)

    # 4) 只寫規定檔案：results/performance_metrics.json
    os.makedirs("results", exist_ok=True)
    perf = {
        "ai_enabled": AI_ENABLED,
        "similarity": sim_metrics,
        "classification": cls_metrics,
        "summarization": sum_metrics,
    }
    out_path = os.path.join("results", "performance_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(perf, f, ensure_ascii=False, indent=2)

    print("\n[saved] results/performance_metrics.json")

if __name__ == "__main__":
    main()
