# modern_methods.py
# ==========================================
# Part B：現代方法（OpenAI GPT-4o）
# - 安全：只讀環境變數 OPENAI_API_KEY，不要求輸入、不寫死金鑰
# - 金鑰缺失/無效時，自動跳過 AI 呼叫並提示
# - 只輸出規定檔案：results/summarization_comparison.txt
#   其他結果僅在終端印出，不落地
# ==========================================

import os
import re
import json
from typing import Optional

import numpy as np
import pandas as pd

# ---- 用於比較摘要的傳統法（本檔內自給，避免跨檔依賴）----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 測試文本（老師可重跑、你朋友和你一致的那組）
TEXTS = [
    "人工智慧正在改變世界，機器學習是其核心技術",
    "深度學習推動了人工智慧的發展，特別是在圖像識別領域",
    "今天天氣很好，適合出去運動",
    "機器學習和深度學習都是人工智慧的重要分支",
    "運動有益健康，每天都應該保持運動習慣",
]

# 用來做摘要比較的文章
ARTICLE = (
    "人工智慧（AI）的發展正深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。\n"
    "在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。\n"
    "教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。\n"
    "然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。\n"
    "面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。\n"
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# A. OpenAI Client（安全版）
# =========================
def _clean_api_key(key: Optional[str]) -> Optional[str]:
    """只接受 ASCII 金鑰，避免非 ASCII 造成 httpx header 失敗。"""
    if not key:
        return None
    try:
        key.encode("ascii")
        return key.strip()
    except Exception:
        return None

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


def _build_client() -> Optional["OpenAI"]:
    """若環境變數沒有金鑰或格式不對，回傳 None 並提示。"""
    api_key = _clean_api_key(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        print("⚠️  未偵測到有效的 OPENAI_API_KEY（缺失或非 ASCII）。將以『無 AI 模式』繼續。")
        return None
    try:
        from openai import OpenAI  # 延遲匯入，沒有套件時也能印出友善提示
    except Exception as e:
        print(f"⚠️  無法匯入 openai 套件：{e}。將以『無 AI 模式』繼續。")
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"⚠️  建立 OpenAI Client 失敗：{e}。將以『無 AI 模式』繼續。")
        return None


# =========================
# B. AI 語意相似度（0~1）
# =========================
JUDGE_MODEL = "gpt-4o"

_SYSTEM_SIM = (
    "你是嚴謹的中文語意相似度評分器。僅輸出 JSON 格式："
    '{"score": 介於 0~1 的浮點數、保留兩位小數}。'
    "評分原則：1.00 幾乎同義；0.70~0.90 主題明顯相關；0.30~0.60 同類但不相近；"
    "0.05~0.25 極弱關聯；0.00 完全無關。著重『語意』而非字面。"
)

def ai_similarity_01(client, a: str, b: str) -> float:
    """呼叫 GPT-4o，回傳 0.00~1.00。client 為 None 時回 0.00。"""
    if client is None:
        return 0.00

    user_msg = (
        f"文本A：{a}\n文本B：{b}\n"
        "只回傳 JSON：{\"score\": 浮點數（0~1，保留兩位小數）}"
    )

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_SIM},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content.strip()

        # 先抓 JSON
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group())
                val = float(obj["score"])
                return float(f"{max(0.0, min(1.0, val)):.2f}")
            except Exception:
                pass

        # 後備：抓最後一個數字
        nums = re.findall(r"\d+(?:\.\d+)?", raw)
        if nums:
            val = float(nums[-1])
            return float(f"{max(0.0, min(1.0, val)):.2f}")
        return 0.00
    except Exception as e:
        print(f"⚠️  ai_similarity_01 呼叫失敗：{e}")
        return 0.00


def print_ai_similarity_matrix(client, texts: list[str]) -> None:
    """列印 0~1 的相似度矩陣（僅終端輸出，不存檔）。"""
    n = len(texts)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.00
            elif j < i:
                mat[i, j] = mat[j, i]
            else:
                mat[i, j] = ai_similarity_01(client, texts[i], texts[j])

    df = pd.DataFrame(mat, index=[f"Doc{i+1}" for i in range(n)],
                      columns=[f"Doc{i+1}" for i in range(n)])
    print("\n=== GPT-4o Semantic Similarity Matrix (0~1) ===")
    print(df.to_string(index=True))


# =========================
# C. AI 文本分類（終端展示）
# =========================
def ai_classify(client, text: str) -> dict:
    """
    回傳：
      {"sentiment": "正面/負面/中性", "topic": "科技/運動/美食/旅遊/娛樂/其他", "confidence": 0~1}
    client 為 None 時回保守結果。
    """
    if client is None:
        return {"sentiment": "中性", "topic": "其他", "confidence": 0.0}

    system_msg = (
        "你是中文文本分類器，只輸出 JSON："
        '{"sentiment":"正面/負面/中性","topic":"科技/運動/美食/旅遊/娛樂/其他","confidence":0~1}'
    )
    user_msg = f"請分類以下文本：\n{text}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(m.group() if m else raw)
        s = data.get("sentiment", "中性")
        t = data.get("topic", "其他")
        c = float(data.get("confidence", 0.0))
        return {"sentiment": s, "topic": t, "confidence": max(0.0, min(1.0, c))}
    except Exception as e:
        print(f"⚠️  ai_classify 呼叫失敗：{e}")
        return {"sentiment": "中性", "topic": "其他", "confidence": 0.0}


def print_ai_classification_demo(client) -> None:
    """示範分類（僅終端輸出，不存檔）。"""
    samples = [
        "這家餐廳的牛肉麵真的太好吃了，湯頭濃郁，麵條Q彈，下次一定再來！",
        "最新的AI技術突破讓人驚豔，深度學習模型的表現越來越好",
        "這部電影劇情空洞，演技糟糕，完全是浪費時間",
        "每天慢跑5公里，配合適當的重訓，體能進步很多",
    ]
    rows = []
    for s in samples:
        r = ai_classify(client, s)
        rows.append({
            "Text": s,
            "Sentiment (AI)": r["sentiment"],
            "Topic (AI)": r["topic"],
            "Confidence": f'{r["confidence"]:.2f}',
        })
    df = pd.DataFrame(rows)
    print("\n=== AI Text Classification (GPT-4o) ===")
    print(df.to_string(index=False))


# =========================
# D. 摘要（會存檔：summarization_comparison.txt）
# =========================
def traditional_summary(text: str, top_n: int = 3) -> str:
    """TF-IDF 抽取關鍵句（簡潔穩定）。"""
    sents = re.split(r"[。！？\n]", text)
    sents = [s for s in sents if s.strip()]
    if not sents:
        return ""
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(sents)
    sim = cosine_similarity(tfidf, tfidf)
    scores = np.sum(sim, axis=1)
    idx = np.argsort(scores)[-top_n:]
    return "。".join(sents[i] for i in sorted(idx)) + "。"

def ai_summarize(client, text: str, ratio: float = 0.35) -> str:
    """GPT-4o 生成摘要；client 為 None 時回空字串。"""
    if client is None:
        return ""
    target = max(50, int(len(text) * ratio))
    system_msg = (
        "你是中文摘要助手。保留主要觀點與關鍵資訊，語句自然，長度接近指定字數。"
        "只輸出摘要內容，不要多餘說明。"
    )
    user_msg = f"請為以下文章撰寫摘要（約 {target} 字）：\n\n{text}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return re.sub(r"\s+", " ", resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"⚠️  ai_summarize 呼叫失敗：{e}")
        return ""

def write_summarization_comparison(client, text: str, out_path: str) -> None:
    """
    產出作業規定檔案：results/summarization_comparison.txt
    - 含「原文長度 / 傳統摘要長度 / AI 摘要長度」與內容
    - 若無金鑰，AI 部分以 [AI disabled] 註記，仍會寫檔
    """
    trad = traditional_summary(text, top_n=3)
    ai = ai_summarize(client, text, ratio=0.35)

    lines = []
    lines.append("=== Summarization Comparison ===")
    lines.append(f"原文長度: {len(text)}")
    lines.append("")
    lines.append("[傳統方法 (TF-IDF)]")
    lines.append(f"摘要長度: {len(trad)}")
    lines.append(trad if trad else "(無法產生摘要)")
    lines.append("")
    lines.append("[AI 方法 (GPT-4o)]")
    if client is None:
        lines.append("（AI disabled：未提供有效 OPENAI_API_KEY）")
        lines.append("摘要長度: 0")
        lines.append("(跳過 AI 摘要)")
    else:
        lines.append(f"摘要長度: {len(ai)}")
        lines.append(ai if ai else "(AI 摘要失敗)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[saved] {out_path}")


# =========================
# Main
# =========================
def main():
    client = _build_client()

    # 1) AI 相似度矩陣（只印、不中檔）
    print_ai_similarity_matrix(client, TEXTS)

    # 2) AI 文本分類展示（只印、不中檔）
    print_ai_classification_demo(client)

    # 3) 摘要比較（會存：results/summarization_comparison.txt）
    out_txt = os.path.join(RESULTS_DIR, "summarization_comparison.txt")
    write_summarization_comparison(client, ARTICLE, out_txt)

if __name__ == "__main__":
    main()
