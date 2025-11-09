# ==========================================
# Part A：傳統方法（TF-IDF / 規則式分類 / 統計式摘要）
# 需求：保留所有輸出（print / display），僅存：
#   1) results/tfidf_similarity_matrix.png
#   2) results/classification_results.csv
# ==========================================

import os
os.makedirs("results", exist_ok=True)

# -------------------------------
# A-1：TF-IDF 文本相似度計算
# -------------------------------
from IPython.display import display
import jieba
import numpy as np
import pandas as pd
from collections import Counter
import math

# 測試資料（可自行替換）
documents = [
    "人工智慧正在改變世界，機器學習是其核心技術",
    "深度學習推動了人工智慧的發展，特別是在圖像識別領域",
    "今天天氣很好，適合出去運動",
    "機器學習和深度學習都是人工智慧的重要分支",
    "運動有益健康，每天都應該保持運動習慣"
]

# 中文斷詞
tokenized_documents = [list(jieba.cut(doc)) for doc in documents]
print("=== 斷詞結果 ===")
for i, doc in enumerate(tokenized_documents, 1):
    print(f"Document {i}: {doc}")

# -------- 手動版 TF-IDF --------
def calculate_tf(word_dict, total_words):
    return {word: cnt / total_words for word, cnt in word_dict.items()}

def calculate_idf(tokenized_docs, word):
    df = sum(1 for doc in tokenized_docs if word in set(doc))
    return math.log(len(tokenized_docs) / (df + 1))  # +1 防止分母為 0

def calculate_tfidf(tokenized_docs):
    vocab = sorted(set(w for d in tokenized_docs for w in d))
    rows = []
    for doc in tokenized_docs:
        cnt = Counter(doc)
        tf = calculate_tf(cnt, len(doc))
        tfidf_row = {}
        for w in vocab:
            tfidf_row[w] = tf.get(w, 0.0) * calculate_idf(tokenized_docs, w)
        rows.append(tfidf_row)
    df = pd.DataFrame(rows)
    df.index = [f"Doc{i+1}" for i in range(len(tokenized_docs))]
    return df

tfidf_manual_df = calculate_tfidf(tokenized_documents)
pd.set_option("display.precision", 6)
print("\n=== 手動版 TF-IDF 權重表（前 5 行） ===")
display(tfidf_manual_df.head())

# -------- sklearn 版 TF-IDF + Cosine --------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

processed_docs = [" ".join(doc) for doc in tokenized_documents]
vectorizer = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"(?u)\b\S+\b",  # 我們已用空白分詞
    use_idf=True,
    smooth_idf=False,
    norm="l2",
)
X = vectorizer.fit_transform(processed_docs)  # (n_docs, n_terms)

feature_names = vectorizer.get_feature_names_out()
doc_labels = [f"Doc{i+1}" for i in range(len(processed_docs))]

tfidf_sklearn_df = pd.DataFrame(X.toarray(), index=doc_labels, columns=feature_names)
print("\n=== sklearn：TF-IDF 權重表（前 5 行） ===")
display(tfidf_sklearn_df.head())

sim_mat = cosine_similarity(X)
sim_df = pd.DataFrame(sim_mat, index=doc_labels, columns=doc_labels)
print("\n=== sklearn：文件×文件 Cosine 相似度矩陣 ===")
display(sim_df.round(3))

# -------- 視覺化（僅存規定的 PNG）--------
import logging
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8, 6))
sns.heatmap(
    sim_df.values,
    annot=True, fmt=".3f",
    cmap="YlGnBu",
    xticklabels=sim_df.columns,
    yticklabels=sim_df.index,
    square=True, linewidths=0.5,
    cbar_kws={"label": "Cosine similarity"},
)
plt.title("TF-IDF + Cosine Similarity Matrix", fontsize=14, pad=12)
plt.xlabel("Document ID")
plt.ylabel("Document ID")
plt.tight_layout()
plt.savefig("results/tfidf_similarity_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("[saved] results/tfidf_similarity_matrix.png")

# -------------------------------
# A-2：規則式文本分類（情感＋主題）
# -------------------------------
print("\n=== 規則式分類：測試資料 ===")
test_texts = [
    "這家餐廳的牛肉麵真的太好吃了，湯頭濃郁，麵條Q彈，下次一定再來！",
    "最新的AI技術突破讓人驚豔，深度學習模型的表現越來越好",
    "這部電影劇情空洞，演技糟糕，完全是浪費時間",
    "每天慢跑5公里，配合適當的重訓，體能進步很多"
]

class RuleBasedSentimentClassifier:
    def __init__(self):
        self.positive_words = ['好','棒','優秀','喜歡','推薦','滿意','開心','值得','精彩','完美','好吃','濃郁','Q彈']
        self.negative_words = ['差','糟','失望','討厭','不推薦','浪費','無聊','爛','糟糕','差勁','空洞','遜']
        self.negation_words = ['不','沒','無','非','別']

    def classify(self, text):
        pos, neg = 0, 0
        tokens = list(text)
        for i, w in enumerate(tokens):
            if w in self.positive_words:
                if i > 0 and tokens[i-1] in self.negation_words:
                    neg += 1
                else:
                    pos += 1
            elif w in self.negative_words:
                if i > 0 and tokens[i-1] in self.negation_words:
                    pos += 1
                else:
                    neg += 1
        if pos > neg: return "正面"
        if neg > pos: return "負面"
        return "中性"

class TopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            '科技': ['AI','人工智慧','電腦','軟體','程式','演算法','技術','模型','深度學習','機器學習'],
            '運動': ['運動','健身','跑步','游泳','球類','比賽','慢跑','體能'],
            '美食': ['吃','食物','餐廳','美味','料理','烹飪','牛肉麵','湯頭','Q彈'],
            '旅遊': ['旅行','景點','飯店','機票','觀光','度假'],
            '娛樂': ['電影','劇情','演技','音樂','遊戲']
        }

    def classify(self, text:str)->str:
        scores = {k:0 for k in self.topic_keywords}
        for topic, kws in self.topic_keywords.items():
            for kw in kws:
                if kw in text:
                    scores[topic] += 1
        best = max(scores, key=scores.get)
        mx = scores[best]
        if mx == 0 or list(scores.values()).count(mx) > 1:
            return "其他"
        return best

sentiment_classifier = RuleBasedSentimentClassifier()
topic_classifier = TopicClassifier()

print("=== 規則式情感分類結果（測試文本） ===")
for t in test_texts:
    print(f'【{sentiment_classifier.classify(t)}】{t}')

print("\n=== 規則式主題分類結果（測試文本） ===")
for t in test_texts:
    print(f'【{topic_classifier.classify(t)}】{t}')

# 也對上面 documents 做一次分類，並存成規定 CSV
rows = []
for txt in documents:
    rows.append({
        "text": txt,
        "sentiment_rule": sentiment_classifier.classify(txt),
        "topic_rule": topic_classifier.classify(txt),
    })
cls_df = pd.DataFrame(rows, columns=["text","sentiment_rule","topic_rule"])
print("\n=== 規則式分類（documents） ===")
display(cls_df)
cls_df.to_csv("results/classification_results.csv", index=False, encoding="utf-8-sig")
print("[saved] results/classification_results.csv")

# -------------------------------
# A-3：統計式自動摘要（只輸出，不存檔）
# -------------------------------
import re
from collections import Counter

class StatisticalSummarizer:
    def __init__(self):
        self.stop_words = set([
            "的","了","和","是","我","也","就","都","而","及","與","著","或","一個","沒有","我們",
            "你","他","她","它","在","到","於","被","對","由","從","為","因此"
        ])
        self.stop_words |= set(list("，。、；：！!？?（）()《》<>「」『』—-…．．,【】[] "))
        self.proper_hint = {"AI","人工智慧","深度學習","模型","醫療","隱私","倫理","法規","數據","演算法"}

    def _split(self, text:str):
        sents = re.split(r"[。！？!?]\s*|\n+", text)
        return [s.strip() for s in sents if s and s.strip()]

    def _tok(self, text:str):
        text = re.sub(r"[^\w\u4e00-\u9fff ]", " ", text)
        return [t for t in jieba.lcut(text) if t.strip()]

    def _freq(self, sents):
        all_tok = []
        for s in sents:
            all_tok += [w for w in self._tok(s) if w not in self.stop_words]
        if not all_tok: return {}
        cnt = Counter(all_tok)
        return {w: 1.0 + math.log(c) for w, c in cnt.items()}

    def _score(self, s, wf, idx, n):
        toks = [w for w in self._tok(s) if w not in self.stop_words]
        if not toks: return 0.0
        kw = sum(wf.get(w, 0.0) for w in toks)
        pos_w = 1.15 if idx <= 1 else (1.08 if idx >= n-2 else 1.0)
        L = len(s)
        length_pen = math.exp(- ((L - 28) ** 2) / (2 * (12 ** 2)))
        length_pen = max(0.6, length_pen)
        has_digit = 1 if re.search(r"\d", s) else 0
        has_proper = 1 if any(p in s for p in self.proper_hint) else 0
        bonus = 0.10*has_digit + 0.15*has_proper
        return (kw + bonus) * pos_w * length_pen

    def summarize_by_chars(self, text, ratio=0.3, max_chars=None,
                           keep_full_sentences=True, overflow_ratio=0.10):
        sents = self._split(text)
        if not sents: return ""
        wf = self._freq(sents)
        n = len(sents)
        scored = [(i, self._score(s, wf, i, n), s, len(s)) for i, s in enumerate(sents)]
        target = int(len(text) * ratio)
        if max_chars is not None:
            target = min(target, max_chars)
        picked, total = [], 0
        for i, sc, s, L in sorted(scored, key=lambda x: x[1], reverse=True):
            picked.append(i)
            total += L
            if total >= target:
                break
        picked.sort()
        out = "。".join(sents[i] for i in picked)
        if out and out[-1] != "。": out += "。"
        if keep_full_sentences:
            allow = int(target * (1 + overflow_ratio))
            while picked and len(out) > allow:
                picked.pop()
                out = "。".join(sents[i] for i in picked)
                if out and out[-1] != "。": out += "。"
        else:
            if len(out) > target:
                out = out[:target].rstrip("。；，、 ") + "…"
        return out

article = (
    "人工智慧（AI）的發展正深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。\n"
    "在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。\n"
    "教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。\n"
    "然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。\n"
    "面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。\n"
)

summarizer = StatisticalSummarizer()
summary = summarizer.summarize_by_chars(article, ratio=0.35, keep_full_sentences=True, overflow_ratio=0.10)

print("\n=== 統計式摘要 ===")
print("原文長度:", len(article))
print("摘要長度:", len(summary))
print("摘要內容：\n", summary)
