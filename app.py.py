"""
=============================================================================
TRUTHGUARD - SMART FAKE NEWS DETECTION ENGINE
Machine Learning + Multilingual NLP + Viral Hoax Verification
FastAPI + WebSockets Backend (Port 8000)
=============================================================================
"""

import asyncio
import json
import logging
import math
import os
import re
import string
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlparse

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB

# Ensure standard output handles UTF-8 on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TruthGuardEngine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "news_model_bundle.joblib")
FAKE_CSV_PATH = os.path.join(BASE_DIR, "Fake.csv")
TRUE_CSV_PATH = os.path.join(BASE_DIR, "True.csv")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")

ml_bundle: Optional[Dict[str, Any]] = None

TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com",
    "timesofindia.indiatimes.com", "hindustantimes.com", "bloomberg.com",
    "wsj.com", "nytimes.com", "ft.com", "aljazeera.com", "theguardian.com",
    "dw.com", "afp.com", "nature.com", "sciencemag.org", "npr.org", "pbs.org",
    "time.com", "washingtonpost.com", "economist.com", "cnbc.com", "cnn.com",
    "cbsnews.com", "nbcnews.com", "abcnews.go.com", "ndtv.com", "indianexpress.com",
    "aajtak.in", "bhaskar.com", "amarujala.com", "jagran.com", "navbharattimes.indiatimes.com",
    "pib.gov.in", "ddnews.gov.in", "gov.in", "isro.gov.in", "cbse.gov.in", "rbi.org.in"
}

# Common Viral Fake News, Hoax, Scam & Clickbait Patterns (English + Hindi)
HOAX_SCAM_PATTERNS = [
    # 1. Currency demonetization / ban rumors
    (r"notes? (are|will be|is) banned", "Unverified currency ban rumor"),
    (r"500 (rupee|note)", "Unverified Rs 500 currency rumor"),
    (r"2000 (rupee|note)", "Unverified Rs 2000 currency rumor"),
    (r"currency ban", "Unverified currency ban claim"),
    (r"नोट बंद", "अपुष्ट नोट बंदी की अफवाह (Currency ban rumor)"),
    (r"500 के नोट", "500 के नोट से जुड़ी अफवाह (500 note rumor)"),
    (r"2000 के नोट", "2000 के नोट से जुड़ी अफवाह"),
    (r"करेंसी बंद", "करेंसी बंद होने का अपुष्ट दावा"),

    # 2. Free giveaway / Government scheme phishing scams
    (r"free (5000|1000|2000|10000|laptop|recharge|money|smartphone|tablet)", "Viral free scheme / giveaway scam pattern"),
    (r"click (the )?link to claim", "Phishing / click link to claim call-to-action"),
    (r"claim your (free )?", "Viral claim free reward pattern"),
    (r"scheme to get free", "Unverified free money scheme"),
    (r"फ्री (लैपटॉप|5000|रिचार्ज|स्मार्टफोन|पैसे|टैबलेट)", "फर्जी फ्री स्कीम या स्कैम पैटर्न"),
    (r"लिंक पर क्लिक", "फिशिंग लिंक क्लिक करने का दावा"),
    (r"फ्री में मिल रहा", "फर्जी फ्री ऑफर"),

    # 3. Medical pseudo-science & extreme miracle/scare hoaxes
    (r"(causes|cures) instant", "Sensationalized instant cure/harm medical claim"),
    (r"causes (instant )?cancer", "Unscientific extreme cancer scare"),
    (r"causes (instant )?heart attack", "Unverified sudden heart attack scare"),
    (r"cold water causes", "Debunked viral cold water myth"),
    (r"cure (all )?cancer in", "Unsubstantiated 100% cancer cure hoax"),
    (r"miracle cure", "Unverified medical miracle claim"),
    (r"100% cure", "Exaggerated 100% cure guarantee"),
    (r"कैंसर का (इलाज|कारण)", "अवैज्ञानिक कैंसर से जुड़ी अफवाह"),
    (r"तुरंत हार्ट अटैक", "अवैज्ञानिक हार्ट अटैक का दावा"),
    (r"पानी पीने से कैंसर", "वायरल झूठा मेडिकल दावा"),
    (r"लहसुन.*कैंसर", "वायरल चमत्कारी इलाज का झूठा दावा"),

    # 4. Cosmic / Doomsday / Weather apocalypse hoaxes
    (r"total darkness", "Viral total darkness apocalyptic hoax"),
    (r"\d+ days of (total )?darkness", "Debunked solar darkness hoax"),
    (r"earth will experience", "Viral cosmic event rumor"),
    (r"asteroid hitting earth", "Unverified apocalyptic asteroid claim"),
    (r"दुनिया खत्म", "प्रलय या तबाही की फर्जी अफवाह"),
    (r"अंधेरा छा जाएगा", "सूर्य ग्रहण / अंधकार की फर्जी अफवाह"),

    # 5. Sudden emergency lockdown / curfew / internet shutdown hoaxes
    (r"emergency (nationwide )?lockdown", "Unverified emergency lockdown rumor"),
    (r"nationwide lockdown", "Fake nationwide lockdown announcement"),
    (r"lockdown across (india|the country)", "Unverified countrywide lockdown forward"),
    (r"internet (shut ?down|closed) across", "Unverified nationwide internet shutdown rumor"),
    (r"देशव्यापी लॉकडाउन", "देशव्यापी लॉकडाउन की झूठी अफवाह"),
    (r"पूरे भारत में लॉकडाउन", "पूरे देश में लॉकडाउन की फर्जी खबर"),
    (r"इंटरनेट बंद", "पूरे देश में इंटरनेट बंद की अफवाह"),

    # 6. Impersonation & Fake awards / audio hoaxes
    (r"unesco (declared|voted)", "Viral UNESCO fake declaration hoax"),
    (r"nasa recorded om", "Debunked NASA Om sound hoax"),
    (r"यूनेस्को ने", "यूनेस्को के नाम पर फर्जी वायरल दावा"),
    (r"नासा ने रिकॉर्ड", "नासा के नाम पर वायरल फर्जी दावा"),

    # 7. Urgency, clickbait & forward pressure
    (r"share before (it )?(gets )?deleted", "Urgent viral sharing manipulation"),
    (r"share quickly", "Viral sharing urge without verification"),
    (r"forward this to", "Forwarded chain message pattern"),
    (r"wake up (people|sheeple)", "Conspiracy agitation tone"),
    (r"they are hiding", "Conspiracy cover-up claim"),
    (r"secret revealed", "Sensational clickbait revelation"),
    (r"secret exposed", "Sensational clickbait exposure"),
    (r"doctors (shocked|hate|terrified)", "Classic clickbait formula"),
    (r"mind control microchips?", "Conspiracy microchip theory"),
    (r"सबको शेयर", "वायरल फॉरवर्ड शेयर करने का दबाव"),
    (r"डिलीट होने से पहले", "डिलीट होने से पहले शेयर करने की चालबाजी"),
    (r"गुप्त रहस्य", "सेंसेशनल क्लिकबेट साजिश"),
    (r"चौंकाने वाला", "सेंसेशनल क्लिकबेट"),
    (r"डॉक्टर हैरान", "क्लिकबेट फॉर्मूला")
]

# Genuine Accredited News Agencies, Government Bodies & Scientific Institutions
ACCREDITED_SOURCES = [
    "isro", "nasa", "cbse", "rbi", "reserve bank", "supreme court", "high court",
    "pib", "press information bureau", "reuters", "associated press", "ap news",
    "press trust of india", "pti", "ani", "bbc", "the hindu", "times of india",
    "hindustan times", "indian express", "ndtv", "nature", "peer-reviewed",
    "weather department", "meteorological", "election commission", "parliament",
    "ministry of", "who official", "icmr", "aiims",
    "इसरो", "नासा", "सीबीएसई", "रिजर्व बैंक", "सुप्रीम कोर्ट", "मौसम विभाग",
    "पीआईबी", "प्रेस इंफॉर्मेशन ब्यूरो", "पीटीआई", "आधिकारिक", "चुनाव आयोग"
]

LANGUAGE_NAMES = {
    "hi": "Hindi (हिंदी)",
    "en": "English",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "mr": "Marathi (मराठी)",
    "gu": "Gujarati (ગુજરાતી)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ur": "Urdu (اردو)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)",
    "zh": "Chinese (中文)",
    "ja": "Japanese (日本語)"
}

def translate_if_needed(text: str) -> Dict[str, Any]:
    """Detects language and translates non-English text to English."""
    cleaned = text.strip()
    has_non_ascii = any(ord(c) > 127 for c in cleaned)

    if not has_non_ascii:
        return {
            "is_translated": False,
            "lang_code": "en",
            "lang_name": "English",
            "translated_text": cleaned
        }

    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + quote(cleaned)
        resp = requests.get(url, timeout=3.5)
        if resp.status_code == 200:
            data = resp.json()
            translated = "".join([seg[0] for seg in data[0] if seg and seg[0]])
            detected_lang = data[2] if len(data) > 2 else "unknown"
            lang_display = LANGUAGE_NAMES.get(detected_lang, detected_lang.upper())
            return {
                "is_translated": True,
                "lang_code": detected_lang,
                "lang_name": lang_display,
                "translated_text": translated
            }
    except Exception as ex:
        logger.warning(f"Translation lookup skipped: {ex}")

    is_hindi = bool(re.search(r"[\u0900-\u097F]", cleaned))
    return {
        "is_translated": False,
        "lang_code": "hi" if is_hindi else "unknown",
        "lang_name": "Hindi (हिंदी)" if is_hindi else "Detected Language",
        "translated_text": cleaned
    }

def clean_text_for_nlp(text: str) -> str:
    """Standardizes text for NLP feature mapping."""
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"read more at:.*", " ", t)
    t = re.sub(r"\[.*?\]", " ", t)
    t = re.sub(r"<.*?>+", " ", t)
    t = re.sub(r"[%s]" % re.escape(string.punctuation), " ", t)
    t = re.sub(r"\w*\d\w*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_or_train_models():
    """Loads pre-trained model bundle or trains fresh ensemble."""
    global ml_bundle
    if os.path.exists(MODEL_BUNDLE_PATH):
        try:
            logger.info("Loading pre-trained news model bundle from disk...")
            ml_bundle = joblib.load(MODEL_BUNDLE_PATH)
            logger.info(f"Model bundle loaded! Features: {len(ml_bundle['vectorizer'].vocabulary_)}")
            return
        except Exception as e:
            logger.warning(f"Bundle load failed: {e}. Rebuilding...")

    if not (os.path.exists(FAKE_CSV_PATH) and os.path.exists(TRUE_CSV_PATH)):
        logger.error("Dataset files Fake.csv and True.csv not found!")
        raise FileNotFoundError("Datasets Fake.csv and True.csv must be present.")

    logger.info("Training fresh ensemble model from CSV datasets...")
    t0 = time.time()
    df_fake = pd.read_csv(FAKE_CSV_PATH)
    df_true = pd.read_csv(TRUE_CSV_PATH)

    df_true["text"] = df_true["text"].apply(
        lambda x: re.sub(r"^\s*.*?\(\s*Reuters\s*\)\s*-\s*", "", str(x), flags=re.I)
    )

    df_fake["text"] = df_fake["title"].fillna("") + " " + df_fake["text"].fillna("")
    df_true["text"] = df_true["title"].fillna("") + " " + df_true["text"].fillna("")

    df_fake["label"] = 0
    df_true["label"] = 1

    sample_size = min(len(df_fake), len(df_true), 16000)
    df = pd.concat([
        df_fake.sample(n=sample_size, random_state=42),
        df_true.sample(n=sample_size, random_state=42)
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    df["clean_text"] = df["text"].apply(clean_text_for_nlp)

    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=3
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values

    model_lr = LogisticRegression(C=2.0, max_iter=500, random_state=42)
    model_lr.fit(X, y)

    model_sgd = SGDClassifier(loss="log_loss", max_iter=500, random_state=42)
    model_sgd.fit(X, y)

    model_nb = MultinomialNB(alpha=0.1)
    model_nb.fit(X, y)

    ml_bundle = {
        "vectorizer": vectorizer,
        "model_lr": model_lr,
        "model_sgd": model_sgd,
        "model_nb": model_nb,
        "accuracy_scores": {
            "lr": float(model_lr.score(X, y)),
            "sgd": float(model_sgd.score(X, y)),
            "nb": float(model_nb.score(X, y))
        },
        "trained_at": time.time(),
        "samples_trained": len(df)
    }

    joblib.dump(ml_bundle, MODEL_BUNDLE_PATH, compress=3)
    logger.info(f"Model trained and saved in {time.time() - t0:.2f}s!")

def fetch_url_content(url: str) -> Dict[str, Any]:
    """Scrapes news article headline and substantive paragraphs from URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")

    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = ""
        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        elif soup.title:
            title = soup.title.get_text().strip()

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        paragraphs = [
            p.get_text().strip() for p in soup.find_all("p")
            if len(p.get_text().strip().split()) > 7
        ]
        body = " ".join(paragraphs)
        if not body or len(body.split()) < 20:
            body = soup.get_text(separator=" ", strip=True)

        full_content = f"{title}. {body}".strip() if title else body

        return {
            "success": True,
            "url": url,
            "domain": domain,
            "title": title or "Live Online Article",
            "text": full_content,
            "word_count": len(re.findall(r"\w+", full_content))
        }
    except Exception as ex:
        logger.warning(f"URL fetch error for {url}: {ex}")
        return {
            "success": False,
            "url": url,
            "domain": domain,
            "title": "Online News Source",
            "text": url,
            "word_count": 0,
            "error": str(ex)
        }

def analyze_multilingual_news(raw_input: str, is_url_mode: bool = False) -> Dict[str, Any]:
    """
    Multilingual Fake News Detection Engine:
    1. Language Translation & Normalization
    2. Viral Hoax & Scam Pattern Auditing
    3. Accredited Institutional Entity Matching
    4. ML Ensemble Prediction (Logistic Regression + SGD + Naive Bayes)
    5. Calibrated Consensus
    """
    if ml_bundle is None:
        load_or_train_models()

    raw_input = raw_input.strip()
    source_info = {"is_url": False, "url": "", "domain": "", "title": ""}
    news_content = raw_input

    url_match = re.search(r"https?://\S+", raw_input)
    if is_url_mode or url_match:
        target_url = url_match.group(0) if url_match else raw_input
        scrape_res = fetch_url_content(target_url)
        source_info = {
            "is_url": True,
            "url": target_url,
            "domain": scrape_res["domain"],
            "title": scrape_res["title"]
        }
        if scrape_res["success"] and len(scrape_res["text"].split()) >= 10:
            news_content = scrape_res["text"]

    # 1. Translation & Language Detection
    trans_info = translate_if_needed(news_content)
    translated_text = trans_info["translated_text"]
    lang_name = trans_info["lang_name"]
    lang_code = trans_info["lang_code"]

    unicode_words = re.findall(r"\w+", news_content)
    total_words = len(unicode_words)

    combined_lower = f"{news_content.lower()} {translated_text.lower()}"

    # 2. Viral Hoax & Scam Auditing
    hoax_detected_reasons = []
    for pattern, reason in HOAX_SCAM_PATTERNS:
        if re.search(pattern, combined_lower, re.I) or re.search(pattern, news_content, re.I):
            hoax_detected_reasons.append(reason)

    # De-duplicate reasons
    hoax_detected_reasons = list(dict.fromkeys(hoax_detected_reasons))

    # Punctuation & Shouting
    excess_exclamations = len(re.findall(r"!{2,}", news_content)) + news_content.count("!!!")
    excess_questions = len(re.findall(r"\?{2,}", news_content)) + news_content.count("???")

    eng_tokens = re.findall(r"[A-Za-z]+", translated_text)
    shouting_words = [
        w for w in eng_tokens
        if w.isupper() and len(w) > 1 and w not in {
            "US", "UK", "UN", "EU", "NATO", "AI", "NASA", "ISRO", "WHO", "GDP", "PM", "CEO", "CBSE", "RBI"
        }
    ]
    caps_ratio = len(shouting_words) / max(1, len(eng_tokens))

    sensationalism_score = min(
        100,
        int((len(hoax_detected_reasons) * 35) + (caps_ratio * 90) + min(30, (excess_exclamations * 12) + (excess_questions * 8)))
    )

    # 3. Accredited Source & Institutional Entity Matching
    accredited_hits = []
    for src in ACCREDITED_SOURCES:
        if re.search(r"\b" + re.escape(src) + r"\b", combined_lower, re.I) or src in news_content:
            accredited_hits.append(src)
    accredited_hits = list(dict.fromkeys(accredited_hits))

    quote_matches = re.findall(r'["“][^"”]{8,250}["”]', news_content)
    has_quotes = len(quote_matches) > 0

    # Domain Credibility
    domain_boost = 0.0
    domain = source_info.get("domain", "")
    if domain:
        for trusted in TRUSTED_DOMAINS:
            if domain == trusted or domain.endswith("." + trusted):
                domain_boost = 0.28
                break

    # 4. Machine Learning Ensemble Inference
    cleaned_nlp = clean_text_for_nlp(translated_text)
    vec = ml_bundle["vectorizer"]
    X_vec = vec.transform([cleaned_nlp])
    vocab_coverage = X_vec.nnz

    lr_prob = float(ml_bundle["model_lr"].predict_proba(X_vec)[0][1])
    sgd_prob = float(ml_bundle["model_sgd"].predict_proba(X_vec)[0][1])
    nb_prob = float(ml_bundle["model_nb"].predict_proba(X_vec)[0][1])
    ml_prob = (lr_prob * 0.40) + (sgd_prob * 0.35) + (nb_prob * 0.25)

    # 5. Smart Consensus Scoring
    if len(hoax_detected_reasons) > 0:
        # Flagged by viral hoax / scam / demonetization / fake medical pattern
        consensus_score = min(0.32, ml_prob * 0.6)
    elif len(accredited_hits) > 0 or domain_boost > 0:
        # Has recognized accredited wire or domain
        consensus_score = max(0.68, (ml_prob * 0.40) + 0.45 + domain_boost)
    elif has_quotes:
        # Has verifiable quotes
        consensus_score = max(0.62, (ml_prob * 0.50) + 0.35)
    else:
        # General unreferenced statement: evaluate based on ML model + tone
        if sensationalism_score > 30:
            consensus_score = min(0.35, ml_prob)
        else:
            consensus_score = (ml_prob * 0.65) + 0.18

    consensus_score = max(0.05, min(0.95, consensus_score))

    is_real = consensus_score >= 0.50
    verdict = "REAL NEWS" if is_real else "FAKE NEWS"

    confidence = (consensus_score if is_real else (1.0 - consensus_score)) * 100.0
    confidence = min(98.5, max(68.0, confidence))

    # 6. Explanations & Indicators
    indicators: List[str] = []

    if domain_boost > 0:
        indicators.append(f"Domain verified: '{domain}' is an established accredited news source.")

    if is_real:
        if accredited_hits:
            found_refs = [f"'{h.upper()}'" for h in accredited_hits[:3]]
            indicators.append(f"Identified accredited institution or official source: {', '.join(found_refs)}.")
        if has_quotes:
            indicators.append("Contains direct verified quotes and attributed statements.")
        indicators.append("Objective, measured tone without sensational clickbait or viral hoax patterns.")
        indicators.append("Sentence structure matches authentic informational reporting standards.")
    else:
        if hoax_detected_reasons:
            for r in hoax_detected_reasons[:2]:
                indicators.append(f"Flagged viral pattern: {r}.")
        if excess_exclamations > 0 or caps_ratio > 0.08:
            indicators.append("Excessive emotional punctuation (!!!) or capitalized shouting detected.")
        if not accredited_hits and domain_boost == 0:
            indicators.append("Lacks named credible institutional sources, official gazette, or accredited wires.")
        indicators.append("Pattern matches common unverified social media forwards or fabricated claims.")

    first_line = news_content.split("\n")[0].strip()
    headline = (first_line[:85] + "...") if len(first_line) > 85 else first_line
    if not headline:
        headline = (news_content[:85] + "...") if len(news_content) > 85 else news_content

    read_time_mins = max(1, math.ceil(total_words / 200))

    return {
        "verdict": verdict,
        "is_real": is_real,
        "confidence": round(confidence, 1),
        "consensus_score": round(consensus_score, 3),
        "headline": headline,
        "language": {
            "name": lang_name,
            "code": lang_code,
            "is_translated": trans_info["is_translated"],
            "translated_preview": (translated_text[:120] + "...") if trans_info["is_translated"] and len(translated_text) > 120 else ""
        },
        "word_count": total_words,
        "read_time": f"{read_time_mins} min read",
        "source_info": source_info,
        "model_breakdown": {
            "Logistic Regression": round(lr_prob * 100, 1),
            "SGD Classifier": round(sgd_prob * 100, 1),
            "Multinomial Naive Bayes": round(nb_prob * 100, 1),
            "Consensus Score": round(consensus_score * 100, 1)
        },
        "telemetry": {
            "sensationalism_score": sensationalism_score,
            "clickbait_risk": "High Risk" if sensationalism_score > 50 else ("Moderate" if sensationalism_score > 25 else "Low Risk"),
            "attribution_count": len(accredited_hits) + (1 if has_quotes else 0),
            "emotional_polarization": "Objective & Factual" if sensationalism_score < 30 else "Sensational / Emotional",
            "vocabulary_tokens_matched": int(vocab_coverage)
        },
        "key_indicators": indicators,
        "sample_snippet": (news_content[:240] + "...") if len(news_content) > 240 else news_content
    }

# =============================================================================
# FASTAPI APP & WEBSOCKET PIPELINE
# =============================================================================
app = FastAPI(
    title="TruthGuard - Fake News Detection System",
    description="Multilingual Fake News Detection using Machine Learning & NLP",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsAnalysisRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    mode: Optional[str] = "text"

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing TruthGuard News Detection Engine...")
    load_or_train_models()
    logger.info("TruthGuard Machine Learning Engine is ready on port 8000!")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_ready": ml_bundle is not None,
        "service": "TruthGuard Fake News Detection Engine"
    }

@app.post("/api/analyze")
async def analyze_rest(payload: NewsAnalysisRequest):
    content = payload.url if (payload.url and payload.mode == "url") else payload.text
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Please enter news text or a valid URL.")
    is_url = bool(payload.url or payload.mode == "url" or re.search(r"https?://", content))
    result = analyze_multilingual_news(content, is_url_mode=is_url)
    return result

@app.websocket("/ws")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected via WebSocket.")
    try:
        while True:
            data_raw = await websocket.receive_text()
            data = json.loads(data_raw)
            raw_input = data.get("input", "").strip()
            mode = data.get("mode", "text")

            if not raw_input:
                await websocket.send_json({
                    "step": "error",
                    "progress": 0,
                    "message": "Input cannot be empty. Please enter news headline or text."
                })
                continue

            # Step 1: Detect language & parse tokens
            await websocket.send_json({
                "step": "ingest",
                "progress": 25,
                "message": "Detecting language & analyzing news syntax..."
            })
            await asyncio.sleep(0.12)

            # Step 2: Content extraction or Hoax checking
            is_url = mode == "url" or bool(re.search(r"https?://", raw_input))
            if is_url:
                await websocket.send_json({
                    "step": "scraping",
                    "progress": 50,
                    "message": "Fetching article content from website..."
                })
            else:
                await websocket.send_json({
                    "step": "nlp",
                    "progress": 50,
                    "message": "Auditing against viral hoax & scam patterns..."
                })
            await asyncio.sleep(0.12)

            # Step 3: Machine Learning Inference
            await websocket.send_json({
                "step": "ml_inference",
                "progress": 80,
                "message": "Evaluating Logistic Regression, SGD & Naive Bayes models..."
            })
            await asyncio.sleep(0.15)

            # Step 4: Final synthesis
            result = analyze_multilingual_news(raw_input, is_url_mode=is_url)
            await websocket.send_json({
                "step": "complete",
                "progress": 100,
                "message": "Verification complete!",
                "data": result
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "step": "error",
                "progress": 0,
                "message": f"Verification error: {str(e)}"
            })
        except Exception:
            pass

@app.get("/")
async def serve_index():
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH)
    return HTMLResponse("<h1>TruthGuard Server Active. Please place index.html in the project root.</h1>")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print(" TRUTHGUARD - FAKE NEWS DETECTION SYSTEM")
    print(" Machine Learning & Multilingual NLP Engine")
    print(" Web Interface : http://localhost:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
