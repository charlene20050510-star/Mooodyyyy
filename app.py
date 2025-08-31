from flask import Flask, request, redirect, session, url_for
import os
import time
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import math
from typing import List, Dict
from spotipy.exceptions import SpotifyException
from datetime import datetime
import random, re
from flask import request, redirect, url_for, session
import random, re
import traceback



app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "devsecret")
SCOPE = "user-library-read user-top-read playlist-modify-public playlist-modify-private"
app.secret_key = "replace-with-a-long-random-secret"

# ======================================================
# Flask & Spotify OAuth setup
# ======================================================

# ========= [新增] 語意描述與分數工具 =========

def _feature_words(f: Dict) -> List[str]:
    """把 Spotify audio features 轉成幾個簡單形容詞（英文），讓文字向量可用"""
    if not f:
        return []
    w = []
    en = f.get("energy"); va = f.get("valence"); da = f.get("danceability")
    ac = f.get("acousticness"); ins = f.get("instrumentalness"); te = f.get("tempo")

    if en is not None:
        if en >= 0.66: w.append("energetic")
        elif en <= 0.34: w.append("calm")
        else: w.append("mid-energy")

    if va is not None:
        if va >= 0.66: w.append("happy")
        elif va <= 0.34: w.append("sad")
        else: w.append("neutral-mood")

    if da is not None:
        if da >= 0.6: w.append("danceable")
        else: w.append("not-very-danceable")

    if ac is not None:
        if ac >= 0.6: w.append("acoustic")
        else: w.append("electronic")

    if ins is not None and ins >= 0.5:
        w.append("instrumental")
    else:
        w.append("vocal")

    if te:
        try:
            w.append(f"{int(round(te))} bpm")
        except Exception:
            pass

    return w


def _track_desc(track: Dict, feat: Dict) -> str:
    """把一首歌轉成一段可被 embedding 的簡短文字描述"""
    name = track.get("name", "")
    artists = ", ".join([a.get("name", "") for a in (track.get("artists") or [])])
    words = " ".join(_feature_words(feat))
    return f"{name} by {artists}. {words}".strip()


def _cosine(a: List[float], b: List[float]) -> float:
    """不依賴 numpy 的 cosine 相似度"""
    dot = sum((x*y for x, y in zip(a, b)))
    na = math.sqrt(sum((x*x for x in a))) + 1e-8
    nb = math.sqrt(sum((y*y for y in b))) + 1e-8
    return dot / (na * nb)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    做文字 embedding（自動偵測新/舊版 openai 套件；失敗就回空）
    你環境要有 OPENAI_API_KEY
    """
    try:
        # 新版 openai 套件
        from openai import OpenAI
        client = OpenAI()
        res = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in res.data]
    except Exception:
        try:
            # 舊版 openai 套件
            import openai
            res = openai.Embedding.create(model="text-embedding-ada-002", input=texts)
            return [d["embedding"] for d in res["data"]]
        except Exception as e:
            print(f"[warn] embedding failed: {e}")
            return []


def _numeric_affinity(feat: Dict, params: Dict) -> float:
    """
    用音樂特徵算一個 0~1 的接近度：energy/valence/danceability/acousticness/tempo
    有就算、沒有就跳過；最後取平均。
    """
    if not feat:
        return 0.5
    score_sum, cnt = 0.0, 0

    def closeness(v, t, scale=1.0):
        # v, t 在 0~1 範圍時直接用；tempo 用 scale 正規化
        return max(0.0, 1.0 - abs((v - t) / scale))

    for k in ("energy", "valence", "danceability", "acousticness"):
        vk = feat.get(k); tk = params.get(f"target_{k}")
        if vk is not None and tk is not None:
            score_sum += closeness(vk, tk, 1.0); cnt += 1

    # tempo：以 120 bpm 當 1 個 scale（可調）
    vtempo = feat.get("tempo"); ttempo = params.get("target_tempo")
    if vtempo and ttempo:
        score_sum += closeness(vtempo, ttempo, 120.0); cnt += 1
    elif vtempo and (params.get("min_tempo") or params.get("max_tempo")):
        # 若只有區間：落在區間內給 1，偏離則線性遞減
        lo = params.get("min_tempo", vtempo); hi = params.get("max_tempo", vtempo)
        if lo <= vtempo <= hi:
            score_sum += 1.0
        else:
            edge = lo if vtempo < lo else hi
            score_sum += max(0.0, 1.0 - abs(vtempo - edge) / 120.0)
        cnt += 1

    return (score_sum / cnt) if cnt else 0.5


def build_semantic_map(prompt: str, tracks: List[Dict], feats_map: Dict[str, Dict]) -> Dict[str, float]:
    """
    回傳 {track_id: 語意相似度(0~1)}。
    實作：把每首歌轉成短描述 → 和 prompt 一起丟 embedding → 計算 cosine。
    如果 embedding 失敗，回傳所有 0.5（不中斷流程）。
    """
    # 準備描述
    tids, descs = [], []
    for tr in tracks:
        tid = tr.get("id")
        if isinstance(tid, str) and len(tid) == 22 and tid in feats_map:
            tids.append(tid)
            descs.append(_track_desc(tr, feats_map.get(tid)))

    if not tids:
        return {}

    embs = _embed_texts([prompt] + descs)
    if not embs or len(embs) != (1 + len(descs)):
        # 失敗：給所有人 0.5
        print("[warn] embedding empty or length mismatch; fallback to 0.5")
        return {tid: 0.5 for tid in tids}

    q = embs[0]
    sims = {}
    for i, tid in enumerate(tids):
        sims[tid] = max(0.0, min(1.0, _cosine(q, embs[i + 1])))

    return sims


def rank_pool_by_semantic_and_features(pool: List[Dict], feats_map: Dict[str, Dict],
                                       sem_map: Dict[str, float], params: Dict,
                                       top_n: int) -> List[Dict]:
    """
    對 pool 排序：final = 0.6 * 語意 + 0.4 * 數值特徵接近度
    排完回傳前 top_n（每首歌在 dict 裡加 _score 供除錯）。
    """
    scored = []
    for tr in pool:
        tid = tr.get("id")
        if not (isinstance(tid, str) and len(tid) == 22):
            continue
        f = feats_map.get(tid)
        sem = sem_map.get(tid, 0.5)
        num = _numeric_affinity(f, params)
        final = 0.6 * sem + 0.4 * num
        tr["_score"] = final
        scored.append(tr)
    scored.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return scored[:top_n]
# ========= [新增工具結束] =========

def oauth():
    """Create a SpotifyOAuth instance. Redirect URI must exactly match Spotify Dashboard."""
    return SpotifyOAuth(
        client_id=os.environ.get("SPOTIPY_CLIENT_ID"),
        client_secret=os.environ.get("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.environ.get("SPOTIPY_REDIRECT_URI"),
        scope=SCOPE,
        cache_path=None,
        open_browser=False,
        show_dialog=True,  # force account picker / consent refresh
    )


# ---------------- Token helpers (access + refresh) -----------------

def _store_token(token_info):
    if not token_info:
        return
    old = session.get("token", {})
    if old and not token_info.get("refresh_token"):
        token_info["refresh_token"] = old.get("refresh_token")
    session["token"] = token_info


def _get_valid_token():
    tok = session.get("token")
    if not tok:
        return None
    # refresh 60s before expiry if we have a refresh token
    try:
        if time.time() > (tok.get("expires_at", 0) - 60) and tok.get("refresh_token"):
            new_tok = oauth().refresh_access_token(tok["refresh_token"]) or {}
            _store_token({**tok, **new_tok})
            tok = session.get("token")
    except Exception as e:
        print(f"⚠️ refresh_access_token failed: {e}")
    return tok


def get_spotify_client():
    tok = _get_valid_token()
    if not tok:
        return None
    return spotipy.Spotify(auth=tok.get("access_token"), requests_timeout=15)


# ======================================================
# Simple rules → target features (can swap to embeddings later)
# ======================================================

# === 風格詞彙 → 音訊特徵對映 ===
STYLE_MAP = {
    "lofi":       {"energy": (0.0, 0.4), "acousticness": (0.6, 1.0), "tempo": (60, 90)},
    "jazz":       {"energy": (0.2, 0.6), "instrumentalness": (0.4, 1.0)},
    "edm":        {"energy": (0.7, 1.0), "danceability": (0.7, 1.0), "tempo": (120, 150)},
    "rock":       {"energy": (0.6, 1.0), "instrumentalness": (0.0, 0.5)},
    "classical":  {"energy": (0.0, 0.3), "acousticness": (0.7, 1.0), "instrumentalness": (0.7, 1.0)},
    "hip hop":    {"energy": (0.6, 0.9), "speechiness": (0.5, 1.0), "tempo": (80, 110)},
    "r&b":        {"energy": (0.3, 0.7), "danceability": (0.5, 0.9)},
    "rb":         {"energy": (0.3, 0.7), "danceability": (0.5, 0.9)},  # 容錯
}

def map_text_to_params(text: str) -> dict:
    """
    將使用者情境文字轉成「音訊特徵偏好」的參數範圍。
    回傳格式示意：
      {
        "energy": (0.2, 0.6),
        "danceability": (0.6, 1.0),
        "acousticness": (0.5, 1.0),
        "tempo": (90, 120),
        ...
      }
    """
    t = (text or "").lower()
    params = {}

    # 風格詞彙對映（可多個同時出現，後面命中的詞彙會覆蓋前者的同名鍵）
    for key, feat in STYLE_MAP.items():
        if key in t:
            params.update(feat)

    # 一些簡易關鍵詞微調（可自行增減）
    if "睡覺" in t or "助眠" in t or "放鬆" in t or "冥想" in t:
        params.setdefault("energy", (0.0, 0.4))
        params.setdefault("acousticness", (0.5, 1.0))
        params.setdefault("tempo", (55, 85))
    if "讀書" in t or "專心" in t or "focus" in t:
        params.setdefault("energy", (0.2, 0.5))
        params.setdefault("instrumentalness", (0.3, 1.0))
    if "運動" in t or "健身" in t or "跑步" in t or "workout" in t:
        params.setdefault("energy", (0.7, 1.0))
        params.setdefault("danceability", (0.6, 1.0))
        params.setdefault("tempo", (120, 170))
    if "爵士" in t:
        params.setdefault("energy", (0.2, 0.6))
        params.setdefault("instrumentalness", (0.4, 1.0))

    return params

# ======================================================
# Spotify helpers (fetch pool, features, ranking)
# ======================================================
CACHE = {"feat": {}}

def fetch_playlist_tracks(sp, playlist_id, max_n=100):
    """Fetch items from a playlist with resilient market fallbacks."""
    tracks = []
    offset = 0
    market_trials = [None, "from_token", "TW", "US"]
    while len(tracks) < max_n:
        try:
            market = market_trials[min(offset // 200, len(market_trials) - 1)]
            kwargs = {"limit": 50, "offset": offset}
            if market is not None:
                kwargs["market"] = market
            batch = sp.playlist_items(playlist_id, **kwargs)
            items = (batch or {}).get("items", [])
            if not items:
                break
            for it in items:
                tr = (it or {}).get("track") or {}
                if tr.get("id") and tr.get("is_playable", True):
                    tracks.append(tr)
                if len(tracks) >= max_n:
                    break
            if not (batch or {}).get("next"):
                break
            offset += 50
        except Exception as e:
            print(f"⚠️ fetch_playlist_tracks failed ({playlist_id}): {e}")
            break
    return tracks
# ---------- Collect user / external pools and pick a 3+7 mix ----------

def collect_user_tracks(sp, max_n=150):
    """抓使用者的常聽/已儲存歌曲，優先常聽。"""
    pool = []
    # Top tracks
    try:
        tops = sp.current_user_top_tracks(limit=50, time_range="medium_term")
        for it in (tops or {}).get("items", []):
            if it and it.get("id"):
                pool.append(it)
            if len(pool) >= max_n:
                return pool[:max_n]
    except Exception as e:
        print(f"⚠️ current_user_top_tracks failed: {e}")

    # Saved tracks
    try:
        offset = 0
        while len(pool) < max_n:
            saved = sp.current_user_saved_tracks(limit=50, offset=offset)
            items = (saved or {}).get("items", [])
            if not items:
                break
            for it in items:
                tr = (it or {}).get("track") or {}
                if tr and tr.get("id"):
                    pool.append(tr)
                if len(pool) >= max_n:
                    break
            offset += 50
    except Exception as e:
        print(f"⚠️ current_user_saved_tracks failed: {e}")

    return pool[:max_n]


def collect_external_tracks(sp, max_n=300):
    """抓外部來源（不依賴固定 ID，避免 404）。"""
    pool = []

    # Featured playlists（區域自動）
    try:
        featured = sp.featured_playlists(country="TW")
        for pl in (featured or {}).get("playlists", {}).get("items", [])[:8]:
            pool.extend(fetch_playlist_tracks(sp, pl.get("id"), max_n=80))
            if len(pool) >= max_n:
                return pool[:max_n]
    except Exception as e:
        print(f"⚠️ featured_playlists failed: {e}")

    # 類別歌單（再補）
    try:
        cats = sp.categories(country="TW", limit=6)
        for c in (cats or {}).get("categories", {}).get("items", []):
            cps = sp.category_playlists(category_id=c.get("id"), country="TW", limit=3)
            for pl in (cps or {}).get("playlists", {}).get("items", []):
                pool.extend(fetch_playlist_tracks(sp, pl.get("id"), max_n=60))
                if len(pool) >= max_n:
                    return pool[:max_n]
    except Exception as e:
        print(f"⚠️ categories fallback failed: {e}")

    # 最後才試固定 ID（可能 404，但無所謂）
    public_lists = [
        "37i9dQZF1DXcBWIGoYBM5M",  # Today's Top Hits
        "37i9dQZEVXbMDoHDwVN2tF",  # Global Top 50
        "37i9dQZF1DX4dyzvuaRJ0n",  # Hot Hits Taiwan
    ]
    for pid in public_lists:
        try:
            pool.extend(fetch_playlist_tracks(sp, pid, max_n=100))
            if len(pool) >= max_n:
                break
        except Exception as e:
            print(f"⚠️ public playlist fallback failed: {e}")

    return pool[:max_n]


def pick_top_n(tracks, feats, params, n, used_ids=None):
    """從 tracks 用 scoring 挑 n 首，避開 used_ids。"""
    used_ids = used_ids or set()
    scored = []
    for t in tracks:
        tid = t.get("id")
        if not tid or tid in used_ids:
            continue
        s = score_track(feats.get(tid), params)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0])
    out = []
    for s, t in scored:
        tid = t.get("id")
        if tid in used_ids:
            continue
        out.append(t)
        used_ids.add(tid)
        if len(out) >= n:
            break
    return out

def _safe_audio_features(sp, ids):
    """
    穩健版 audio_features：
    - 去重
    - 只保留看起來像 Spotify track 的 22 字元 id
    - 以批次(<=50)查；若批次失敗（403/400），把批次切成兩半遞迴重試
    """
    # 只留 22 長度的字串，避免混到 episode/local/非法 id
    clean = [i for i in ids if isinstance(i, str) and len(i) == 22]
    seen = set()
    clean = [i for i in clean if not (i in seen or seen.add(i))]

    feats = {}

    def fetch_chunk(chunk):
        if not chunk:
            return
        try:
            res = sp.audio_features(tracks=chunk) or []
            for f in res:
                if not f:
                    continue
                tid = f.get("id")
                if tid:
                    feats[tid] = f
        except Exception as e:
            # 批次失敗就切半重試，直到單顆
            if len(chunk) == 1:
                print(f"⚠️ audio_features single-id failed: {chunk[0]} -> {e}")
                return
            mid = len(chunk) // 2
            fetch_chunk(chunk[:mid])
            fetch_chunk(chunk[mid:])

    # 以 50 筆為一批（官方上限 100；保守一點更穩）
    for i in range(0, len(clean), 50):
        fetch_chunk(clean[i:i+50])

    return feats

def audio_features_map(sp, track_ids, batch_size: int = 50):
    """
    安全版：批次查 audio features；失敗時改逐首查，403/無資料就跳過。
    回傳 {track_id: features_dict}（只包含成功拿到特徵的歌）
    """
    # 1) 過濾合法 id（Spotify track id 長度 22）
    valid_ids = [
        tid for tid in track_ids
        if isinstance(tid, str) and len(tid) == 22
    ]

    feats = {}
    skipped = []  # 紀錄拿不到特徵而被跳過的 id

    def _single_lookup(tid: str):
        """單首查詢；拿不到就跳過"""
        try:
            row = sp.audio_features([tid])  # 會回 list 長度 1
            if row and isinstance(row, list) and row[0]:
                feats[tid] = row[0]
            else:
                skipped.append(tid)
        except SpotifyException as se:
            status = getattr(se, "http_status", None)
            print(f"[warn] audio_features single-id failed: {tid} -> status {status}")
            skipped.append(tid)
        except Exception as se:
            print(f"[warn] audio_features single-id failed: {tid} -> {se}")
            skipped.append(tid)

    # 2) 批次查，失敗再逐首補
    for i in range(0, len(valid_ids), batch_size):
        chunk = valid_ids[i:i + batch_size]
        try:
            rows = sp.audio_features(chunk)  # 正常會回每個 id 對應的 row
            # rows 可能是 None 或含 None，逐一檢查
            if not rows:
                # 如果整包 None，逐首補查
                print(f"[warn] batch audio_features empty for {len(chunk)} ids; fallback to single")
                for tid in chunk:
                    _single_lookup(tid)
                continue

            for tid, row in zip(chunk, rows):
                if row and isinstance(row, dict):
                    feats[tid] = row
                else:
                    # 這首拿不到 → 單首再試一次；還是不行就跳過
                    _single_lookup(tid)

        except SpotifyException as e:
            status = getattr(e, "http_status", None)
            print(f"[warn] batch audio_features failed (status {status}): fallback to single for {len(chunk)} ids")
            for tid in chunk:
                _single_lookup(tid)

        except Exception as e:
            print(f"[warn] batch audio_features unexpected error: {e}; fallback to single for {len(chunk)} ids")
            for tid in chunk:
                _single_lookup(tid)

    print(f"[features] ok={len(feats)}  skipped={len(set(skipped))}")
    return feats

# ========= 依情境切換外部來源（耐撞版：歌單→搜尋→推薦 三層退場） =========
def collect_external_tracks_by_category(sp, text: str, max_n: int = 200):
    """
    依輸入文字分類，從：1) 精選歌單ID → 2) 站內搜尋歌單 → 3) Spotify Recommendations
    三層來源抓外部歌曲；任何一層抓到就先收，盡量湊滿 max_n。
    """
    text = (text or "").lower()

    # 1) 分類
    if any(k in text for k in ["party", "派對", "嗨", "開心", "快樂"]):
        category = "party"
    elif any(k in text for k in ["sad", "傷心", "難過", "哭", "失戀", "emo"]):
        category = "sad"
    elif any(k in text for k in ["chill", "放鬆", "冷靜", "悠閒", "輕鬆"]):
        category = "chill"
    elif any(k in text for k in ["focus", "讀書", "專注", "工作", "coding", "專心"]):
        category = "focus"
    else:
        category = "default"

    print(f"[external] category={category}")

    # 2) 第一層：我們手選的公開歌單 IDs（可能因區域/下架失效，所以只是優先嘗試）
    curated = {
        "party":   ["37i9dQZF1DX0BcQWzuB7ZO", "37i9dQZF1DXaXB8fQg7xif"],
        "sad":     ["37i9dQZF1DX7qK8ma5wgG1", "37i9dQZF1DX3YSRoSdA634"],
        "chill":   ["37i9dQZF1DX4WYpdgoIcn6", "37i9dQZF1DWUvQoIOFMFUT"],
        "focus":   ["37i9dQZF1DX8Uebhn9wzrS", "37i9dQZF1DX9sIqqvKsjG8"],
        "default": ["37i9dQZF1DXcBWIGoYBM5M", "37i9dQZF1DX1s9knjP51Oa"],
    }
    playlist_ids = curated.get(category, curated["default"])

    tracks = []

    def _pull_from_playlists(pids, budget):
        got = []
        for pid in pids:
            if len(got) >= budget:
                break
            try:
                # 注意：不加 market 限制，避免整包被過濾成空
                pl = sp.playlist_items(pid, additional_types=["track"], limit=100)
                for item in (pl or {}).get("items", []):
                    tr = (item or {}).get("track")
                    if tr and isinstance(tr.get("id"), str):
                        got.append(tr)
                        if len(got) >= budget:
                            break
            except Exception as e:
                print(f"[warn] pull playlist {pid} failed: {e}")
        return got

    # 第一層：嘗試精選歌單
    tracks += _pull_from_playlists(playlist_ids, max_n)
    print(f"[external] curated_got={len(tracks)}")

    # 3) 第二層：站內搜尋相關歌單關鍵字（避免精選歌單失效/區域差異）
    if len(tracks) < max_n // 3:
        query_map = {
            "party":  ["edm", "dance party", "party pop"],
            "sad":    ["sad songs", "mellow", "heartbreak"],
            "chill":  ["chill", "lofi", "acoustic chill"],
            "focus":  ["focus", "study", "instrumental"],
            "default":["hot hits", "pop hits"],
        }
        queries = query_map.get(category, query_map["default"])
        found_pids = []
        for q in queries:
            try:
                res = sp.search(q=q, type="playlist", limit=3)
                pitems = (((res or {}).get("playlists") or {}).get("items") or [])
                for pl in pitems:
                    pid = (pl or {}).get("id")
                    if pid: found_pids.append(pid)
            except Exception as e:
                print(f"[warn] search playlists '{q}' failed: {e}")
        if found_pids:
            need = max_n - len(tracks)
            tracks += _pull_from_playlists(found_pids, need)
            print(f"[external] after_search_got={len(tracks)}")

    # 4) 第三層：Spotify Recommendations API（用 genre 當種子）
    if len(tracks) == 0:
        seed_map = {
            "party":  ["dance", "edm", "pop"],
            "sad":    ["sad", "acoustic", "mellow"],
            "chill":  ["chill", "ambient", "lo-fi"],
            "focus":  ["study", "instrumental", "ambient"],
            "default":["pop", "indie", "rock"],
        }
        seeds = seed_map.get(category, seed_map["default"])[:2]
        try:
            rec = sp.recommendations(seed_genres=seeds, limit=min(100, max_n))
            for tr in (rec or {}).get("tracks", []):
                if tr and isinstance(tr.get("id"), str):
                    tracks.append(tr)
            print(f"[external] recommendations_got={len(tracks)} seeds={seeds}")
        except Exception as e:
            print(f"[warn] recommendations failed: {e}")

    # 去重，保險起見
    dedup, seen = [], set()
    for tr in tracks:
        tid = tr.get("id")
        if tid and tid not in seen:
            dedup.append(tr); seen.add(tid)

    print(f"[external] total={len(dedup)} (before cap {max_n})")
    return dedup[:max_n]
# ======================================================
# Routes
# ======================================================

@app.route("/")
def home():
    return '''
<!doctype html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Mooodyyy - AI 音樂推薦</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Circular', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
            background: linear-gradient(135deg, #191414 0%, #0d1117 50%, #121212 100%);
            color: #ffffff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow-x: hidden;
        }
        
        .hero-container {
            text-align: center;
            max-width: 500px;
            padding: 40px 20px;
            position: relative;
        }
        
        .logo {
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #1DB954, #1ed760);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 16px;
            letter-spacing: -2px;
        }
        
        .tagline {
            font-size: 1.3rem;
            color: #b3b3b3;
            margin-bottom: 40px;
            font-weight: 300;
            line-height: 1.5;
        }
        
        .login-btn {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            background: linear-gradient(135deg, #1DB954, #1ed760);
            color: #000;
            padding: 16px 32px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 700;
            font-size: 1.1rem;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            box-shadow: 0 8px 32px rgba(29, 185, 84, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .login-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(29, 185, 84, 0.4);
        }
        
        .login-btn:active {
            transform: translateY(0);
        }
        
        .floating-notes {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: hidden;
        }
        
        .note {
            position: absolute;
            color: rgba(29, 185, 84, 0.1);
            font-size: 2rem;
            animation: float 8s infinite ease-in-out;
        }
        
        .note:nth-child(1) { left: 10%; animation-delay: 0s; }
        .note:nth-child(2) { left: 80%; animation-delay: 2s; }
        .note:nth-child(3) { left: 50%; animation-delay: 4s; }
        .note:nth-child(4) { left: 20%; animation-delay: 6s; }
        
        @keyframes float {
            0%, 100% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
            50% { transform: translateY(-20px) rotate(180deg); opacity: 1; }
        }
        
        .features {
            margin-top: 60px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            max-width: 600px;
        }
        
        .feature {
            text-align: center;
            padding: 20px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 12px;
            display: block;
        }
        
        .feature-title {
            font-weight: 600;
            margin-bottom: 8px;
            color: #1DB954;
        }
        
        .feature-desc {
            font-size: 0.9rem;
            color: #b3b3b3;
            line-height: 1.4;
        }
    </style>
</head>
<body>
    <div class="floating-notes">
        <div class="note">♪</div>
        <div class="note">♫</div>
        <div class="note">♪</div>
        <div class="note">♫</div>
    </div>
    
    <div class="hero-container">
        <h1 class="logo">Mooodyyy</h1>
        <p class="tagline">用 AI 讀懂你的心情<br>為每個時刻找到完美音樂</p>
        
        <a href="/login" class="login-btn">
            <span>🎧</span>
            <span>Connect with Spotify</span>
        </a>
        
        <div class="features">
            <div class="feature">
                <span class="feature-icon">🧠</span>
                <div class="feature-title">智能理解</div>
                <div class="feature-desc">描述情境，AI 自動解析情緒與場景</div>
            </div>
            <div class="feature">
                <span class="feature-icon">🎯</span>
                <div class="feature-title">精準推薦</div>
                <div class="feature-desc">結合你的喜好與音樂特徵分析</div>
            </div>
            <div class="feature">
                <span class="feature-icon">⚡</span>
                <div class="feature-title">即時生成</div>
                <div class="feature-desc">秒速創建專屬歌單</div>
            </div>
        </div>
    </div>
</body>
</html>
'''


@app.route("/login")
def login():
    return redirect(oauth().get_authorize_url())


@app.route("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return redirect(url_for("home"))
    try:
        token_info = oauth().get_access_token(code, as_dict=True)
        _store_token(token_info)
        return redirect(url_for("welcome"))
    except Exception as e:
        print(f"❌ OAuth callback error: {e}")
        return '''
<!doctype html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>授權失敗</title>
    <style>
        body {
            font-family: 'Circular', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #191414, #0d1117);
            color: #fff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .error-container {
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .retry-btn {
            background: #1DB954;
            color: #000;
            padding: 12px 24px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            margin-top: 20px;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="error-container">
        <h2>❌ 授權失敗</h2>
        <p>連接 Spotify 時發生錯誤</p>
        <a href="/" class="retry-btn">重新嘗試</a>
    </div>
</body>
</html>
'''

@app.route("/welcome")
def welcome():
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    try:
        me = sp.current_user()
        name = (me or {}).get("display_name") or "音樂愛好者"
    except Exception:
        name = "音樂愛好者"

    return f'''
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Mooodyyy - 開始創建歌單</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      font-family:'Circular',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif;
      background:linear-gradient(135deg,#191414 0%,#0d1117 50%,#121212 100%);
      color:#fff; min-height:100vh; line-height:1.6;
    }}
    .container {{ max-width:800px; margin:0 auto; padding:40px 20px;
      min-height:100vh; display:flex; flex-direction:column; justify-content:center; }}
    .header {{ text-align:center; margin-bottom:40px; }}
    .logo {{ font-size:2.5rem; font-weight:900;
      background:linear-gradient(135deg,#1DB954,#1ed760);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      margin-bottom:8px; letter-spacing:-1px; }}
    .welcome-text {{ font-size:1.2rem; color:#b3b3b3; margin-bottom:8px; }}
    .user-name {{ font-size:1.4rem; font-weight:600; color:#1DB954; }}
    .main-card {{
      background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08);
      border-radius:24px; padding:40px; backdrop-filter:blur(20px);
      box-shadow:0 20px 60px rgba(0,0,0,0.3);
      position:relative; overflow:hidden;
    }}
    .main-card::before {{
      content:''; position:absolute; top:0; left:0; right:0; height:1px;
      background:linear-gradient(90deg,transparent,#1DB954,transparent);
    }}
    textarea {{
      width:100%; min-height:120px; padding:20px; font-size:1.1rem;
      border-radius:16px; border:2px solid rgba(255,255,255,0.1);
      background:rgba(0,0,0,0.2); color:#fff; resize:vertical;
    }}
    textarea:focus {{
      outline:none; border-color:#1DB954;
      box-shadow:0 0 0 3px rgba(29,185,84,0.1);
      background:rgba(0,0,0,0.4);
    }}
    .submit-btn {{
      background:linear-gradient(135deg,#1DB954,#1ed760); color:#000;
      border:none; padding:16px 40px; border-radius:50px;
      font-size:1.1rem; font-weight:700; cursor:pointer;
      transition:all .3s cubic-bezier(.25,.46,.45,.94);
      box-shadow:0 8px 32px rgba(29,185,84,.25); width:100%;
    }}
    .examples {{ margin-top:32px; padding:24px; background:rgba(29,185,84,.05);
      border-radius:16px; border:1px solid rgba(29,185,84,.1); }}
    .example-tag {{
      background:rgba(255,255,255,.05); color:#b3b3b3;
      padding:8px 16px; border-radius:20px; font-size:.9rem;
      cursor:pointer; transition:.2s;
    }}
    .example-tag:hover {{ background:rgba(29,185,84,.1); color:#1DB954; }}
    .footer {{ text-align:center; margin-top:40px; }}
    .logout-link {{ color:#b3b3b3; text-decoration:none; font-size:.9rem; }}
    .logout-link:hover {{ color:#1DB954; }}

    /* === Loading Overlay === */
    .loading-overlay {{
      position:fixed; inset:0; background:rgba(0,0,0,0);
      backdrop-filter:blur(0px); display:flex; align-items:center; justify-content:center;
      opacity:0; pointer-events:none;
      transition:opacity .35s ease,background .35s ease,backdrop-filter .35s ease;
      z-index:9999;
    }}
    .loading-overlay.show {{
      opacity:1; background:rgba(0,0,0,.75); backdrop-filter:blur(6px); pointer-events:all;
    }}
    .loading-card {{
      display:flex; flex-direction:column; align-items:center; gap:14px;
      padding:32px 28px; border-radius:20px;
      background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);
      box-shadow:0 8px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
    }}
    .loading-logo {{
      width:72px;height:72px;border-radius:18px;display:grid;place-items:center;
      background:radial-gradient(circle at 30% 30%,#1ed760,#1DB954 60%,#128a3e 100%);
      filter:drop-shadow(0 6px 24px rgba(29,185,84,.35));
    }}
    .loading-logo svg {{ width:38px;height:38px;fill:#000; }}
    .loading-text {{ color:#e8e8e8; font-weight:700; letter-spacing:.2px; }}
    .loading-sub {{ color:#b3b3b3; font-size:.92rem; }}

    /* === Pulse 動畫 === */
    @keyframes pulse {{0%{{transform:scale(1);}}50%{{transform:scale(1.05);}}100%{{transform:scale(1);}}}}
    .loading-overlay.show .loading-logo {{ animation:pulse 1.4s ease-in-out infinite; }}

    /* === 打字效果 cursor === */
    .loading-text::after {{
      content:""; display:inline-block; width:1ch; height:1em;
      vertical-align:-0.2em; border-right:2px solid #e8e8e8;
      animation:caret .8s steps(1,end) infinite;
    }}
    @keyframes caret {{0%,49%{{opacity:1;}}50%,100%{{opacity:0;}}}}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1 class="logo">Mooodyyy</h1>
      <p class="welcome-text">歡迎回來</p>
      <p class="user-name">{name}</p>
    </div>
    <div class="main-card">
      <h2 class="form-title">描述你的當下情境</h2>
      <p class="form-subtitle">告訴我你的心情、活動或想要的氛圍，我會為你推薦最適合的歌單</p>
      <form id="gen-form" action="/recommend" method="post">
        <div class="textarea-container">
          <textarea name="text" placeholder="例如：深夜漫步、雨天咖啡廳寫作、專心讀書需要輕音樂..." required></textarea>
        </div>
        <input type="hidden" name="preview" value="1">
        <button type="submit" class="submit-btn">🎵 開始推薦音樂</button>
      </form>
      <div class="examples">
        <div class="examples-title">💡 靈感提示</div>
        <div class="example-tags">
          <span class="example-tag" onclick="fillExample(this)">深夜散步</span>
          <span class="example-tag" onclick="fillExample(this)">下雨天寫作</span>
          <span class="example-tag" onclick="fillExample(this)">週末早晨</span>
          <span class="example-tag" onclick="fillExample(this)">運動健身</span>
          <span class="example-tag" onclick="fillExample(this)">放鬆冥想</span>
        </div>
      </div>
    </div>
    <div class="footer"><a href="/logout" class="logout-link">登出 Spotify</a></div>
  </div>

  <!-- Overlay -->
  <div class="loading-overlay" id="loading">
    <div class="loading-card">
      <div class="loading-logo" aria-hidden="true">
        <svg viewBox="0 0 24 24"><path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0z"/></svg>
      </div>
      <div class="loading-text">🎧 為你量身打造歌單中...</div>
      <div class="loading-sub">Mooodyyy 正在理解你的情境</div>
    </div>
  </div>

  <script>
    function fillExample(el){{
      const t=document.querySelector('textarea[name="text"]');
      t.value=el.textContent; t.focus();
    }}
    document.addEventListener('DOMContentLoaded',function(){{
      const textarea=document.querySelector('textarea');
      const btn=document.querySelector('.submit-btn');
      const form=document.getElementById('gen-form');
      const loading=document.getElementById('loading');
      const textEl=document.querySelector('.loading-text');
      textarea.addEventListener('input',function(){{
        if(this.value.trim()){{btn.style.opacity='1';btn.style.pointerEvents='auto';}}
        else{{btn.style.opacity='0.7';}}
      }});
      form.addEventListener('submit',function(){{ loading.classList.add('show'); }});
      // 打字機文字輪播
      const msgs=[
        "🎧 為你量身打造歌單中...",
        "🔍 正在理解你的情境...",
        "💡 思考最適合的音樂...",
        "🎵 快好了，馬上帶來推薦..."
      ];
      let idx=0;
      function type(full,i=0){{
        textEl.textContent=full.slice(0,i);
        if(i<full.length) setTimeout(()=>type(full,i+1),24);
        else setTimeout(()=>{{idx=(idx+1)%msgs.length; type(msgs[idx],0);}},800);
      }}
      const obs=new MutationObserver(()=>{{
        if(loading.classList.contains('show')){{idx=0; type(msgs[idx],0);}}
      }});
      obs.observe(loading,{{attributes:true,attributeFilter:['class']}});
    }});
  </script>
</body>
</html>
'''


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    產生「Playlist 預覽頁」：
    - 顯示推薦歌曲清單
    - 支援：拖曳排序 / 單曲預覽 (Spotify embed) / 複製連結
    - 兩個表單：重新產生（保留 avoid）／存到 Spotify（依照拖曳後順序）
    - 存檔時顯示 Saving Overlay、並防雙送出
    """
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    # 1) 取得使用者名稱（失敗就用預設）
    try:
        me = sp.current_user()
        display_name = (me or {}).get("display_name") or "音樂愛好者"
    except Exception:
        display_name = "音樂愛好者"

    # 2) 拿表單資料
    context_text = (request.form.get("text") or "").strip()
    avoid_raw = (request.form.get("avoid") or "").strip()
    avoid_ids = set([x for x in avoid_raw.split(",") if x])

    # 3) 取得推薦清單
    # ---------------------------------------------------------
    # === TODO: 這裡接上你的推薦邏輯 ===
    # 期望輸出格式：
    # recommended = [
    #   {"id": "spotify_track_id", "name": "歌名", "artists": "歌手A, 歌手B", "uri": "spotify:track:..."},
    #   ...
    # ]
    # 可依 avoid_ids 過濾，避免重複
    recommended = []
    try:
        # 例如：
        # recommended = build_recommendations(sp, context_text, avoid_ids=avoid_ids, limit=10)
        pass
    except Exception as e:
        print("recommend() build error:", e)

    # 如果還沒接上邏輯，先給示範資料，讓 UI 能跑
    if not recommended:
        demo = [
            ("4uLU6hMCjMI75M1A2tKUQC", "Never Gonna Give You Up", "Rick Astley"),
            ("7ouMYWpwJ422jRcDASZB7P", "Beautiful Things", "Benson Boone"),
            ("3VlbOrM6nYPprVvzBZllE5", "River Flows In You", "Yiruma"),
            ("2X485T9Z5Ly0xyaghN73ed", "lovely (with Khalid)", "Billie Eilish, Khalid"),
            ("3AJwUDP919kvQ9QcozQPxg", "Yellow", "Coldplay"),
        ]
        for tid, name, artists in demo:
            if tid not in avoid_ids:
                recommended.append({"id": tid, "name": name, "artists": artists, "uri": f"spotify:track:{tid}"})

    # 4) 更新 avoid（把這次顯示的也加進去）
    shown_ids = [t["id"] for t in recommended if t.get("id")]
    avoid_ids |= set(shown_ids)
    avoid_str = ",".join(sorted(list(avoid_ids)))

    # 5) 組 HTML
    # ---------------------------------------------------------
    def track_row(idx, t):
        tid = t.get("id", "")
        name = t.get("name", "（未命名）")
        artists = t.get("artists", "")
        open_url = f"https://open.spotify.com/track/{tid}" if tid else "#"
        return f"""
        <div class="track-item" draggable="true" data-id="{tid}">
          <div class="drag-handle" title="拖曳排序">⠿</div>
          <div class="track-index">{idx:02d}</div>
          <div class="track-meta">
            <div class="track-name">{name}</div>
            <div class="track-artists">{artists}</div>
          </div>
          <div class="track-actions">
            <button type="button" class="btn btn-mini preview-btn" data-id="{tid}" title="預覽 30 秒">▶ 預覽</button>
            <button type="button" class="btn btn-mini copy-btn" data-url="{open_url}" title="複製連結">⧉ 複製</button>
            <a class="btn btn-mini open-btn" href="{open_url}" target="_blank" rel="noopener" title="在 Spotify 開啟">↗ 開啟</a>
          </div>
        </div>
        """

    tracks_html = "\n".join([track_row(i, t) for i, t in enumerate(recommended, start=1)])
    track_ids_str = ",".join(shown_ids)

    page = f"""
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>為你推薦的歌單 - Mooodyyy</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      font-family:'Circular',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif;
      background:linear-gradient(135deg,#191414 0%,#0d1117 50%,#121212 100%);
      color:#fff; min-height:100vh; line-height:1.6;
    }}
    .container {{ max-width:960px; margin:0 auto; padding:40px 20px; }}
    .header {{ text-align:center; margin-bottom:24px; }}
    .logo {{
      font-size:2rem; font-weight:900;
      background:linear-gradient(135deg,#1DB954,#1ed760);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
      margin-bottom:6px;
    }}
    .subtitle {{ color:#b3b3b3; }}
    .context-display {{
      background:rgba(29,185,84,.09); border:1px solid rgba(29,185,84,.22);
      border-radius:16px; padding:16px 18px; margin:18px auto 26px; color:#d9ffd9;
    }}
    .card {{
      background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
      border-radius:20px; padding:22px; backdrop-filter:blur(16px);
      box-shadow:0 16px 48px rgba(0,0,0,0.3);
    }}

    /* 清單與曲目 */
    .track-list {{ display:block; }}
    .track-item {{
      display:flex; align-items:center; gap:12px; padding:12px 10px; border-radius:12px;
      border:1px dashed rgba(255,255,255,0.06); background:rgba(255,255,255,0.02);
      transition:background .2s ease, border-color .2s ease, transform .15s ease;
      user-select:none;
    }}
    .track-item + .track-item {{ margin-top:10px; }}
    .track-item:hover {{ background:rgba(255,255,255,0.04); border-color:rgba(255,255,255,0.12); }}
    .track-item.dragging {{ opacity:.8; transform:scale(0.995); border-color:#1DB954; background:rgba(29,185,84,.08); }}
    .drag-handle {{ cursor:grab; color:#98ffa7; width:22px; text-align:center; font-size:1rem; opacity:.9; }}
    .track-index {{ width:36px; color:#1ed760; font-weight:800; opacity:.95; text-align:right; }}
    .track-meta {{ flex:1 1 auto; }}
    .track-name {{ font-weight:700; color:#fff; }}
    .track-artists {{ color:#b3b3b3; font-size:.95rem; }}
    .track-actions {{ display:flex; gap:8px; flex-wrap:wrap; }}

    /* 按鈕 */
    .actions {{ display:flex; gap:12px; margin-top:20px; flex-wrap:wrap; }}
    .btn {{
      appearance:none; border:none; border-radius:999px; cursor:pointer;
      font-weight:800; padding:12px 16px; line-height:1;
      transition:transform .12s ease, box-shadow .2s ease, filter .2s ease;
    }}
    .btn-mini {{ padding:8px 12px; font-size:.9rem; border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.06); color:#e8e8e8; }}
    .btn-primary {{
      background:linear-gradient(135deg,#1DB954,#1ed760); color:#000;
      box-shadow:0 10px 28px rgba(29,185,84,.32);
    }}
    .btn-secondary {{ background:rgba(255,255,255,.06); color:#e8e8e8; border:1px solid rgba(255,255,255,.14); }}
    .btn:disabled {{ opacity:.6; cursor:not-allowed; }}
    .btn:hover {{ transform:translateY(-1px); filter:brightness(1.02); }}

    .footer {{ text-align:center; margin-top:24px; }}
    .link {{ color:#b3b3b3; text-decoration:none; }}
    .link:hover {{ color:#1DB954; }}

    /* Saving Overlay */
    .saving-overlay {{
      position: fixed; inset: 0; z-index: 9999;
      display: flex; align-items: center; justify-content: center;
      background: rgba(0,0,0,0); backdrop-filter: blur(0px);
      opacity: 0; pointer-events: none;
      transition: opacity .35s ease, background .35s ease, backdrop-filter .35s ease;
    }}
    .saving-overlay.show {{
      opacity: 1; background: rgba(0,0,0,0.75); backdrop-filter: blur(6px);
      pointer-events: all;
    }}
    .saving-card {{
      display: flex; flex-direction: column; align-items: center; gap: 14px;
      padding: 32px 28px; border-radius: 20px;
      background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
      box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
    }}
    .saving-logo {{
      width: 72px; height: 72px; border-radius: 18px; display: grid; place-items: center;
      background: radial-gradient(circle at 30% 30%, #1ed760, #1DB954 60%, #128a3e 100%);
      filter: drop-shadow(0 6px 24px rgba(29,185,84,.35));
      animation: pulse 1.4s ease-in-out infinite;
    }}
    .saving-logo svg {{ width: 38px; height: 38px; fill: #000; }}
    .saving-text {{ color: #e8e8e8; font-weight: 700; letter-spacing: .2px; }}
    .saving-sub {{ color: #b3b3b3; font-size: .92rem; }}
    @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} 100% {{ transform: scale(1); }} }}

    /* 預覽 Modal */
    .modal {{
      position:fixed; inset:0; z-index:10000; display:none;
      align-items:center; justify-content:center;
      background:rgba(0,0,0,.6); backdrop-filter:blur(4px);
    }}
    .modal.show {{ display:flex; }}
    .modal-card {{
      width:min(560px,92vw); border-radius:16px; overflow:hidden;
      background:#121212; border:1px solid rgba(255,255,255,.12);
      box-shadow:0 20px 60px rgba(0,0,0,.5);
    }}
    .modal-head {{
      display:flex; align-items:center; justify-content:space-between;
      padding:12px 16px; background:#0f1115; color:#e8e8e8; font-weight:700;
    }}
    .modal-body {{ padding:12px 12px 16px; }}
    .close-btn {{ background:transparent; color:#e8e8e8; border:none; font-size:1.2rem; cursor:pointer; }}
    .toast {{
      position:fixed; bottom:18px; left:50%; transform:translateX(-50%);
      background:rgba(0,0,0,.8); color:#e8e8e8; padding:10px 14px; border-radius:10px;
      border:1px solid rgba(255,255,255,.12); display:none;
    }}
    .toast.show {{ display:block; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">Mooodyyy</div>
      <div class="subtitle">嗨，{display_name}，根據你的情境我先幫你配了幾首：</div>
    </div>

    <div class="context-display">「{context_text}」</div>

    <div class="card">
      <div id="track-list" class="track-list">
        {tracks_html}
      </div>

      <div class="actions">
        <!-- 重新產生（保留 avoid） -->
        <form id="regen-form" action="/recommend" method="post" style="display:inline;">
          <input type="hidden" name="text" value="{context_text}">
          <input id="regen-avoid" type="hidden" name="avoid" value="{avoid_str}">
          <button class="btn btn-secondary" type="submit">↻ 再幫我換一批</button>
        </form>

        <!-- 存到 Spotify：會依照拖曳後順序送出 -->
        <form id="save-form" action="/create_playlist" method="post" style="display:inline;">
          <input type="hidden" name="text" value="{context_text}">
          <input id="save-track-ids" type="hidden" name="track_ids" value="{track_ids_str}">
          <button class="btn btn-primary" type="submit">🎵 存到 Spotify</button>
        </form>
      </div>
    </div>

    <div class="footer">
      <a class="link" href="/welcome">← 回到輸入頁</a>
    </div>
  </div>

  <!-- Saving Overlay -->
  <div class="saving-overlay" id="saving">
    <div class="saving-card">
      <div class="saving-logo" aria-hidden="true">
        <svg viewBox="0 0 24 24" role="img" aria-label="Spotify">
          <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.84-.179-.84-.6 0-.359.24-.66.54-.78 4.56-1.021 8.52-.6 11.64.301.42.12.66.54.42.96zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.301.421-1.02.599-1.56.3z"/>
        </svg>
      </div>
      <div class="saving-text">🎵 正在建立你的 Spotify 歌單...</div>
      <div class="saving-sub">稍等一下下，我們把歌曲加入清單</div>
    </div>
  </div>

  <!-- 預覽 Modal -->
  <div id="modal" class="modal" aria-hidden="true">
    <div class="modal-card">
      <div class="modal-head">
        <div>🔊 單曲預覽</div>
        <button class="close-btn" type="button" id="modal-close">✕</button>
      </div>
      <div class="modal-body">
        <iframe id="preview-frame" style="border-radius:12px" src="" width="100%" height="152" frameborder="0" allow="autoplay; clipboard-write; encrypted-media; picture-in-picture" loading="lazy"></iframe>
      </div>
    </div>
  </div>

  <!-- Toast -->
  <div id="toast" class="toast">已複製連結！</div>

  <script>
    // ====== 工具 ======
    function showToast(msg) {{
      const t = document.getElementById('toast');
      t.textContent = msg || '已複製連結！';
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 1400);
    }}
    function currentIds() {{
      return Array.from(document.querySelectorAll('.track-item')).map(el => el.getAttribute('data-id')).filter(Boolean);
    }}
    function renumber() {{
      document.querySelectorAll('.track-item .track-index').forEach((el, i) => el.textContent = String(i+1).padStart(2,'0'));
    }}

    // ====== 複製連結 / 預覽播放 ======
    document.addEventListener('click', async (e) => {{
      const copyBtn = e.target.closest('.copy-btn');
      if (copyBtn) {{
        const url = copyBtn.getAttribute('data-url');
        try {{
          await navigator.clipboard.writeText(url);
          showToast('已複製連結！');
        }} catch (_) {{
          showToast('複製失敗，請手動複製');
        }}
      }}
      const previewBtn = e.target.closest('.preview-btn');
      if (previewBtn) {{
        const tid = previewBtn.getAttribute('data-id');
        const src = `https://open.spotify.com/embed/track/${{tid}}?utm_source=generator`;
        document.getElementById('preview-frame').src = src;
        document.getElementById('modal').classList.add('show');
      }}
      const closeBtn = e.target.closest('#modal-close');
      const modal = document.getElementById('modal');
      if (closeBtn || e.target === modal) {{
        modal.classList.remove('show');
        document.getElementById('preview-frame').src = '';
      }}
    }});

    // ====== 拖曳排序 (原生 DnD) ======
    const list = document.getElementById('track-list');
    let draggingEl = null;

    list.addEventListener('dragstart', (e) => {{
      const item = e.target.closest('.track-item');
      if (!item) return;
      draggingEl = item;
      item.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
      // for Firefox
      e.dataTransfer.setData('text/plain', item.getAttribute('data-id') || '');
    }});
    list.addEventListener('dragend', (e) => {{
      const item = e.target.closest('.track-item');
      if (item) item.classList.remove('dragging');
      draggingEl = null;
      renumber();
    }});
    list.addEventListener('dragover', (e) => {{
      e.preventDefault();
      const afterEl = getDragAfterElement(list, e.clientY);
      if (!draggingEl) return;
      if (afterEl == null) {{
        list.appendChild(draggingEl);
      }} else {{
        list.insertBefore(draggingEl, afterEl);
      }}
    }});
    function getDragAfterElement(container, y) {{
      const els = [...container.querySelectorAll('.track-item:not(.dragging)')];
      return els.reduce((closest, child) => {{
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        if (offset < 0 && offset > closest.offset) {{
          return {{ offset, element: child }};
        }} else {{
          return closest;
        }}
      }}, {{ offset: Number.NEGATIVE_INFINITY }}).element;
    }}

    // ====== 表單提交：寫入正確順序、顯示 Saving、與防重複送出 ======
    const saveForm = document.getElementById('save-form');
    const regenForm = document.getElementById('regen-form');
    const saving = document.getElementById('saving');

    function disableAllButtons() {{
      document.querySelectorAll('button').forEach(b => b.disabled = true);
    }}

    if (saveForm && saving) {{
      saveForm.addEventListener('submit', function () {{
        // 依拖曳後順序寫入
        const ids = currentIds();
        document.getElementById('save-track-ids').value = ids.join(',');
        disableAllButtons();
        saving.classList.add('show');
      }});
    }}
    if (regenForm) {{
      regenForm.addEventListener('submit', function () {{
        // 保留 avoid
        const ids = currentIds();
        // 把當前顯示的都加入 avoid，避免下次重複
        const avoidInput = document.getElementById('regen-avoid');
        const existed = (avoidInput.value || '').split(',').filter(Boolean);
        const merged = Array.from(new Set([...existed, ...ids]));
        avoidInput.value = merged.join(',');
        disableAllButtons();
      }});
    }}
  </script>
</body>
</html>
"""
    return page


@app.route("/create_playlist", methods=["POST"])
def create_playlist():
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    text = (request.form.get("text") or "").strip()

    # 1) 只用預覽頁送過來的這一批歌曲（看到什麼就存什麼）
    track_ids_raw = (request.form.get("track_ids") or "").strip()
    ids = [i for i in track_ids_raw.split(",") if len(i) == 22] if track_ids_raw else []

    if not ids:
        return '''
<!doctype html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>建立歌單失敗 - Mooodyyy</title>
    <style>
        body {
            font-family: 'Circular', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #191414, #0d1117);
            color: #fff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .error-container {
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .retry-btn {
            background: #1DB954;
            color: #000;
            padding: 12px 24px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            margin-top: 20px;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="error-container">
        <h3>❌ 沒有歌曲可以加入</h3>
        <p>請先生成歌單預覽，再點擊「存到 Spotify」</p>
        <a href="/welcome" class="retry-btn">↩︎ 回首頁</a>
    </div>
</body>
</html>
'''

    try:
        # 2) 建立「私人」歌單並加入歌曲
        user = sp.current_user()
        user_id = (user or {}).get("id")
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        title = f"Mooodyyy · {ts} UTC"
        desc  = f"情境：{text}（由預覽頁直接保存）"

        playlist = sp.user_playlist_create(
            user=user_id,
            name=title,
            public=False,   # 一律私人
            description=desc
        )
        sp.playlist_add_items(playlist_id=playlist["id"], items=ids)

        # 3) 成功頁面，然後自動跳轉到 Spotify
        playlist_url = (playlist.get("external_urls") or {}).get("spotify", url_for("welcome"))
        
        return f'''
<!doctype html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>歌單建立成功 - Mooodyyy</title>
    <style>
        body {{
            font-family: 'Circular', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #191414, #0d1117);
            color: #fff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .success-container {{
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            border: 1px solid rgba(29, 185, 84, 0.2);
            max-width: 500px;
            position: relative;
            overflow: hidden;
        }}
        .success-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, #1DB954, transparent);
        }}
        .success-icon {{
            font-size: 4rem;
            margin-bottom: 20px;
            animation: bounce 1s ease-in-out;
        }}
        .spotify-btn {{
            background: #1DB954;
            color: #000;
            padding: 16px 32px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 700;
            margin: 20px 10px;
            display: inline-block;
            transition: all 0.2s ease;
        }}
        .home-btn {{
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            padding: 12px 24px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            margin: 10px;
            display: inline-block;
        }}
        @keyframes bounce {{
            0%, 20%, 60%, 100% {{ transform: translateY(0); }}
            40% {{ transform: translateY(-10px); }}
            80% {{ transform: translateY(-5px); }}
        }}
    </style>
    <meta http-equiv="refresh" content="3;url={playlist_url}">
</head>
<body>
    <div class="success-container">
        <div class="success-icon">🎉</div>
        <h2>歌單建立成功！</h2>
        <p>你的專屬歌單已保存到 Spotify</p>
        <p style="color: #b3b3b3; margin: 16px 0;">正在跳轉到 Spotify...</p>
        <a href="{playlist_url}" class="spotify-btn">🎧 在 Spotify 中打開</a>
        <a href="/welcome" class="home-btn">↩︎ 回首頁</a>
    </div>
</body>
</html>
'''

    except Exception as e:
        print(f"❌ create_playlist error: {e}")
        return f'''
<!doctype html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>建立失敗 - Mooodyyy</title>
    <style>
        body {{
            font-family: 'Circular', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #191414, #0d1117);
            color: #fff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .error-container {{
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .retry-btn {{
            background: #1DB954;
            color: #000;
            padding: 12px 24px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            margin-top: 20px;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div class="error-container">
        <h2>❌ 建立歌單失敗</h2>
        <p>錯誤訊息：{str(e)}</p>
        <a href="/welcome" class="retry-btn">↩︎ 回首頁</a>
    </div>
</body>
</html>
'''


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/ping")
def ping():
    return "PING OK", 200


@app.route("/env")
def env_show():
    v = os.environ.get("SPOTIPY_REDIRECT_URI", "<none>")
    return f"[{v}] len={len(v)}", 200


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=PORT)
