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

    # 取用戶名稱（失敗就顯示「音樂愛好者」）
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
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Circular', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
            background: linear-gradient(135deg, #191414 0%, #0d1117 50%, #121212 100%);
            color: #ffffff;
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        
        .logo {{
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #1DB954, #1ed760);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            letter-spacing: -1px;
        }}
        
        .welcome-text {{
            font-size: 1.2rem;
            color: #b3b3b3;
            margin-bottom: 8px;
        }}
        
        .user-name {{
            font-size: 1.4rem;
            font-weight: 600;
            color: #1DB954;
        }}
        
        .main-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 40px;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }}
        
        .main-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, #1DB954, transparent);
        }}
        
        .character-container {{
            display: flex;
            justify-content: center;
            margin-bottom: 24px;
        }}
        
        .character-mascot {{
            width: 100px;
            height: auto;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(29, 185, 84, 0.2);
            transition: all 0.3s ease;
        }}
        
        .character-mascot:hover {{
            transform: scale(1.05);
            box-shadow: 0 12px 40px rgba(29, 185, 84, 0.3);
        }}
        
        .form-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: #ffffff;
            text-align: center;
        }}
        
        .form-subtitle {{
            color: #b3b3b3;
            margin-bottom: 32px;
            font-size: 1rem;
            text-align: center;
        }}
        
        .textarea-container {{
            position: relative;
            margin-bottom: 24px;
        }}
        
        textarea {{
            width: 100%;
            min-height: 120px;
            padding: 20px;
            font-size: 1.1rem;
            border-radius: 16px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            background: rgba(0, 0, 0, 0.2);
            color: #ffffff;
            resize: vertical;
            transition: all 0.3s ease;
            font-family: inherit;
            line-height: 1.5;
        }}
        
        textarea:focus {{
            outline: none;
            border-color: #1DB954;
            box-shadow: 0 0 0 3px rgba(29, 185, 84, 0.1);
            background: rgba(0, 0, 0, 0.4);
        }}
        
        textarea::placeholder {{
            color: #757575;
            font-style: italic;
        }}
        
        .submit-btn {{
            background: linear-gradient(135deg, #1DB954, #1ed760);
            color: #000000;
            border: none;
            padding: 16px 40px;
            border-radius: 50px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            box-shadow: 0 8px 32px rgba(29, 185, 84, 0.25);
            position: relative;
            overflow: hidden;
            width: 100%;
        }}
        
        .submit-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(29, 185, 84, 0.35);
        }}
        
        .submit-btn:active {{
            transform: translateY(0);
        }}
        
        .examples {{
            margin-top: 32px;
            padding: 24px;
            background: rgba(29, 185, 84, 0.05);
            border-radius: 16px;
            border: 1px solid rgba(29, 185, 84, 0.1);
        }}
        
        .examples-title {{
            font-weight: 600;
            margin-bottom: 16px;
            color: #1DB954;
            font-size: 1rem;
        }}
        
        .example-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        
        .example-tag {{
            background: rgba(255, 255, 255, 0.05);
            color: #b3b3b3;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .example-tag:hover {{
            background: rgba(29, 185, 84, 0.1);
            color: #1DB954;
            border-color: rgba(29, 185, 84, 0.2);
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 24px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .logout-link {{
            color: #b3b3b3;
            text-decoration: none;
            font-size: 0.9rem;
            transition: color 0.2s ease;
        }}
        
        .logout-link:hover {{
            color: #1DB954;
        }}

        /* ===== Loading Overlay (glassmorphism, Spotify 深色系) ===== */
        .loading-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0);
            backdrop-filter: blur(0px);
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            pointer-events: none;
            transition: opacity .35s ease, background .35s ease, backdrop-filter .35s ease;
            z-index: 9999;
        }}
        .loading-overlay.show {{
            opacity: 1;
            background: rgba(0,0,0,0.75);
            backdrop-filter: blur(6px);
            pointer-events: all;
        }}
        .loading-card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 14px;
            padding: 32px 28px;
            border-radius: 20px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
        }}
        .loading-logo {{
            width: 72px; height: 72px;
            border-radius: 18px;
            display: grid; place-items: center;
            background: radial-gradient(circle at 30% 30%, #1ed760, #1DB954 60%, #128a3e 100%);
            filter: drop-shadow(0 6px 24px rgba(29,185,84,.35));
        }}
        .loading-logo svg {{
            width: 38px; height: 38px;
            fill: #000;
        }}
        .loading-text {{
            color: #e8e8e8;
            font-weight: 700;
            letter-spacing: .2px;
        }}
        .loading-sub {{
            color: #b3b3b3;
            font-size: .92rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 20px 16px;
            }}
            .main-card {{
                padding: 24px;
            }}
            .logo {{
                font-size: 2.2rem;
            }}
            .welcome-text {{
                font-size: 1.1rem;
            }}
            .character-mascot {{
                width: 80px;
            }}
        }}
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
            <div class="character-container">
                <img src="/static/character.png" alt="Mooodyyy Character" class="character-mascot">
            </div>
            
            <h2 class="form-title">描述你的當下情境</h2>
            <p class="form-subtitle">告訴我你的心情、活動或想要的氛圍，我會為你推薦最適合的歌單</p>
            
            <form id="gen-form" action="/recommend" method="post">
                <div class="textarea-container">
                    <textarea 
                        name="text" 
                        placeholder="例如：深夜漫步思考人生、雨天在咖啡廳寫作、想念遠方的朋友、專心讀書需要輕音樂、週五晚上想放鬆..."
                        required
                    ></textarea>
                </div>
                
                <input type="hidden" name="preview" value="1">
                <button type="submit" class="submit-btn">
                    🎵 開始推薦音樂
                </button>
            </form>
            
            <div class="examples">
                <div class="examples-title">💡 靈感提示</div>
                <div class="example-tags">
                    <span class="example-tag" onclick="fillExample(this)">深夜散步</span>
                    <span class="example-tag" onclick="fillExample(this)">下雨天寫作</span>
                    <span class="example-tag" onclick="fillExample(this)">週末早晨</span>
                    <span class="example-tag" onclick="fillExample(this)">運動健身</span>
                    <span class="example-tag" onclick="fillExample(this)">放鬆冥想</span>
                    <span class="example-tag" onclick="fillExample(this)">專心讀書</span>
                    <span class="example-tag" onclick="fillExample(this)">思念某人</span>
                    <span class="example-tag" onclick="fillExample(this)">開車兜風</span>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <a href="/logout" class="logout-link">登出 Spotify</a>
        </div>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loading">
        <div class="loading-card">
            <div class="loading-logo" aria-hidden="true">
                <!-- Spotify glyph -->
                <svg viewBox="0 0 24 24" role="img" aria-label="Spotify">
                    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.84-.179-.84-.6 0-.359.24-.66.54-.78 4.56-1.021 8.52-.6 11.64.301.42.12.66.54.42.96zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.301.421-1.02.599-1.56.3z"/>
                </svg>
            </div>
            <div class="loading-text">🎧 為你量身打造歌單中...</div>
            <div class="loading-sub">Mooodyyy 正在理解你的情境與喜好</div>
        </div>
    </div>
    
    <script>
        function fillExample(element) {{
            const textarea = document.querySelector('textarea[name="text"]');
            textarea.value = element.textContent;
            textarea.focus();
        }}
        
        document.addEventListener('DOMContentLoaded', function() {{
            const textarea = document.querySelector('textarea');
            const submitBtn = document.querySelector('.submit-btn');
            const form = document.getElementById('gen-form');
            const loading = document.getElementById('loading');
            
            textarea.addEventListener('input', function() {{
                if (this.value.trim()) {{
                    submitBtn.style.opacity = '1';
                    submitBtn.style.pointerEvents = 'auto';
                }} else {{
                    submitBtn.style.opacity = '0.7';
                }}
            }});

            // 提交時顯示 Loading Overlay（不阻擋原本 form 提交）
            form.addEventListener('submit', function() {{
                loading.classList.add('show');
            }});
        }});
    </script>
</body>
</html>
'''
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    import traceback
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    # ========= 小工具（封裝在 route 內，方便一次貼上） =========
    _CJK_RE = re.compile(r'[\u4e00-\u9fff]')  # 中日韓漢字

    def _lang_hint_from_text(text: str):
        t = (text or "").lower()
        if "英文" in t or "english" in t or "eng only" in t or "english only" in t:
            return "en"
        if "中文" in t or "華語" in t or "國語" in t or "chinese" in t:
            return "zh"
        return None

    def _track_lang(tr: dict):
        name = str(tr.get("name") or "")
        artists = tr.get("artists") or []
        artist_names = []
        if isinstance(artists, list):
            for a in artists:
                artist_names.append(str(a.get("name") if isinstance(a, dict) else a))
        else:
            artist_names.append(str(artists))
        s = name + " " + " ".join(artist_names)
        return "zh" if _CJK_RE.search(s) else "en"

    def _lang_filter(cands: list, want: str, keep_if_none=False):
        if not want:
            return cands
        out = []
        for tr in cands:
            try:
                l = _track_lang(tr)
                if l == want or (keep_if_none and l is None):
                    out.append(tr)
            except Exception:
                if keep_if_none:
                    out.append(tr)
        return out

    def _weighted_pick(tracks, k=3):
        """根據 _score 權重抽樣，避免永遠同幾首（高分仍較易被選）"""
        bag = [t for t in tracks if isinstance(t.get("_score"), (int, float))]
        if not bag:
            return tracks[:k]
        weights = [max(1e-6, t["_score"]) ** 2 for t in bag]
        chosen, used = [], set()
        for _ in range(min(k, len(bag))):
            s = sum(weights)
            r, acc, idx = random.random() * s, 0.0, 0
            for i, w in enumerate(weights):
                acc += w
                if acc >= r:
                    idx = i
                    break
            if bag[idx].get("id") in used:
                for j in range(len(bag)):
                    if bag[j].get("id") not in used:
                        idx = j
                        break
            chosen.append(bag[idx]); used.add(bag[idx].get("id"))
            del bag[idx]; del weights[idx]
        return chosen

    def _first_artist_id(tr):
        a = tr.get("artists") or tr.get("artist") or []
        if isinstance(a, list) and a and isinstance(a[0], dict):
            return a[0].get("id")
        if isinstance(a, dict):
            return a.get("id")
        return None

    # ========= 取得情境文字 =========
    text = (request.form.get("text") or request.args.get("text") or "").strip()
    if not text:
        return redirect(url_for("welcome"))

    # ========= 情境歷史（跨多次避重） =========
    ctx_key = (" ".join(text.lower().split()))[:80]
    history = session.get("hist", {})              # { ctx_key: [track_id, ...] }
    recent_ids = history.get(ctx_key, [])[:40]     # 最近 4 批（最多 40 首）

    try:
        # === 1) 收集候選池（縮小以加速） ===
        params    = map_text_to_params(text)
        user_pool = collect_user_tracks(sp, max_n=100)
        ext_pool  = collect_external_tracks_by_category(sp, text, 150)

        # === 2) 特徵 & 語意分數 ===
        ids_for_feat = []
        for tr in (user_pool + ext_pool):
            tid = tr.get("id")
            if isinstance(tid, str) and len(tid) == 22:
                ids_for_feat.append(tid)
                if len(ids_for_feat) >= 250:  # 上限 250 → 更快
                    break
        feats   = audio_features_map(sp, ids_for_feat)
        sem_map = build_semantic_map(text, user_pool + ext_pool, feats)

        # === 3) 排序（top_n 放寬，給抽樣空間） ===
        user_candidates = rank_pool_by_semantic_and_features(user_pool, feats, sem_map, params, top_n=20)
        ext_candidates  = rank_pool_by_semantic_and_features(ext_pool,  feats, sem_map, params, top_n=200)

        # 輕度打散
        random.shuffle(user_candidates)
        random.shuffle(ext_candidates)

        # === 3.5) 語言偏好（英文/中文） ===
        want_lang = _lang_hint_from_text(text)  # 'en' / 'zh' / None
        if want_lang in ("en", "zh"):
            user_candidates = _lang_filter(user_candidates, want_lang)
            ext_candidates  = _lang_filter(ext_candidates,  want_lang)

        # 讀取上一批要避開（由預覽頁傳回）
        avoid_raw = (request.form.get("avoid") or request.args.get("avoid") or "").strip()
        avoid_ids = set(i for i in avoid_raw.split(",") if len(i) == 22) if avoid_raw else set()

        # 你的曲庫所有 id（避免外部重複你的曲庫曲目）—— 修正為正確的 set comprehension
        user_all_ids = {
            t.get("id") for t in user_pool
            if isinstance(t.get("id"), str) and len(t.get("id")) == 22
        }

        # === 4) 混合：3 首你的曲庫 + 7 首外部（避開 avoid_ids + recent_ids） ===
        used = set(avoid_ids) | set(recent_ids)

        # 4a) 曲庫錨點（加權抽樣取 3）
        anchors_pool = [t for t in user_candidates if isinstance(t.get("id"), str) and t["id"] not in used][:20]
        anchors = _weighted_pick(anchors_pool, k=3)
        for tr in anchors:
            if isinstance(tr.get("id"), str):
                used.add(tr["id"])

        # 4b) 外部 7 首（避免同藝人洗版、避開 used 與你的曲庫）
        ext_chosen, seen_artists = [], set()
        for tr in ext_candidates:
            if len(ext_chosen) >= 7:
                break
            tid = tr.get("id")
            if not (isinstance(tid, str) and len(tid) == 22):
                continue
            if tid in used or tid in user_all_ids:
                continue
            aid = _first_artist_id(tr)
            if aid and aid in seen_artists:
                continue
            seen_artists.add(aid)
            ext_chosen.append(tr); used.add(tid)

        # 不足就放寬補滿到 7
        if len(ext_chosen) < 7:
            for tr in ext_candidates:
                if len(ext_chosen) >= 7:
                    break
                tid = tr.get("id")
                if not (isinstance(tid, str) and len(tid) == 22) or tid in used:
                    continue
                ext_chosen.append(tr); used.add(tid)

        top10 = (anchors + ext_chosen)[:10]

        # === 5) 預覽頁（精簡顯示、無「匹配度」） ===
        preview = (request.values.get("preview") or "").strip()
        if preview == "1":
            # 清單：只顯示「歌手 — 歌名」
            items = []
            for i, tr in enumerate(top10, 1):
                name = tr.get("name", "")
                arts = tr.get("artists", [])
                if isinstance(arts, list) and arts and isinstance(arts[0], dict):
                    artists = ", ".join(a.get("name", "") for a in arts)
                elif isinstance(arts, list):
                    artists = ", ".join(str(a) for a in arts)
                else:
                    artists = str(arts) if arts else ""
                url = (tr.get("external_urls") or {}).get("spotify") or tr.get("url") or "#"

                items.append(f'''
                <div class="track-item">
                    <div class="track-number">{i:02d}</div>
                    <div class="track-info">
                        <div class="track-name">{name}</div>
                        <div class="track-artist">{artists}</div>
                    </div>
                    <div class="track-actions">
                        <a href="{url}" target="_blank" class="spotify-link" title="在 Spotify 開啟">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="#1DB954" aria-hidden="true">
                                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.84-.179-.84-.6 0-.359.24-.66.54-.78 4.56-1.021 8.52-.6 11.64.301.42.12.66.54.42.96zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.301.421-1.02.599-1.56.3z"/>
                            </svg>
                        </a>
                    </div>
                </div>
                ''')

            songs_html = "\n".join(items)

            # 當前這批歌曲 IDs
            ids_str   = ",".join([t.get("id") for t in top10 if isinstance(t.get("id"), str) and len(t.get("id")) == 22])
            safe_text = text.replace("'", "&#39;").replace('"', '&quot;')

            # 把這批寫回情境歷史（新在前、去重保序、最多 40）
            cur_ids = [t.get("id") for t in top10 if isinstance(t.get("id"), str) and len(t.get("id")) == 22]
            old_ids = [x for x in recent_ids if x not in cur_ids]
            history[ctx_key] = (cur_ids + old_ids)[:40]
            session["hist"] = history

            page = f'''
            <!doctype html>
            <html lang="zh-Hant">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width,initial-scale=1">
                <title>為你推薦的歌單 - Mooodyyy</title>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{
                        font-family: 'Circular', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
                        background: linear-gradient(135deg, #191414 0%, #0d1117 50%, #121212 100%);
                        color: #ffffff;
                        min-height: 100vh;
                        line-height: 1.6;
                    }}
                    .container {{ max-width: 900px; margin: 0 auto; padding: 40px 20px; }}
                    .header {{ text-align: center; margin-bottom: 40px; }}
                    .logo {{ font-size: 2rem; font-weight: 900; background: linear-gradient(135deg, #1DB954, #1ed760);
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 16px; }}
                    .result-title {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 12px; color: #ffffff; }}
                    .context-display {{ background: rgba(29, 185, 84, 0.1); border: 1px solid rgba(29, 185, 84, 0.2); border-radius: 16px;
                                        padding: 20px; margin-bottom: 32px; text-align: center; }}
                    .context-label {{ color: #1DB954; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
                    .context-text {{ font-size: 1.1rem; color: #ffffff; font-style: italic; }}
                    .playlist-container {{ background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 24px;
                                           padding: 32px; backdrop-filter: blur(20px); margin-bottom: 32px; position: relative; overflow: hidden; }}
                    .playlist-container::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
                                                   background: linear-gradient(90deg, transparent, #1DB954, transparent); }}
                    .playlist-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding-bottom: 16px;
                                        border-bottom: 1px solid rgba(255, 255, 255, 0.05); }}
                    .playlist-icon {{ width: 48px; height: 48px; background: linear-gradient(135deg, #1DB954, #1ed760); border-radius: 12px;
                                      display: flex; align-items: center; justify-content: center; font-size: 1.5rem; }}
                    .playlist-info h3 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }}
                    .playlist-info p {{ color: #b3b3b3; font-size: 0.95rem; }}
                    .tracks-list {{ display: flex; flex-direction: column; gap: 8px; }}
                    .track-item {{ display: flex; align-items: center; gap: 16px; padding: 12px 16px; border-radius: 12px;
                                   background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); transition: all 0.2s ease;
                                   position: relative; overflow: hidden; }}
                    .track-item:hover {{ background: rgba(29, 185, 84, 0.05); border-color: rgba(29, 185, 84, 0.1); transform: translateX(4px); }}
                    .track-number {{ font-size: 0.9rem; color: #757575; font-weight: 600; width: 24px; text-align: center; }}
                    .track-info {{ flex: 1; min-width: 0; }}
                    .track-name {{ font-weight: 600; font-size: 1rem; color: #ffffff; margin-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
                    .track-artist {{ color: #b3b3b3; font-size: 0.9rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
                    .track-actions {{ display: flex; align-items: center; gap: 12px; }}
                    .spotify-link {{ display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 50%;
                                     background: rgba(29, 185, 84, 0.1); border: 1px solid rgba(29, 185, 84, 0.2); transition: all 0.2s ease; text-decoration: none; }}
                    .spotify-link:hover {{ background: #1DB954; transform: scale(1.1); }}
                    .spotify-link:hover svg {{ fill: #000000; }}
                    .actions {{ display: flex; gap: 16px; margin-top: 32px; justify-content: center; flex-wrap: wrap; }}
                    .btn {{ padding: 14px 28px; border: none; border-radius: 50px; cursor: pointer; font-weight: 700; font-size: 1rem;
                            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94); text-decoration: none; display: inline-flex; align-items: center; gap: 8px; }}
                    .btn-primary {{ background: linear-gradient(135deg, #1DB954, #1ed760); color: #000000; box-shadow: 0 6px 24px rgba(29, 185, 84, 0.25); }}
                    .btn-primary:hover {{ transform: translateY(-2px); box-shadow: 0 8px 32px rgba(29, 185, 84, 0.35); }}
                    .btn-secondary {{ background: rgba(255, 255, 255, 0.05); color: #ffffff; border: 1px solid rgba(255, 255, 255, 0.1); }}
                    .btn-secondary:hover {{ background: rgba(255, 255, 255, 0.1); transform: translateY(-1px); }}
                    .back-link {{ text-align: center; margin-top: 32px; }}
                    .back-link a {{ color: #b3b3b3; text-decoration: none; font-size: 0.95rem; transition: color 0.2s ease; }}
                    .back-link a:hover {{ color: #1DB954; }}
                    @media (max-width: 768px) {{
                        .container {{ padding: 20px 16px; }}
                        .playlist-container {{ padding: 20px; }}
                        .track-item {{ padding: 12px; gap: 12px; }}
                        .actions {{ flex-direction: column; }}
                        .btn {{ width: 100%; justify-content: center; }}
                        .track-actions {{ flex-direction: column; gap: 8px; align-items: flex-end; }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1 class="logo">Mooodyyy</h1>
                        <h2 class="result-title">🎯 為你找到了 {len(top10)} 首歌</h2>
                    </div>
                    <div class="context-display">
                        <div class="context-label">你的情境</div>
                        <div class="context-text">"{safe_text}"</div>
                    </div>
                    <div class="playlist-container">
                        <div class="playlist-header">
                            <div class="playlist-icon">🎵</div>
                            <div class="playlist-info">
                                <h3>專屬推薦歌單</h3>
                                <p>基於你的聆聽習慣與情境分析</p>
                            </div>
                        </div>
                        <div class="tracks-list">
                            {songs_html}
                        </div>
                    </div>
                    <div class="actions">
                        <form method="POST" action="/recommend" style="display:inline;">
                            <input type="hidden" name="text" value="{safe_text}">
                            <input type="hidden" name="preview" value="1">
                            <input type="hidden" name="avoid" value="{ids_str}">
                            <button type="submit" class="btn btn-secondary">🔄 重新生成</button>
                        </form>
                        <form method="POST" action="/create_playlist" style="display:inline;">
                            <input type="hidden" name="text" value="{safe_text}">
                            <input type="hidden" name="track_ids" value="{ids_str}">
                            <button type="submit" class="btn btn-primary">➕ 存到 Spotify</button>
                        </form>
                    </div>
                    <div class="back-link">
                        <a href="/welcome">↩︎ 回到首頁</a>
                    </div>
                </div>
                <script>
                    document.addEventListener('DOMContentLoaded', function() {{
                        const tracks = document.querySelectorAll('.track-item');
                        tracks.forEach((track, index) => {{
                            track.style.opacity = '0';
                            track.style.transform = 'translateY(20px)';
                            setTimeout(() => {{
                                track.style.transition = 'all 0.5s ease';
                                track.style.opacity = '1';
                                track.style.transform = 'translateY(0)';
                            }}, index * 100);
                        }});
                    }});
                </script>
            </body>
            </html>
            '''
            return page

        # === 非預覽：直接建私人歌單（向後相容） ===
        user   = sp.current_user(); user_id = (user or {}).get("id")
        ts     = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        title  = f"Mooodyyy · {ts} UTC"
        desc   = f"情境：{text}（由即時推薦建立）"
        plist  = sp.user_playlist_create(user=user_id, name=title, public=False, description=desc)
        sp.playlist_add_items(playlist_id=plist["id"], items=[t["id"] for t in top10 if t.get("id")])
        url    = (plist.get("external_urls") or {}).get("spotify", url_for("welcome"))
        return redirect(url)

    except Exception as e:
        # 讓錯誤真的可見，並回 500
        print("❌ recommend error:", e)
        print(traceback.format_exc())
        return '''
<!doctype html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>系統錯誤 - Mooodyyy</title>
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
        <h2>😵 系統出錯</h2>
        <p>我們已記錄錯誤，請稍後重試</p>
        <a href="/welcome" class="retry-btn">回首頁</a>
    </div>
</body>
</html>
'''

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
