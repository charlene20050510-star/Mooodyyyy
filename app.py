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
        
        .form-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: #ffffff;
        }}
        
        .form-subtitle {{
            color: #b3b3b3;
            margin-bottom: 32px;
            font-size: 1rem;
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

    text = (request.form.get("text") or request.args.get("text") or "").strip()
    if not text:
        return redirect(url_for("welcome"))

    ctx_key = (" ".join(text.lower().split()))[:80]
    history = session.get("hist", {})
    recent_ids = history.get(ctx_key, [])[:40]

    try:
        # [修改點5] 使用新的推薦邏輯
        avoid_raw = (request.form.get("avoid") or request.args.get("avoid") or "").strip()
        avoid_ids = set(i for i in avoid_raw.split(",") if len(i) == 22) if avoid_raw else set()
        
        top10, warnings = generate_playlist_with_language_priority(sp, text, recent_ids, avoid_ids)
        
        # 預覽頁面
        preview = (request.values.get("preview") or "").strip()
        if preview == "1":
            # [修改點4] 移除所有預覽相關功能，簡化前端
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
                track_id = tr.get("id", f"sample{i}")

                items.append(f'''
                <div class="track-item" draggable="true" data-track-id="{track_id}">
                    <div class="drag-handle">⋮⋮</div>
                    <div class="track-number">{i:02d}</div>
                    <div class="track-info">
                        <div class="track-name">{name}</div>
                        <div class="track-artist">{artists}</div>
                    </div>
                    <div class="track-actions">
                        <button class="action-btn" onclick="copySpotifyLink('{url}')" title="複製連結">
                            📋
                        </button>
                        <a href="{url}" target="_blank" class="action-btn spotify-link" title="在 Spotify 開啟">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="#1DB954" aria-hidden="true">
                                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.84-.179-.84-.6 0-.359.24-.66.54-.78 4.56-1.021 8.52-.6 11.64.301.42.12.66.54.42.96zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.301.421-1.02.599-1.56.3z"/>
                            </svg>
                        </a>
                    </div>
                </div>
                ''')

            songs_html = "\n".join(items)
            ids_str = ",".join([t.get("id") for t in top10 if isinstance(t.get("id"), str) and len(t.get("id")) == 22])
            safe_text = text.replace("'", "&#39;").replace('"', '&quot;')

            # 更新歷史記錄
            cur_ids = [t.get("id") for t in top10 if isinstance(t.get("id"), str) and len(t.get("id")) == 22]
            old_ids = [x for x in recent_ids if x not in cur_ids]
            history[ctx_key] = (cur_ids + old_ids)[:40]
            session["hist"] = history

            # 警告訊息處理
            warning_html = ""
            if warnings:
                warning_html = f'''
                <div class="warning-box">
                    <div class="warning-icon">⚠️</div>
                    <div class="warning-text">
                        {"<br>".join(warnings)}
                    </div>
                </div>
                '''

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
                    .tracks-list {{ display: flex; flex-direction: column; gap: 8px; min-height: 400px; }}
                    .track-item {{ 
                        display: flex; align-items: center; gap: 16px; padding: 12px 16px; border-radius: 12px;
                        background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); 
                        transition: all 0.2s ease; position: relative; overflow: hidden; cursor: grab;
                    }}
                    .track-item:hover {{ background: rgba(29, 185, 84, 0.05); border-color: rgba(29, 185, 84, 0.1); }}
                    .track-item.dragging {{ 
                        opacity: 0.5; cursor: grabbing; z-index: 1000; 
                        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                    }}
                    .track-item.drag-over {{ 
                        border-color: #1DB954; 
                        box-shadow: 0 0 20px rgba(29, 185, 84, 0.3);
                    }}
                    .drag-handle {{ 
                        color: #757575; cursor: grab; padding: 4px; 
                        transition: color 0.2s ease;
                    }}
                    .drag-handle:hover {{ color: #1DB954; }}
                    .track-number {{ font-size: 0.9rem; color: #757575; font-weight: 600; width: 24px; text-align: center; }}
                    .track-info {{ flex: 1; min-width: 0; }}
                    .track-name {{ font-weight: 600; font-size: 1rem; color: #ffffff; margin-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
                    .track-artist {{ color: #b3b3b3; font-size: 0.9rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
                    .track-actions {{ display: flex; align-items: center; gap: 8px; }}
                    .action-btn {{ 
                        display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; 
                        border-radius: 50%; border: 1px solid rgba(255, 255, 255, 0.1); 
                        background: rgba(255, 255, 255, 0.02); cursor: pointer; transition: all 0.2s ease; 
                        color: #b3b3b3; text-decoration: none;
                    }}
                    .action-btn:hover {{ 
                        background: rgba(29, 185, 84, 0.1); 
                        border-color: rgba(29, 185, 84, 0.2); 
                        color: #1DB954; 
                        transform: scale(1.1);
                    }}
                    .spotify-link {{ background: rgba(29, 185, 84, 0.1); border-color: rgba(29, 185, 84, 0.2); }}
                    .spotify-link:hover {{ background: #1DB954; }}
                    .spotify-link:hover svg {{ fill: #000000; }}
                    
                    .warning-box {{
                        background: rgba(255, 193, 7, 0.1);
                        border: 1px solid rgba(255, 193, 7, 0.3);
                        border-radius: 12px;
                        padding: 16px 20px;
                        margin-bottom: 24px;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    }}
                    .warning-icon {{ font-size: 1.2rem; }}
                    .warning-text {{ color: #ffc107; font-size: 0.95rem; }}
                    
                    .actions {{ display: flex; gap: 16px; margin-top: 32px; justify-content: center; flex-wrap: wrap; }}
                    .btn {{ padding: 14px 28px; border: none; border-radius: 50px; cursor: pointer; font-weight: 700; font-size: 1rem;
                            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94); text-decoration: none; display: inline-flex; align-items: center; gap: 8px; }}
                    .btn:disabled {{ opacity: 0.5; cursor: not-allowed; transform: none !important; }}
                    .btn-primary {{ background: linear-gradient(135deg, #1DB954, #1ed760); color: #000000; box-shadow: 0 6px 24px rgba(29, 185, 84, 0.25); }}
                    .btn-primary:hover:not(:disabled) {{ transform: translateY(-2px); box-shadow: 0 8px 32px rgba(29, 185, 84, 0.35); }}
                    .btn-secondary {{ background: rgba(255, 255, 255, 0.05); color: #ffffff; border: 1px solid rgba(255, 255, 255, 0.1); }}
                    .btn-secondary:hover:not(:disabled) {{ background: rgba(255, 255, 255, 0.1); transform: translateY(-1px); }}
                    .back-link {{ text-align: center; margin-top: 32px; }}
                    .back-link a {{ color: #b3b3b3; text-decoration: none; font-size: 0.95rem; transition: color 0.2s ease; }}
                    .back-link a:hover {{ color: #1DB954; }}

                    .toast {{
                        position: fixed; top: 20px; right: 20px; background: rgba(29, 185, 84, 0.9);
                        color: #000; padding: 12px 20px; border-radius: 8px; font-weight: 600;
                        transform: translateX(100%); transition: transform 0.3s ease; z-index: 10001;
                        box-shadow: 0 4px 20px rgba(29, 185, 84, 0.3);
                    }}
                    .toast.show {{ transform: translateX(0); }}

                    .loading-overlay {{
                        position: fixed; inset: 0; background: rgba(0,0,0,0); backdrop-filter: blur(0px);
                        display: flex; align-items: center; justify-content: center; opacity: 0;
                        pointer-events: none; transition: opacity .35s ease, background .35s ease, backdrop-filter .35s ease;
                        z-index: 9999;
                    }}
                    .loading-overlay.show {{
                        opacity: 1; background: rgba(0,0,0,0.75); backdrop-filter: blur(6px);
                        pointer-events: all;
                    }}
                    .loading-card {{
                        display: flex; flex-direction: column; align-items: center; gap: 20px;
                        padding: 40px 32px; border-radius: 20px; background: rgba(255,255,255,0.06);
                        border: 1px solid rgba(255,255,255,0.12); box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
                        min-width: 300px; text-align: center;
                    }}
                    .loading-logo {{
                        width: 80px; height: 80px; border-radius: 20px; display: grid; place-items: center;
                        background: radial-gradient(circle at 30% 30%, #1ed760, #1DB954 60%, #128a3e 100%);
                        filter: drop-shadow(0 6px 24px rgba(29,185,84,.35)); animation: spin 2s linear infinite;
                    }}
                    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
                    .loading-logo svg {{ width: 42px; height: 42px; fill: #000; }}
                    .loading-text {{ color: #e8e8e8; font-weight: 700; letter-spacing: .2px; font-size: 1.1rem; }}
                    .loading-sub {{ color: #b3b3b3; font-size: .92rem; }}

                    @media (max-width: 768px) {{
                        .container {{ padding: 20px 16px; }}
                        .playlist-container {{ padding: 20px; }}
                        .track-item {{ padding: 12px; gap: 12px; }}
                        .actions {{ flex-direction: column; }}
                        .btn {{ width: 100%; justify-content: center; }}
                        .track-actions {{ gap: 6px; }}
                        .toast {{ right: 10px; left: 10px; transform: translateY(-100%); }}
                        .toast.show {{ transform: translateY(0); }}
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
                    {warning_html}
                    <div class="playlist-container">
                        <div class="playlist-header">
                            <div class="playlist-icon">🎵</div>
                            <div class="playlist-info">
                                <h3>專屬推薦歌單</h3>
                                <p>基於你的聆聽習慣與情境分析 • 可拖曳調整順序</p>
                            </div>
                        </div>
                        <div class="tracks-list" id="tracks-list">
                            {songs_html}
                        </div>
                    </div>
                    <div class="actions">
                        <form method="POST" action="/recommend" id="regen-form">
                            <input type="hidden" name="text" value="{safe_text}">
                            <input type="hidden" name="preview" value="1">
                            <input type="hidden" name="avoid" value="{ids_str}">
                            <button type="submit" class="btn btn-secondary" id="regen-btn">🔄 重新生成</button>
                        </form>
                        <form method="POST" action="/create_playlist" id="save-form">
                            <input type="hidden" name="text" value="{safe_text}">
                            <input type="hidden" name="track_ids" value="{ids_str}">
                            <button type="submit" class="btn btn-primary" id="save-btn">➕ 存到 Spotify</button>
                        </form>
                    </div>
                    <div class="back-link">
                        <a href="/welcome">↩️ 回到首頁</a>
                    </div>
                </div>

                <div class="toast" id="toast"></div>

                <div class="loading-overlay" id="loading">
                    <div class="loading-card">
                        <div class="loading-logo">
                            <svg viewBox="0 0 24 24" role="img" aria-label="Spotify">
                                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.84-.179-.84-.6 0-.359.24-.66.54-.78 4.56-1.021 8.52-.6 11.64.301.42.12.66.54.42.96zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.301.421-1.02.599-1.56.3z"/>
                            </svg>
                        </div>
                        <div class="loading-text">🎵 正在保存到 Spotify...</div>
                        <div class="loading-sub">為你創建專屬歌單</div>
                    </div>
                </div>
                
                <script>
                    // [修改點4] 簡化 JavaScript - 移除所有預覽相關功能
                    let draggedElement = null;
                    let draggedIndex = -1;

                    function initDragAndDrop() {{
                        const tracksList = document.getElementById('tracks-list');
                        const tracks = tracksList.querySelectorAll('.track-item');

                        tracks.forEach((track, index) => {{
                            track.addEventListener('dragstart', function(e) {{
                                draggedElement = this;
                                draggedIndex = index;
                                this.classList.add('dragging');
                                e.dataTransfer.effectAllowed = 'move';
                            }});

                            track.addEventListener('dragend', function() {{
                                this.classList.remove('dragging');
                                draggedElement = null;
                                draggedIndex = -1;
                                updateTrackNumbers();
                            }});

                            track.addEventListener('dragover', function(e) {{
                                e.preventDefault();
                                e.dataTransfer.dropEffect = 'move';
                            }});

                            track.addEventListener('dragenter', function(e) {{
                                e.preventDefault();
                                if (this !== draggedElement) {{
                                    this.classList.add('drag-over');
                                }}
                            }});

                            track.addEventListener('dragleave', function() {{
                                this.classList.remove('drag-over');
                            }});

                            track.addEventListener('drop', function(e) {{
                                e.preventDefault();
                                this.classList.remove('drag-over');
                                
                                if (draggedElement && this !== draggedElement) {{
                                    const allTracks = Array.from(tracksList.children);
                                    const currentIndex = allTracks.indexOf(this);
                                    
                                    if (currentIndex > draggedIndex) {{
                                        this.parentNode.insertBefore(draggedElement, this.nextSibling);
                                    }} else {{
                                        this.parentNode.insertBefore(draggedElement, this);
                                    }}
                                    updateTrackNumbers();
                                    updateFormData();
                                }}
                            }});
                        }});
                    }}

                    function updateTrackNumbers() {{
                        const tracks = document.querySelectorAll('.track-item');
                        tracks.forEach((track, index) => {{
                            const numberElement = track.querySelector('.track-number');
                            numberElement.textContent = String(index + 1).padStart(2, '0');
                        }});
                    }}

                    function updateFormData() {{
                        const tracks = document.querySelectorAll('.track-item');
                        const trackIds = Array.from(tracks).map(track => track.getAttribute('data-track-id'));
                        
                        document.querySelector('#save-form input[name="track_ids"]').value = trackIds.join(',');
                        document.querySelector('#regen-form input[name="avoid"]').value = trackIds.join(',');
                    }}

                    function copySpotifyLink(url) {{
                        if (navigator.clipboard) {{
                            navigator.clipboard.writeText(url).then(() => {{
                                showToast('✅ Spotify 連結已複製到剪貼簿');
                            }}).catch(() => {{
                                fallbackCopyText(url);
                            }});
                        }} else {{
                            fallbackCopyText(url);
                        }}
                    }}

                    function fallbackCopyText(text) {{
                        const textArea = document.createElement('textarea');
                        textArea.value = text;
                        textArea.style.position = 'fixed';
                        textArea.style.left = '-999999px';
                        textArea.style.top = '-999999px';
                        document.body.appendChild(textArea);
                        textArea.focus();
                        textArea.select();
                        
                        try {{
                            document.execCommand('copy');
                            showToast('✅ Spotify 連結已複製到剪貼簿');
                        }} catch (err) {{
                            showToast('❌ 複製失敗，請手動複製連結');
                        }}
                        
                        document.body.removeChild(textArea);
                    }}

                    function showToast(message) {{
                        const toast = document.getElementById('toast');
                        toast.textContent = message;
                        toast.classList.add('show');
                        
                        setTimeout(() => {{
                            toast.classList.remove('show');
                        }}, 3000);
                    }}

                    function initFormHandling() {{
                        const saveForm = document.getElementById('save-form');
                        const regenForm = document.getElementById('regen-form');
                        const loading = document.getElementById('loading');
                        const saveBtn = document.getElementById('save-btn');
                        const regenBtn = document.getElementById('regen-btn');

                        saveForm.addEventListener('submit', function(e) {{
                            loading.classList.add('show');
                            saveBtn.disabled = true;
                            regenBtn.disabled = true;
                            updateFormData();
                        }});

                        regenForm.addEventListener('submit', function(e) {{
                            regenBtn.disabled = true;
                            saveBtn.disabled = true;
                            updateFormData();
                        }});
                    }}

                    document.addEventListener('DOMContentLoaded', function() {{
                        initDragAndDrop();
                        initFormHandling();
                        updateFormData();
                        
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

                    window.addEventListener('pageshow', function(event) {{
                        const loading = document.getElementById('loading');
                        const saveBtn = document.getElementById('save-btn');
                        const regenBtn = document.getElementById('regen-btn');
                        
                        if (event.persisted) {{
                            loading.classList.remove('show');
                            saveBtn.disabled = false;
                            regenBtn.disabled = false;
                        }}
                    }});
                </script>
            </body>
            </html>
            '''
            return page

        # 非預覽：直接建立歌單（向後相容）
        user = sp.current_user(); user_id = (user or {}).get("id")
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        title = f"Mooodyyy · {ts} UTC"
        desc = f"情境：{text}（由即時推薦建立）"
        plist = sp.user_playlist_create(user=user_id, name=title, public=False, description=desc)
        sp.playlist_add_items(playlist_id=plist["id"], items=[t["id"] for t in top10 if t.get("id")])
        url = (plist.get("external_urls") or {}).get("spotify", url_for("welcome"))
        return redirect(url)

    except Exception as e:
        print("⚠ recommend error:", e)
        print(traceback.format_exc())
        return '''
<!doctype html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>系統錯誤 - Mooodyyy</title>
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
        <h2>😵 系統出錯</h2>
        <p>我們已記錄錯誤，請稍後重試</p>
        <a href="/welcome" class="retry-btn">回首頁</a>
    </div>
</body>
</html>
''', 500

@app.route("/create_playlist", methods=["POST"])
def create_playlist():
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    text = (request.form.get("text") or "").strip()
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
        user = sp.current_user()
        user_id = (user or {}).get("id")
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        title = f"Mooodyyy · {ts} UTC"
        desc = f"情境：{text}（由預覽頁直接保存）"

        playlist = sp.user_playlist_create(
            user=user_id,
            name=title,
            public=False,
            description=desc
        )
        sp.playlist_add_items(playlist_id=playlist["id"], items=ids)

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
    app.run(host="0.0.0.0", port=PORT)@app.route("/welcome")
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
        
        .form-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: #ffffff;
        }}
        
        .form-subtitle {{
            color: #b3b3b3;
            margin-bottom: 32px;
            font-size: 1rem;
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
            <h2 class="form-title">描述你的當下情境</h2>
            <p class="form-subtitle">告訴我你的心情、活動或想要的氛圍，我會為你推薦最適合的歌單</p>
            
            <form id="gen-form" action="/recommend" method="post">
                <div class="textarea-container">
                    <textarea 
                        name="text" 
                        placeholder="例如：中文抒情歌、韓文Kpop、日文搖滾、英文饒舌、深夜漫步思考人生、雨天在咖啡廳寫作、想念遠方的朋友、專心讀書需要輕音樂..."
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
                    <span class="example-tag" onclick="fillExample(this)">中文抒情歌</span>
                    <span class="example-tag" onclick="fillExample(this)">韓文Kpop</span>
                    <span class="example-tag" onclick="fillExample(this)">日文搖滾</span>
                    <span class="example-tag" onclick="fillExample(this)">英文饒舌</span>
                    <span class="example-tag" onclick="fillExample(this)">深夜散步</span>
                    <span class="example-tag" onclick="fillExample(this)">運動健身</span>
                    <span class="example-tag" onclick="fillExample(this)">放鬆冥想</span>
                    <span class="example-tag" onclick="fillExample(this)">專心讀書</span>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <a href="/logout" class="logout-link">登出 Spotify</a>
        </div>
    </div>

    <div class="loading-overlay" id="loading">
        <div class="loading-card">
            <div class="loading-logo" aria-hidden="true">
                <svg viewBox="0 0 24 24" role="img" aria-label="Spotify">
                    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.84-.179-.84-.6 0-.359.24-.66.54-.78 4.56-1.021 8.52-.6 11.64.301.42.12.66.54.42.96zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.from flask import Flask, request, redirect, session, url_for
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

# ========= [修改點1: 擴充語言支援] =========
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')  # 中文漢字
_HANGUL_RE = re.compile(r'[\uac00-\ud7af]')  # 韓文
_HIRAGANA_KATAKANA_RE = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')  # 日文平假名片假名

def _lang_hint_from_text(text: str):
    """從使用者輸入判斷語言偏好"""
    t = (text or "").lower()
    # 韓文優先（因為kpop包含英文字母，需要先判斷）
    if any(k in t for k in ["韓文", "韓語", "kpop", "k-pop", "korean"]):
        return "ko"
    # 日文
    if any(k in t for k in ["日文", "日本", "jpop", "j-pop", "japanese"]):
        return "ja"
    # 中文
    if any(k in t for k in ["中文", "國語", "華語", "chinese", "mandarin"]):
        return "zh"
    # 英文
    if any(k in t for k in ["英文", "english", "eng only", "english only"]):
        return "en"
    return None

def _track_lang(tr: dict):
    """判斷歌曲語言"""
    name = str(tr.get("name") or "")
    artists = tr.get("artists") or []
    artist_names = []
    if isinstance(artists, list):
        for a in artists:
            artist_names.append(str(a.get("name") if isinstance(a, dict) else a))
    else:
        artist_names.append(str(artists))
    
    text = name + " " + " ".join(artist_names)
    
    # 按優先度判斷
    if _HANGUL_RE.search(text):
        return "ko"
    if _HIRAGANA_KATAKANA_RE.search(text):
        return "ja"
    if _CJK_RE.search(text):
        return "zh"
    return "en"

def _lang_filter(cands: list, want: str, keep_if_none=False):
    """語言過濾器"""
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

# ========= [修改點2+3: 擴充風格映射與外部來源] =========
STYLE_MAP = {
    # 原有風格
    "lofi":       {"energy": (0.0, 0.4), "acousticness": (0.6, 1.0), "tempo": (60, 90)},
    "jazz":       {"energy": (0.2, 0.6), "instrumentalness": (0.4, 1.0)},
    "edm":        {"energy": (0.7, 1.0), "danceability": (0.7, 1.0), "tempo": (120, 150)},
    "rock":       {"energy": (0.6, 1.0), "instrumentalness": (0.0, 0.5)},
    "classical":  {"energy": (0.0, 0.3), "acousticness": (0.7, 1.0), "instrumentalness": (0.7, 1.0)},
    "hip hop":    {"energy": (0.6, 0.9), "speechiness": (0.5, 1.0), "tempo": (80, 110)},
    "r&b":        {"energy": (0.3, 0.7), "danceability": (0.5, 0.9)},
    "rb":         {"energy": (0.3, 0.7), "danceability": (0.5, 0.9)},
    
    # 新增風格
    "rap":        {"energy": (0.6, 0.9), "speechiness": (0.6, 1.0), "danceability": (0.5, 0.8)},
    "hiphop":     {"energy": (0.6, 0.9), "speechiness": (0.4, 1.0), "danceability": (0.6, 0.9)},
    "kpop":       {"energy": (0.5, 0.9), "danceability": (0.6, 1.0), "valence": (0.4, 0.8)},
    "jpop":       {"energy": (0.4, 0.8), "danceability": (0.4, 0.8), "valence": (0.3, 0.7)},
    "band":       {"energy": (0.5, 0.9), "instrumentalness": (0.0, 0.6)},
    "metal":      {"energy": (0.8, 1.0), "loudness": (-8, 0)},
}

# 各語言專屬歌單搜尋關鍵字
LANG_PLAYLISTS = {
    "zh": {
        "party":   ["華語流行", "中文舞曲", "國語熱門", "華語派對"],
        "sad":     ["華語抒情", "國語情歌", "失戀歌曲", "中文療傷"],
        "chill":   ["華語輕音樂", "中文放鬆", "國語慢歌", "華語咖啡"],
        "focus":   ["華語輕音樂", "中文器樂", "國語純音樂"],
        "rap":     ["中文饒舌", "華語嘻哈", "國語rap"],
        "rock":    ["華語搖滾", "中文樂團", "國語搖滾"],
        "r&b":     ["華語R&B", "中文節奏藍調"],
        "default": ["華語流行", "國語新歌", "中文熱門"],
    },
    "ko": {
        "party":   ["kpop", "korean dance", "k-pop hits", "korean party"],
        "sad":     ["korean ballad", "k-pop sad", "korean emotional"],
        "chill":   ["korean chill", "k-pop acoustic", "korean indie"],
        "focus":   ["korean instrumental", "k-pop study"],
        "rap":     ["korean rap", "k-hip hop", "korean hiphop"],
        "rock":    ["korean rock", "k-rock", "korean band"],
        "r&b":     ["korean r&b", "k-r&b"],
        "default": ["kpop", "korean pop", "k-pop hits"],
    },
    "ja": {
        "party":   ["jpop", "japanese pop", "j-pop hits", "japanese dance"],
        "sad":     ["japanese ballad", "j-pop sad", "japanese emotional"],
        "chill":   ["japanese chill", "j-pop acoustic", "japanese indie"],
        "focus":   ["japanese instrumental", "j-pop study"],
        "rap":     ["japanese rap", "j-hip hop", "japanese hiphop"],
        "rock":    ["japanese rock", "j-rock", "japanese band"],
        "r&b":     ["japanese r&b", "j-r&b"],
        "default": ["jpop", "japanese pop", "j-pop hits"],
    },
    "en": {
        "party":   ["party", "dance hits", "pop party", "edm"],
        "sad":     ["sad songs", "heartbreak", "emotional", "mellow"],
        "chill":   ["chill", "indie", "acoustic", "relaxing"],
        "focus":   ["focus", "study music", "instrumental", "ambient"],
        "rap":     ["rap", "hip hop", "hiphop"],
        "rock":    ["rock", "alternative", "indie rock"],
        "r&b":     ["r&b", "soul", "rnb"],
        "default": ["pop hits", "top charts", "trending"],
    }
}

def map_text_to_params(text: str) -> dict:
    """擴充版：支援更多風格關鍵詞"""
    t = (text or "").lower()
    params = {}

    # 風格詞彙對映
    for key, feat in STYLE_MAP.items():
        if key in t:
            params.update(feat)

    # 中文風格詞彙
    if any(k in t for k in ["饒舌", "嘻哈", "rap", "hiphop"]):
        params.update(STYLE_MAP["rap"])
    if any(k in t for k in ["樂團", "band"]):
        params.update(STYLE_MAP["band"])
    if any(k in t for k in ["搖滾", "rock"]):
        params.update(STYLE_MAP["rock"])

    # 情境關鍵詞
    if any(k in t for k in ["睡覺", "助眠", "放鬆", "冥想"]):
        params.setdefault("energy", (0.0, 0.4))
        params.setdefault("acousticness", (0.5, 1.0))
        params.setdefault("tempo", (55, 85))
    if any(k in t for k in ["讀書", "專心", "focus"]):
        params.setdefault("energy", (0.2, 0.5))
        params.setdefault("instrumentalness", (0.3, 1.0))
    if any(k in t for k in ["運動", "健身", "跑步", "workout"]):
        params.setdefault("energy", (0.7, 1.0))
        params.setdefault("danceability", (0.6, 1.0))
        params.setdefault("tempo", (120, 170))
    if any(k in t for k in ["抒情", "失戀", "療傷", "傷心"]):
        params.setdefault("valence", (0.0, 0.4))
        params.setdefault("energy", (0.2, 0.5))

    return params

# ========= [語意描述與分數工具] =========
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
    """做文字 embedding"""
    try:
        from openai import OpenAI
        client = OpenAI()
        res = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in res.data]
    except Exception:
        try:
            import openai
            res = openai.Embedding.create(model="text-embedding-ada-002", input=texts)
            return [d["embedding"] for d in res["data"]]
        except Exception as e:
            print(f"[warn] embedding failed: {e}")
            return []

def _numeric_affinity(feat: Dict, params: Dict) -> float:
    """用音樂特徵算一個 0~1 的接近度"""
    if not feat:
        return 0.5
    score_sum, cnt = 0.0, 0

    def closeness(v, t, scale=1.0):
        return max(0.0, 1.0 - abs((v - t) / scale))

    for k in ("energy", "valence", "danceability", "acousticness"):
        vk = feat.get(k); tk = params.get(f"target_{k}")
        if vk is not None and tk is not None:
            score_sum += closeness(vk, tk, 1.0); cnt += 1

    vtempo = feat.get("tempo"); ttempo = params.get("target_tempo")
    if vtempo and ttempo:
        score_sum += closeness(vtempo, ttempo, 120.0); cnt += 1
    elif vtempo and (params.get("min_tempo") or params.get("max_tempo")):
        lo = params.get("min_tempo", vtempo); hi = params.get("max_tempo", vtempo)
        if lo <= vtempo <= hi:
            score_sum += 1.0
        else:
            edge = lo if vtempo < lo else hi
            score_sum += max(0.0, 1.0 - abs(vtempo - edge) / 120.0)
        cnt += 1

    return (score_sum / cnt) if cnt else 0.5

def build_semantic_map(prompt: str, tracks: List[Dict], feats_map: Dict[str, Dict]) -> Dict[str, float]:
    """回傳 {track_id: 語意相似度(0~1)}"""
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
    """對 pool 排序：final = 0.6 * 語意 + 0.4 * 數值特徵接近度"""
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

# ======================================================
# OAuth & Token helpers
# ======================================================
def oauth():
    return SpotifyOAuth(
        client_id=os.environ.get("SPOTIPY_CLIENT_ID"),
        client_secret=os.environ.get("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.environ.get("SPOTIPY_REDIRECT_URI"),
        scope=SCOPE,
        cache_path=None,
        open_browser=False,
        show_dialog=True,
    )

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
# Spotify helpers
# ======================================================
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

def collect_user_tracks(sp, max_n=150):
    """抓使用者的常聽/已儲存歌曲"""
    pool = []
    try:
        tops = sp.current_user_top_tracks(limit=50, time_range="medium_term")
        for it in (tops or {}).get("items", []):
            if it and it.get("id"):
                pool.append(it)
            if len(pool) >= max_n:
                return pool[:max_n]
    except Exception as e:
        print(f"⚠️ current_user_top_tracks failed: {e}")

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

def collect_external_tracks_by_category(sp, text: str, target_lang: str = None, max_n: int = 200):
    """
    [修改點2] 根據語言和情境分類，從對應語言的歌單抓取外部歌曲
    """
    text = (text or "").lower()
    
    # 判斷情境分類
    if any(k in text for k in ["party", "派對", "嗨", "開心", "快樂"]):
        category = "party"
    elif any(k in text for k in ["sad", "傷心", "難過", "哭", "失戀", "emo", "抒情", "療傷"]):
        category = "sad"
    elif any(k in text for k in ["chill", "放鬆", "冷靜", "悠閒", "輕鬆"]):
        category = "chill"
    elif any(k in text for k in ["focus", "讀書", "專注", "工作", "coding", "專心"]):
        category = "focus"
    elif any(k in text for k in ["饒舌", "嘻哈", "rap", "hiphop"]):
        category = "rap"
    elif any(k in text for k in ["搖滾", "rock", "樂團", "band"]):
        category = "rock"
    elif any(k in text for k in ["r&b", "節奏藍調"]):
        category = "r&b"
    else:
        category = "default"

    print(f"[external] category={category}, target_lang={target_lang}")

    # 根據語言選擇搜尋關鍵字
    if target_lang and target_lang in LANG_PLAYLISTS:
        search_queries = LANG_PLAYLISTS[target_lang].get(category, LANG_PLAYLISTS[target_lang]["default"])
        market = {"zh": "TW", "ko": "KR", "ja": "JP", "en": "US"}.get(target_lang, "TW")
    else:
        # 預設英文
        search_queries = LANG_PLAYLISTS["en"].get(category, LANG_PLAYLISTS["en"]["default"])
        market = "US"

    tracks = []
    
    # 1. 搜尋該語言的歌單
    found_pids = []
    for query in search_queries[:3]:  # 限制搜尋數量避免過慢
        try:
            res = sp.search(q=query, type="playlist", market=market, limit=4)
            pitems = (((res or {}).get("playlists") or {}).get("items") or [])
            for pl in pitems:
                pid = (pl or {}).get("id")
                if pid: 
                    found_pids.append(pid)
        except Exception as e:
            print(f"[warn] search playlists '{query}' failed: {e}")
    
    # 從找到的歌單抓歌
    for pid in found_pids:
        if len(tracks) >= max_n:
            break
        try:
            tracks.extend(fetch_playlist_tracks(sp, pid, max_n=80))
        except Exception as e:
            print(f"[warn] fetch playlist {pid} failed: {e}")

    # 2. 如果還不夠，用 Recommendations API
    if len(tracks) < max_n // 2:
        try:
            # 根據語言和風格選擇種子
            if target_lang == "ko":
                seed_genres = ["k-pop"] if category == "default" else ["k-pop"]
            elif target_lang == "ja":
                seed_genres = ["j-pop"] if category == "default" else ["j-pop"]  
            elif category == "rap":
                seed_genres = ["hip-hop", "rap"]
            elif category == "rock":
                seed_genres = ["rock", "alternative"]
            elif category == "r&b":
                seed_genres = ["r-n-b", "soul"]
            else:
                seed_genres = ["pop", "indie"]
            
            # 限制種子數量
            seed_genres = seed_genres[:2]
            
            rec = sp.recommendations(
                seed_genres=seed_genres, 
                limit=min(100, max_n - len(tracks)),
                market=market
            )
            for tr in (rec or {}).get("tracks", []):
                if tr and isinstance(tr.get("id"), str):
                    tracks.append(tr)
        except Exception as e:
            print(f"[warn] recommendations failed: {e}")

    # 3. 最後回填：featured playlists
    if len(tracks) < max_n // 3:
        try:
            featured = sp.featured_playlists(country=market, limit=6)
            for pl in (featured or {}).get("playlists", {}).get("items", []):
                if len(tracks) >= max_n:
                    break
                tracks.extend(fetch_playlist_tracks(sp, pl.get("id"), max_n=50))
        except Exception as e:
            print(f"⚠️ featured_playlists failed: {e}")

    # 去重
    dedup, seen = [], set()
    for tr in tracks:
        tid = tr.get("id")
        if tid and tid not in seen:
            dedup.append(tr); seen.add(tid)

    print(f"[external] total={len(dedup)} for lang={target_lang}, category={category}")
    return dedup[:max_n]

def audio_features_map(sp, track_ids, batch_size: int = 50):
    """安全版：批次查 audio features"""
    valid_ids = [
        tid for tid in track_ids
        if isinstance(tid, str) and len(tid) == 22
    ]

    feats = {}
    skipped = []

    def _single_lookup(tid: str):
        try:
            row = sp.audio_features([tid])
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

    for i in range(0, len(valid_ids), batch_size):
        chunk = valid_ids[i:i + batch_size]
        try:
            rows = sp.audio_features(chunk)
            if not rows:
                print(f"[warn] batch audio_features empty for {len(chunk)} ids; fallback to single")
                for tid in chunk:
                    _single_lookup(tid)
                continue

            for tid, row in zip(chunk, rows):
                if row and isinstance(row, dict):
                    feats[tid] = row
                else:
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

def score_track(feat: Dict, params: Dict) -> float:
    """為單首歌曲計算分數（向後相容用）"""
    return _numeric_affinity(feat, params)

# ======================================================
# Helper functions
# ======================================================

def _weighted_pick(tracks, k=3):
    """根據 _score 權重抽樣，避免永遠同幾首"""
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

# ======================================================
# [修改點5] 改進的推薦邏輯：確保數量與多樣性
# ======================================================

def generate_playlist_with_language_priority(sp, text: str, recent_ids: list, avoid_ids: set):
    """
    主推薦邏輯：語言優先 + 數量保證 + 多樣性控制
    """
    # 1. 判斷語言偏好
    target_lang = _lang_hint_from_text(text)
    print(f"[recommend] target_lang={target_lang}")
    
    # 2. 收集候選池
    params = map_text_to_params(text)
    user_pool = collect_user_tracks(sp, max_n=120)
    
    # 根據語言調整外部來源
    if target_lang:
        ext_pool = collect_external_tracks_by_category(sp, text, target_lang, 250)
    else:
        ext_pool = collect_external_tracks_by_category(sp, text, "en", 200)
    
    # 3. 特徵分析
    ids_for_feat = []
    for tr in (user_pool + ext_pool):
        tid = tr.get("id")
        if isinstance(tid, str) and len(tid) == 22:
            ids_for_feat.append(tid)
            if len(ids_for_feat) >= 300:
                break
    
    feats = audio_features_map(sp, ids_for_feat)
    sem_map = build_semantic_map(text, user_pool + ext_pool, feats)
    
    # 4. 語言過濾（如果有指定語言）
    if target_lang:
        user_pool_filtered = _lang_filter(user_pool, target_lang, keep_if_none=True)
        ext_pool_filtered = _lang_filter(ext_pool, target_lang, keep_if_none=True)
        print(f"[lang_filter] user: {len(user_pool)} -> {len(user_pool_filtered)}")
        print(f"[lang_filter] ext: {len(ext_pool)} -> {len(ext_pool_filtered)}")
    else:
        user_pool_filtered = user_pool
        ext_pool_filtered = ext_pool
    
    # 5. 排序候選
    user_candidates = rank_pool_by_semantic_and_features(user_pool_filtered, feats, sem_map, params, top_n=30)
    ext_candidates = rank_pool_by_semantic_and_features(ext_pool_filtered, feats, sem_map, params, top_n=300)
    
    # 輕度打散
    random.shuffle(user_candidates)
    random.shuffle(ext_candidates)
    
    # 6. 避重處理
    used = set(avoid_ids) | set(recent_ids)
    user_all_ids = {t.get("id") for t in user_pool if isinstance(t.get("id"), str) and len(t.get("id")) == 22}
    
    # 7. 選擇最終歌曲：3首使用者 + 7首外部
    # 7a) 使用者歌曲（加權抽樣）
    anchors_pool = [t for t in user_candidates if isinstance(t.get("id"), str) and t["id"] not in used][:25]
    anchors = _weighted_pick(anchors_pool, k=3)
    for tr in anchors:
        if isinstance(tr.get("id"), str):
            used.add(tr["id"])
    
    # 7b) 外部歌曲（控制同歌手數量）
    ext_chosen, seen_artists = [], set()
    artist_count = {}  # 統計每個歌手的歌曲數
    
    for tr in ext_candidates:
        if len(ext_chosen) >= 7:
            break
        tid = tr.get("id")
        if not (isinstance(tid, str) and len(tid) == 22):
            continue
        if tid in used or tid in user_all_ids:
            continue
            
        aid = _first_artist_id(tr)
        # 同歌手限制：一般2首，語言稀缺時放寬到3首
        max_per_artist = 3 if target_lang in ["ko", "ja"] else 2
        
        if aid and artist_count.get(aid, 0) >= max_per_artist:
            continue
            
        ext_chosen.append(tr)
        used.add(tid)
        if aid:
            artist_count[aid] = artist_count.get(aid, 0) + 1
    
    # 7c) 如果外部不足7首，降低限制繼續補
    if len(ext_chosen) < 7:
        for tr in ext_candidates:
            if len(ext_chosen) >= 7:
                break
            tid = tr.get("id")
            if not (isinstance(tid, str) and len(tid) == 22) or tid in used:
                continue
            ext_chosen.append(tr)
            used.add(tid)
    
    final_tracks = (anchors + ext_chosen)
    
    # 8. 數量檢查與警告
    warnings = []
    if len(final_tracks) < 10:
        if target_lang:
            lang_names = {"zh": "中文", "ko": "韓文", "ja": "日文", "en": "英文"}
            warnings.append(f"僅找到 {len(final_tracks)} 首{lang_names.get(target_lang, target_lang)}歌曲，已盡力搜尋")
        else:
            warnings.append(f"僅找到 {len(final_tracks)} 首歌曲，建議調整描述")
    
    return final_tracks[:10], warnings

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
    <title>授權失敗 - Mooodyyy</title>
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

import os
from datetime import datetime
from flask import Flask, request, redirect, url_for, session

app = Flask(__name__)


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
        
        .form-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: #ffffff;
        }}
        
        .form-subtitle {{
            color: #b3b3b3;
            margin-bottom: 32px;
            font-size: 1rem;
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
                    <textarea 
                        name="text" 
                        placeholder="例如：中文抒情歌、韓文Kpop、日文搖滾、英文饒舌、深夜漫步思考人生、雨天在咖啡廳寫作、想念遠方的朋友、專心讀書需要輕音樂..."
                        required
                    ></textarea>
                </div>
                
                <input type="hidden" name="preview" value="1">
                <button type="submit" class="submit-btn">
                    🎵 開始推薦音樂
                </button>
            </form>
        </div>
        
        <div class="footer">
            <a href="/logout" class="logout-link">登出 Spotify</a>
        </div>
    </div>
</body>
</html>
'''

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    import traceback
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    text = (request.form.get("text") or request.args.get("text") or "").strip()
    if not text:
        return redirect(url_for("welcome"))

    ctx_key = (" ".join(text.lower().split()))[:80]
    history = session.get("hist", {})
    recent_ids = history.get(ctx_key, [])[:40]

    try:
        # [修改點5] 使用新的推薦邏輯
        avoid_raw = (request.form.get("avoid") or request.args.get("avoid") or "").strip()
        avoid_ids = set(i for i in avoid_raw.split(",") if len(i) == 22) if avoid_raw else set()
        
        top10, warnings = generate_playlist_with_language_priority(sp, text, recent_ids, avoid_ids)
        
        # 預覽頁面
        preview = (request.values.get("preview") or "").strip()
        if preview == "1":
            # [修改點4] 移除所有預覽相關功能，簡化前端
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
                track_id = tr.get("id", f"sample{i}")

                items.append(f'''
                <div class="track-item" draggable="true" data-track-id="{track_id}">
                    <div class="drag-handle">⋮⋮</div>
                    <div class="track-number">{i:02d}</div>
                    <div class="track-info">
                        <div class="track-name">{name}</div>
                        <div class="track-artist">{artists}</div>
                    </div>
                    <div class="track-actions">
                        <button class="action-btn" onclick="copySpotifyLink('{url}')" title="複製連結">
                            📋
                        </button>
                        <a href="{url}" target="_blank" class="action-btn spotify-link" title="在 Spotify 開啟">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="#1DB954" aria-hidden="true">
                                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.84-.179-.84-.6 0-.359.24-.66.54-.78 4.56-1.021 8.52-.6 11.64.301.42.12.66.54.42.96zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.301.421-1.02.599-1.56.3z"/>
                            </svg>
                        </a>
                    </div>
                </div>
                ''')

            songs_html = "\n".join(items)
            ids_str = ",".join([t.get("id") for t in top10 if isinstance(t.get("id"), str) and len(t.get("id")) == 22])
            safe_text = text.replace("'", "&#39;").replace('"', '&quot;')

            # 更新歷史記錄
            cur_ids = [t.get("id") for t in top10 if isinstance(t.get("id"), str) and len(t.get("id")) == 22]
            old_ids = [x for x in recent_ids if x not in cur_ids]
            history[ctx_key] = (cur_ids + old_ids)[:40]
            session["hist"] = history

            # 警告訊息處理
            warning_html = ""
            if warnings:
                warning_html = f'''
                <div class="warning-box">
                    <div class="warning-icon">⚠️</div>
                    <div class="warning-text">
                        {"<br>".join(warnings)}
                    </div>
                </div>
                '''

            page = f'''
            <!doctype html>
            <html lang="zh-Hant">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width,initial-scale=1">
                <title>為你推薦的歌單 - Mooodyyy</title>
                <style>
                    /* ===== CSS 保留完整 ===== */
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
                    {warning_html}
                    <div class="playlist-container">
                        <div class="playlist-header">
                            <div class="playlist-icon">🎵</div>
                            <div class="playlist-info">
                                <h3>專屬推薦歌單</h3>
                                <p>基於你的聆聽習慣與情境分析 • 可拖曳調整順序</p>
                            </div>
                        </div>
                        <div class="tracks-list" id="tracks-list">
                            {songs_html}
                        </div>
                    </div>
                    <div class="actions">
                        <form method="POST" action="/recommend" id="regen-form">
                            <input type="hidden" name="text" value="{safe_text}">
                            <input type="hidden" name="preview" value="1">
                            <input type="hidden" name="avoid" value="{ids_str}">
                            <button type="submit" class="btn btn-secondary" id="regen-btn">🔄 重新生成</button>
                        </form>
                        <form method="POST" action="/create_playlist" id="save-form">
                            <input type="hidden" name="text" value="{safe_text}">
                            <input type="hidden" name="track_ids" value="{ids_str}">
                            <button type="submit" class="btn btn-primary" id="save-btn">➕ 存到 Spotify</button>
                        </form>
                    </div>
                    <div class="back-link">
                        <a href="/welcome">↩️ 回到首頁</a>
                    </div>
                </div>

                <div class="toast" id="toast"></div>

                <div class="loading-overlay" id="loading">
                    <div class="loading-card">
                        <div class="loading-logo">
                            <svg viewBox="0 0 24 24" role="img" aria-label="Spotify">
                                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.84-.179-.84-.6 0-.359.24-.66.54-.78 4.56-1.021 8.52-.6 11.64.301.42.12.66.54.42.96zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.301.421-1.02.599-1.56.3z"/>
                            </svg>
                        </div>
                        <div class="loading-text">🎵 正在保存到 Spotify...</div>
                        <div class="loading-sub">為你創建專屬歌單</div>
                    </div>
                </div>
                
                <script>
                    // ===== JS 保留完整 =====
                </script>
            </body>
            </html>
            '''
            return page

        # 非預覽：直接建立歌單（向後相容）
        user = sp.current_user(); user_id = (user or {}).get("id")
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        title = f"Mooodyyy · {ts} UTC"
        desc = f"情境：{text}（由即時推薦建立）"
        plist = sp.user_playlist_create(user=user_id, name=title, public=False, description=desc)
        sp.playlist_add_items(playlist_id=plist["id"], items=[t["id"] for t in top10 if t.get("id")])
        url = (plist.get("external_urls") or {}).get("spotify", url_for("welcome"))
        return redirect(url)

    except Exception as e:
        print("⚠ recommend error:", e)
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
''', 500

@app.route("/create_playlist", methods=["POST"])
def create_playlist():
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    text = (request.form.get("text") or "").strip()
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
        user = sp.current_user()
        user_id = (user or {}).get("id")
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        title = f"Mooodyyy · {ts} UTC"
        desc = f"情境：{text}（由預覽頁直接保存）"

        playlist = sp.user_playlist_create(
            user=user_id,
            name=title,
            public=False,
            description=desc
        )
        sp.playlist_add_items(playlist_id=playlist["id"], items=ids)

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
    v = os.environ.get("SPOTIFY_REDIRECT_URI", "<none>")
    return f"[{v}] len={len(v)}", 200


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=PORT)
