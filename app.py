from flask import Flask, request, redirect, session, url_for
import os
import time
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import math
from typing import List, Dict
from spotipy.exceptions import SpotifyException
from flask import request, redirect, url_for
from datetime import datetime



app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "devsecret")
SCOPE = "user-library-read user-top-read playlist-modify-public playlist-modify-private"

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

def map_text_to_params(text):
    t = (text or "").lower()
    params = {
        "target_energy": 0.5,
        "target_valence": 0.5,
        "target_tempo": 110.0,
        "prefer_acoustic": False,
        "prefer_instrumental": False,
        "seed_genres": [],
    }
    if any(k in t for k in ["累", "疲", "sad", "lonely", "emo", "哭"]):
        params.update({"target_energy": 0.2, "target_valence": 0.25, "target_tempo": 80})
    if any(k in t for k in ["開心", "爽", "happy", "party", "嗨"]):
        params.update({"target_energy": 0.8, "target_valence": 0.8, "target_tempo": 125})
    if any(k in t for k in ["讀書", "專心", "focus", "工作", "coding"]):
        params.update({"target_energy": 0.3, "target_valence": 0.5, "target_tempo": 90})
    if any(k in t for k in ["爵士", "jazz"]):
        params["seed_genres"].append("jazz")
    if any(k in t for k in ["lofi", "lo-fi", "lo fi", "輕音"]):
        params["seed_genres"].append("lo-fi")
    if any(k in t for k in ["鋼琴", "piano", "acoustic"]):
        params["prefer_acoustic"] = True
    if any(k in t for k in ["純音樂", "instrumental"]):
        params["prefer_instrumental"] = True
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
    return '<a href="/login">🔐 Login with Spotify</a>'


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
        return "<h3>Authorization failed.</h3><a href='/'>Try again</a>"

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

    return f"""
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Mooodyyy · Welcome</title>
  <style>
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif; background:#0f1115; color:#fff; margin:0;}}
    .wrap{{max-width:720px; margin:0 auto; padding:32px 20px;}}
    h1{{margin:0 0 12px;}}
    p.muted{{color:#aeb4be; margin:6px 0 20px;}}
    textarea{{width:100%; min-height:140px; padding:12px; font-size:16px; border-radius:10px; border:1px solid #2a2f3a; background:#12161f; color:#e9eef7; box-sizing:border-box;}}
    button{{padding:10px 16px; border:none; border-radius:10px; cursor:pointer; background:#1DB954; color:#0b0f14; font-weight:700;}}
    .card{{background:#12161f; border:1px solid #1b2030; border-radius:16px; padding:20px;}}
    a{{color:#8bd9ff; text-decoration:none;}}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>🎧 Mooodyyy</h1>
    <p class="muted">嗨，{name}。描述你的情境，我會推薦一份 10 首的歌單。</p>

    <div class="card">
      <form action="/recommend" method="post">
        <textarea name="text" placeholder="例：深夜散步、下雨寫作、想念老朋友、專心讀書的輕音樂…"></textarea>

        <!-- 預設走「先預覽」 -->
        <input type="hidden" name="preview" value="1">

        <div style="margin-top:12px;">
          <button type="submit">推薦歌單</button>
        </div>
      </form>
    </div>

    <p class="muted" style="margin-top:16px;">送出後會先顯示「歌單預覽」。你可以選擇 🔄 重新生成 或 ➕ 存到 Spotify。</p>
    <p style="text-align:center; margin-top:16px;"><a href="/logout">登出</a></p>
  </div>
</body>
</html>
"""


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    # 取得情境文字（POST 優先、GET 次之）
    text = (request.form.get("text") or request.args.get("text") or "").strip()
    if not text:
        return redirect(url_for("welcome"))

    try:
        # === 1) 收集候選池（沿用你原本的方法） ===
        params = map_text_to_params(text)
        user_pool = collect_user_tracks(sp, max_n=150)                   # 你的曲庫候選
        ext_pool  = collect_external_tracks_by_category(sp, text, 300)   # 外部候選

        # === 2) 準備特徵與語意分數（沿用你原本的方法） ===
        ids_for_feat = []
        for tr in (user_pool + ext_pool):
            tid = tr.get("id")
            if isinstance(tid, str) and len(tid) == 22:
                ids_for_feat.append(tid)
                if len(ids_for_feat) >= 300:
                    break
        feats   = audio_features_map(sp, ids_for_feat)
        sem_map = build_semantic_map(text, user_pool + ext_pool, feats)

        # === 3) 排序（沿用你原本的方法） ===
        user_candidates = rank_pool_by_semantic_and_features(user_pool, feats, sem_map, params, top_n=10)
        ext_candidates  = rank_pool_by_semantic_and_features(ext_pool,  feats, sem_map, params, top_n=50)

        # === 4) 混合：最多 3 首你的曲庫 + 最多 7 首外部（沿用你的策略） ===
        used, anchors = set(), []
        for tr in user_candidates:
            tid = tr.get("id")
            if isinstance(tid, str) and len(tid) == 22 and tid not in used:
                anchors.append(tr); used.add(tid)
                if len(anchors) >= 3: break

        user_all_ids = {t.get("id") for t in user_pool if isinstance(t.get("id"), str) and len(t.get("id")) == 22}

        def _first_artist_id(tr):
            a = tr.get("artists") or tr.get("artist") or []
            if isinstance(a, list) and a and isinstance(a[0], dict):
                return a[0].get("id")
            if isinstance(a, dict):
                return a.get("id")
            return None

        ext_chosen, seen_artists = [], set()
        for tr in ext_candidates:
            if len(ext_chosen) >= 7: break
            tid = tr.get("id")
            if not (isinstance(tid, str) and len(tid) == 22): continue
            if tid in used or tid in user_all_ids: continue
            aid = _first_artist_id(tr)
            if aid and aid in seen_artists: continue
            seen_artists.add(aid)
            ext_chosen.append(tr); used.add(tid)

        # 不足就放寬補滿到 7
        if len(ext_chosen) < 7:
            for tr in ext_candidates:
                if len(ext_chosen) >= 7: break
                tid = tr.get("id")
                if not (isinstance(tid, str) and len(tid) == 22) or tid in used: continue
                ext_chosen.append(tr); used.add(tid)

        top10 = (anchors + ext_chosen)[:10]

        # === 5) 預覽頁（精簡顯示） ===
        preview = (request.values.get("preview") or "").strip()
        if preview == "1":
            # 生成乾淨的清單：1. {Artist} — {Name}
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
                items.append(f"<li>{i}. <a href='{url}' target='_blank'>{artists} — {name}</a></li>")
            songs_html = "\n".join(items)

            ids_str   = ",".join([t.get("id") for t in top10 if isinstance(t.get("id"), str) and len(t.get("id")) == 22])
            safe_text = text.replace("'", "&#39;")

            page = f"""
            <!doctype html>
            <html lang="zh-Hant">
            <head>
              <meta charset="utf-8">
              <title>推薦結果（預覽）</title>
              <meta name="viewport" content="width=device-width,initial-scale=1">
              <style>
                body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif; background:#0f1115; color:#fff;}}
                .wrap{{max-width:800px; margin:24px auto; padding:0 16px;}}
                button{{padding:10px 12px; border:none; border-radius:8px; cursor:pointer;}}
                .btn-save{{background:#1DB954; color:#0b0f14; font-weight:700;}}
                .btn-regen{{background:#2e323c; color:#fff;}}
                a{{color:#8bd9ff; text-decoration:none;}}
                ol{{line-height:1.8;}}
              </style>
            </head>
            <body>
              <div class="wrap">
                <h1>🎯 為你找到了 {len(top10)} 首歌</h1>
                <p><strong>你的情境：</strong>"{safe_text}"</p>
                <ol>
                  {songs_html}
                </ol>
                <div style="margin:20px 0; display:flex; gap:10px; flex-wrap:wrap;">
                  <form method="POST" action="/recommend" style="display:inline;">
                    <input type="hidden" name="text" value="{safe_text}">
                    <input type="hidden" name="preview" value="1">
                    <button type="submit" class="btn-regen">🔄 重新生成</button>
                  </form>
                  <form method="POST" action="/create_playlist" style="display:inline;">
                    <input type="hidden" name="text" value="{safe_text}">
                    <input type="hidden" name="track_ids" value="{ids_str}">
                    <button type="submit" class="btn-save">➕ 存到 Spotify</button>
                  </form>
                </div>
                <p><a href="/welcome">↩︎ 回首頁</a></p>
              </div>
            </body>
            </html>
            """
            return page

        # 非預覽：直接建「私人」歌單後跳轉（保留舊行為以向後相容）
        user   = sp.current_user(); user_id = (user or {}).get("id")
        ts     = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        title  = f"Mooodyyy · {ts} UTC"
        desc   = f"情境：{text}（由即時推薦建立）"
        plist  = sp.user_playlist_create(user=user_id, name=title, public=False, description=desc)
        sp.playlist_add_items(playlist_id=plist["id"], items=[t["id"] for t in top10 if t.get("id")])
        url    = (plist.get("external_urls") or {}).get("spotify", url_for("welcome"))
        return redirect(url)

    except Exception as e:
        print(f"❌ recommend error: {e}")
        return (
            "<h2>❌ 系統暫時出錯</h2>"
            f"<p>錯誤訊息：{str(e)}</p>"
            "<a href='/welcome'>回首頁</a>"
        )

@app.route("/create_playlist", methods=["POST"])
def create_playlist():
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    text = (request.form.get("text") or "").strip()

    # 1) 讀取預覽頁的歌曲清單
    track_ids_raw = (request.form.get("track_ids") or "").strip()
    ids = [i for i in track_ids_raw.split(",") if len(i) == 22] if track_ids_raw else []

    if not ids:
        return (
            "<h3>沒有要加入的歌曲</h3>"
            "<p>請先到預覽頁，再按「存到 Spotify」。</p>"
            "<p><a href='/welcome'>↩︎ 回首頁</a></p>"
        )

    try:
        # 2) 建立私人歌單並加入歌曲
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

        # 3) 成功後直接跳 Spotify
        url = (playlist.get("external_urls") or {}).get("spotify", url_for("welcome"))
        return redirect(url)

    except Exception as e:
        print(f"❌ create_playlist error: {e}")
        return (
            "<h2>❌ 建立歌單失敗</h2>"
            f"<p>錯誤訊息：{str(e)}</p>"
            "<p><a href='/welcome'>↩︎ 回首頁</a></p>"
        )


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
