from flask import Flask, request, redirect, session, url_for
import os
import time
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import math
from typing import List, Dict
from spotipy.exceptions import SpotifyException


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
    try:
        me = sp.current_user()
        name = (me or {}).get("display_name") or "音樂愛好者"
    except Exception as e:
        print(f"⚠️ current_user failed: {e}")
        name = "音樂愛好者"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset='UTF-8' />
      <title>Mooodyyy - AI 音樂情境推薦</title>
      <style>
        body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:linear-gradient(135deg,#1DB954,#1ed760); color:#fff; margin:0; padding:20px; min-height:100vh; }}
        .container {{ max-width:640px; margin:0 auto; }}
        .card {{ background:rgba(255,255,255,0.1); border-radius:16px; padding:28px; backdrop-filter:blur(10px); }}
        textarea {{ width:100%; box-sizing:border-box; border:none; border-radius:10px; padding:14px; font-size:16px; resize:vertical; }}
        button {{ background:#FF6B6B; color:#fff; border:none; padding:12px 20px; border-radius:8px; font-size:16px; cursor:pointer; margin-top:10px; }}
        button:hover {{ background:#ff5252; }}
        a {{ color:#fff; }}
      </style>
    </head>
    <body>
      <div class='container'>
        <h1>🎵 Hello {name}</h1>
        <p>歡迎來到 Mooodyyy — 用一句話描述你的情境，我來幫你配歌。</p>

        <div class='card'>
          <h2>🎯 情境推薦</h2>
          <p>輸入你的心情或場景，例如：下雨夜的鋼琴、專心讀書的輕音樂、失戀的深夜車程⋯</p>
          <form action='/recommend' method='post'>
            <textarea name='text' rows='4' placeholder='例如：下班後的放鬆小酒館氛圍'></textarea><br/>
            <button type='submit'>生成 Top 10</button>
          </form>
        </div>

        <p style='text-align:center; margin-top:32px; opacity:.85;'>
          <a href='/logout'>登出</a>
        </p>
      </div>
    </body>
    </html>
    """
    return html


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for("home"))

    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
    else:
        text = (request.args.get("text") or "").strip()

    if not text:
        return redirect(url_for("welcome"))

    try:
        # 1) 收集候選池
        user_pool = collect_user_tracks(sp, max_n=150)
        ext_pool  = collect_external_tracks_by_category(sp, text, max_n=300)

        if not user_pool and not ext_pool:
            return (
                "<h2>❌ 暫時無法獲取歌曲</h2>"
                "<p>請先重新登入授權（讀取常聽/已儲存），或稍後再試。</p>"
                "<a href='/welcome'>回首頁</a>"
            )

        # 2) 開始計分
        t0 = time.time()
        params = map_text_to_params(text)

        # 收集有效 track id，最多 300
        ids, seen = [], set()
        for t in (user_pool + ext_pool):
            tid = t.get("id")
            if isinstance(tid, str) and len(tid) == 22 and tid not in seen:
                ids.append(tid)
                seen.add(tid)
                if len(ids) >= 300:
                    break

        feats = audio_features_map(sp, ids)

        # 3) 語意 + 特徵排分（重點）
        all_candidates = user_pool + ext_pool
        sem_map = build_semantic_map(text, all_candidates, feats)

        user_candidates = rank_pool_by_semantic_and_features(
            user_pool, feats, sem_map, params, top_n=10
        )
        ext_candidates = rank_pool_by_semantic_and_features(
            ext_pool, feats, sem_map, params, top_n=50
        )

        # 工具：安全拿 artist id
        def _safe_artist_id(tr):
            a = tr.get("artists") or tr.get("artist") or []
            if isinstance(a, list) and a:
                first = a[0]
                return first.get("id") if isinstance(first, dict) else None
            if isinstance(a, dict):
                return a.get("id")
            return None

        user_all_ids = {
            t.get("id") for t in user_pool
            if isinstance(t.get("id"), str) and len(t.get("id")) == 22
        }

        # 4) 3 熟悉 + 7 新歌（不硬塞）
        used = set()

        # 4a) 你的曲庫：最多 3 首當熟悉 anchor
        anchors = []
        for tr in user_candidates:
            tid = tr.get("id")
            if not isinstance(tid, str) or len(tid) != 22:
                continue
            if tid in used:
                continue
            tr["source"] = "user"
            anchors.append(tr)
            used.add(tid)
            if len(anchors) >= 3:
                break

        # 4b) 外部新歌：最多 7 首，先嚴格排除你曲庫 + 同歌手抑制
        ext_chosen, seen_artists = [], set()
        for tr in ext_candidates:
            if len(ext_chosen) >= 7:
                break
            tid = tr.get("id")
            if not isinstance(tid, str) or len(tid) != 22:
                continue
            if tid in used or tid in user_all_ids:
                continue
            aid = _safe_artist_id(tr)
            if aid and aid in seen_artists:
                continue
            seen_artists.add(aid)
            tr["source"] = "external"
            ext_chosen.append(tr)
            used.add(tid)

        # 4c) 若還不滿 7：放寬（可含你曲庫也有的，但仍避免重複/洗版）
        if len(ext_chosen) < 7:
            for tr in ext_candidates:
                if len(ext_chosen) >= 7:
                    break
                tid = tr.get("id")
                if not isinstance(tid, str) or len(tid) != 22 or tid in used:
                    continue
                aid = _safe_artist_id(tr)
                if aid and aid in seen_artists:
                    continue
                seen_artists.add(aid)
                tr["source"] = "external"
                ext_chosen.append(tr)
                used.add(tid)

        # 4d) 混合 + 補齊到 10（優先外部，再回頭用你的曲庫）
        mixed = anchors + ext_chosen

        if len(mixed) < 10:
            for tr in ext_candidates:
                if len(mixed) >= 10:
                    break
                tid = tr.get("id")
                if not isinstance(tid, str) or len(tid) != 22 or tid in used:
                    continue
                tr["source"] = "external"
                mixed.append(tr)
                used.add(tid)

        if len(mixed) < 10:
            for tr in user_candidates:
                if len(mixed) >= 10:
                    break
                tid = tr.get("id")
                if not isinstance(tid, str) or len(tid) != 22 or tid in used:
                    continue
                tr["source"] = "user"
                mixed.append(tr)
                used.add(tid)

        top10 = mixed[:10]
        dt = time.time() - t0

        # 5) 預覽模式 or 直接建立私人歌單
        preview = (request.args.get("preview") or request.form.get("preview") or "").strip()

        if preview == "1":
            # 預覽頁（保留你原本的樣式）
            try:
                songs_html = "\n".join(item_li(i + 1, tr) for i, tr in enumerate(top10))
            except Exception:
                # 若你的專案沒有 item_li()，退回簡易列印
                items = []
                for i, tr in enumerate(top10, 1):
                    nm = tr.get("name", "")
                    artists = ", ".join(a.get("name", "") for a in tr.get("artists", []))
                    u = (tr.get("external_urls") or {}).get("spotify", "#")
                    src = "（你的曲庫）" if tr.get("source") == "user" else "（新探索）"
                    items.append(f"<li>{i:02d}. <a href='{u}' target='_blank'>{artists} — {nm}</a> {src}</li>")
                songs_html = "\n".join(items)

            buttons_html = f"""
            <div style='margin: 20px 0;'>
              <form method='POST' action='/create_playlist' style='display:inline; margin-right:10px;'>
                <input type='hidden' name='mode' value='private'>
                <input type='hidden' name='text' value='{text}'>
                <button type='submit' style='background:#333; color:#fff; border:none; padding:10px 20px; border-radius:6px;'>➕ 存成「私人歌單」</button>
              </form>
              <form method='POST' action='/create_playlist' style='display:inline;'>
                <input type='hidden' name='mode' value='public'>
                <input type='hidden' name='text' value='{text}'>
                <button type='submit' style='background:#1DB954; color:#fff; border:none; padding:10px 20px; border-radius:6px;'>➕ 存成「公開歌單」</button>
              </form>
            </div>
            """

            page = f"""
            <html><head><meta charset='utf-8'><title>推薦結果（預覽）</title></head>
            <body>
              <div style='max-width:800px;margin:24px auto;font-family:sans-serif;'>
                <h1>🎯 為你找到了 {len(top10)} 首歌</h1>
                <p><strong>你的情境：</strong>"{text}"</p>
                <p style='opacity:.85;'>候選來源：{len(user_pool)}（個人） + {len(ext_pool)}（外部） → 耗時 {dt:.1f} 秒｜規則：最多 3（個人）+ 至多 7（外部）</p>
                <h2>🎵 推薦歌單：</h2>
                <ol style='padding-left:0;'>
                  {songs_html}
                </ol>
                {buttons_html}
                <p style='margin-top:24px;'><a href='/welcome'>↩️ 回首頁</a> | <a href='/recommend'>🔄 再試一次</a></p>
              </div>
            </body></html>
            """
            return page

        # 預設：直接建立「私人」歌單並導去 Spotify
        user = sp.current_user()
        user_id = (user or {}).get("id")

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        title = f"Mooodyyy · {ts} UTC"
        desc  = f"情境：{text}（最多 3 首來自個人曲庫 + 其餘外部）"

        playlist = sp.user_playlist_create(
            user=user_id,
            name=title,
            public=False,  # 固定私人
            description=desc
        )
        sp.playlist_add_items(playlist_id=playlist["id"], items=[t["id"] for t in top10])

        url = (playlist.get("external_urls") or {}).get("spotify", "#")
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

    mode = (request.form.get("mode") or "private").strip()
    text = (request.form.get("text") or "").strip()
    if not text or mode not in ("public", "private"):
        return "參數不完整。<a href='/recommend?preview=1'>返回</a>"

    try:
        # 1) 收集候選池
        params = map_text_to_params(text)
        user_pool = collect_user_tracks(sp, max_n=150)
        ext_pool  = collect_external_tracks_by_category(sp, text, max_n=300)
        if not user_pool and not ext_pool:
            return "沒有可加入的歌曲。<a href='/recommend?preview=1'>返回</a>"

        # 收集有效 track id，最多 300
        ids, seen = [], set()
        for t in (user_pool + ext_pool):
            tid = t.get("id")
            if isinstance(tid, str) and len(tid) == 22 and tid not in seen:
                ids.append(tid)
                seen.add(tid)
                if len(ids) >= 300:
                    break

        feats = audio_features_map(sp, ids)

        # 2) 語意 + 特徵排分
        all_candidates = user_pool + ext_pool
        sem_map = build_semantic_map(text, all_candidates, feats)

        user_candidates = rank_pool_by_semantic_and_features(
            user_pool, feats, sem_map, params, top_n=10
        )
        ext_candidates = rank_pool_by_semantic_and_features(
            ext_pool, feats, sem_map, params, top_n=50
        )

        def _safe_artist_id(tr):
            a = tr.get("artists") or tr.get("artist") or []
            if isinstance(a, list) and a:
                first = a[0]
                return first.get("id") if isinstance(first, dict) else None
            if isinstance(a, dict):
                return a.get("id")
            return None

        user_all_ids = {
            t.get("id") for t in user_pool
            if isinstance(t.get("id"), str) and len(t.get("id")) == 22
        }

        used = set()

        # 3) 3 熟悉 + 7 新歌（不硬塞）
        anchors = []
        for tr in user_candidates:
            tid = tr.get("id")
            if not isinstance(tid, str) or len(tid) != 22 or tid in used:
                continue
            tr["source"] = "user"
            anchors.append(tr)
            used.add(tid)
            if len(anchors) >= 3:
                break

        ext_chosen, seen_artists = [], set()
        for tr in ext_candidates:
            if len(ext_chosen) >= 7:
                break
            tid = tr.get("id")
            if not isinstance(tid, str) or len(tid) != 22:
                continue
            if tid in used or tid in user_all_ids:
                continue
            aid = _safe_artist_id(tr)
            if aid and aid in seen_artists:
                continue
            seen_artists.add(aid)
            tr["source"] = "external"
            ext_chosen.append(tr)
            used.add(tid)

        if len(ext_chosen) < 7:
            for tr in ext_candidates:
                if len(ext_chosen) >= 7:
                    break
                tid = tr.get("id")
                if not isinstance(tid, str) or len(tid) != 22 or tid in used:
                    continue
                aid = _safe_artist_id(tr)
                if aid and aid in seen_artists:
                    continue
                seen_artists.add(aid)
                tr["source"] = "external"
                ext_chosen.append(tr)
                used.add(tid)

        mixed = anchors + ext_chosen

        if len(mixed) < 10:
            for tr in ext_candidates:
                if len(mixed) >= 10:
                    break
                tid = tr.get("id")
                if not isinstance(tid, str) or len(tid) != 22 or tid in used:
                    continue
                tr["source"] = "external"
                mixed.append(tr)
                used.add(tid)

        if len(mixed) < 10:
            for tr in user_candidates:
                if len(mixed) >= 10:
                    break
                tid = tr.get("id")
                if not isinstance(tid, str) or len(tid) != 22 or tid in used:
                    continue
                tr["source"] = "user"
                mixed.append(tr)
                used.add(tid)

        top10 = mixed[:10]

        # 4) 建立歌單（公開/私人 由按鈕決定）
        user = sp.current_user()
        user_id = (user or {}).get("id")
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        title = f"Mooodyyy · {ts} UTC"
        desc  = f"情境：{text}（最多 3 首來自個人曲庫 + 其餘外部）"

        playlist = sp.user_playlist_create(
            user=user_id,
            name=title,
            public=(mode == "public"),
            description=desc
        )
        sp.playlist_add_items(playlist_id=playlist["id"], items=[t["id"] for t in top10])
        url = (playlist.get("external_urls") or {}).get("spotify", "#")

        # 成功頁（保留，方便從預覽模式回來）
        items_html = []
        for i, tr in enumerate(top10, 1):
            nm = tr.get("name", "")
            artists = ", ".join(a.get("name", "") for a in tr.get("artists", []))
            u = (tr.get("external_urls") or {}).get("spotify", "#")
            src = tr.get("source", "")
            badge = "（你的曲庫）" if src == "user" else "（新探索）"
            items_html.append(f"<li>{i:02d}. <a href='{u}' target='_blank'>{artists} — {nm}</a> {badge}</li>")

        return f"""
            <h2>✅ 已建立歌單：<a href='{url}' target='_blank'>{title}</a></h2>
            <p>模式：{"公開" if mode=="public" else "私人"}</p>
            <p>情境：{text}</p>
            <h3>曲目：</h3>
            <ol>{''.join(items_html)}</ol>
            <p><a href='/recommend?preview=1'>↩︎ 回預覽頁</a> ｜ <a href='/welcome'>🏠 回首頁</a></p>
        """

    except Exception as e:
        print(f"❌ create_playlist error: {e}")
        return (
            "<h2>❌ 建立歌單失敗</h2>"
            f"<p>錯誤訊息：{str(e)}</p>"
            "<a href='/recommend?preview=1'>返回</a>"
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
