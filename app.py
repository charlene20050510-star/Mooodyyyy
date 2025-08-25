from flask import Flask, request, redirect, session, url_for
import os
import time
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "devsecret")
SCOPE = "user-library-read user-top-read playlist-modify-public playlist-modify-private"

# ======================================================
# Flask & Spotify OAuth setup
# ======================================================

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

def audio_features_map(sp, track_ids):
    # 先把 cache 裡有的拿出來
    feats = {}
    to_query = []
    for tid in track_ids:
        if tid in CACHE["feat"]:
            feats[tid] = CACHE["feat"][tid]
        else:
            to_query.append(tid)

    # 安全起見限制最多查 300 筆（足夠 3+7 流程）
    to_query = to_query[:300]

    # 用安全批次查
    fresh = _safe_audio_features(sp, to_query)

    # 寫回 cache
    for tid, f in fresh.items():
        CACHE["feat"][tid] = f
        feats[tid] = f

    return feats


def _dist(a, b):
    return abs((a or 0) - (b or 0))


def score_track(f, p):
    if not f:
        return 1e9
    s = 0.0
    s += _dist(f.get("energy"), p["target_energy"])  # energy gap
    s += _dist(f.get("valence"), p["target_valence"])  # positivity gap
    tempo = f.get("tempo") or p["target_tempo"]
    s += abs(tempo - p["target_tempo"]) / 100.0  # scale to similar magnitude
    if p["prefer_acoustic"]:
        s += (1.0 - (f.get("acousticness") or 0)) * 0.5
    if p["prefer_instrumental"]:
        s += (1.0 - (f.get("instrumentalness") or 0)) * 0.5
    return s


def select_top(tracks, feats, params, top_n=10):
    scored = []
    for t in tracks:
        tid = t.get("id")
        s = score_track(feats.get(tid), params)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0])
    return [t for _, t in scored[:top_n]]

def item_li(i, tr):
    name = tr.get("name", "Unknown")
    artists = ", ".join([a.get("name", "") for a in tr.get("artists", [])])
    url = (tr.get("external_urls") or {}).get("spotify", "#")
    return f"<li style='margin:8px 0; list-style:none;'>{i:02d}. <a href='{url}' target='_blank' style='color:#1DB954'><strong>{artists}</strong> - {name}</a></li>"

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
        # 收集候選池
        user_pool = collect_user_tracks(sp, max_n=150)
        ext_pool  = collect_external_tracks(sp, max_n=300)

        if not user_pool and not ext_pool:
            return (
                "<h2>❌ 暫時無法獲取歌曲</h2>"
                "<p>請先重新登入授權（讀取常聽/已儲存），或稍後再試。</p>"
                "<a href='/welcome'>回首頁</a>"
            )

        # 開始推薦
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

        # ===== 3 熟悉 + 7 新鮮（不硬塞）基本版 =====
        used = set()

        user_all_ids = {
            t.get("id") for t in user_pool
            if isinstance(t.get("id"), str) and len(t.get("id")) == 22
        }

        def _safe_artist_id(tr):
            a = tr.get("artists") or tr.get("artist") or []
            if isinstance(a, list) and a:
                first = a[0]
                return first.get("id") if isinstance(first, dict) else None
            if isinstance(a, dict):
                return a.get("id")
            return None

        # 1) 你的曲庫：先挑最多 10 首候選，再取前 3 當熟悉基底
        user_candidates = pick_top_n(user_pool, feats, params, n=10, used_ids=set())

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
                break  # 不足就少於 3，不硬塞

        # 2) 外部候選：抓較大池做多樣性/新鮮度過濾
        ext_candidates = pick_top_n(ext_pool, feats, params, n=50, used_ids=set())

        ext_chosen, seen_artists = [], set()
        # 2a) 嚴格新鮮：排除你曲庫已有 + 同歌手抑制
        for tr in ext_candidates:
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
            if len(ext_chosen) >= 7:
                break

        # 2b) 若還不滿 7，放寬：可含你曲庫有的，但仍避免重複/洗版
        if len(ext_chosen) < 7:
            for tr in ext_candidates:
                if len(ext_chosen) >= 7:
                    break
                tid = tr.get("id")
                if not isinstance(tid, str) or len(tid) != 22:
                    continue
                if tid in used:
                    continue
                aid = _safe_artist_id(tr)
                if aid and aid in seen_artists:
                    continue
                seen_artists.add(aid)
                tr["source"] = "external"
                ext_chosen.append(tr)
                used.add(tid)

        # 3) 混合 + 補齊到 10
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

        # === 預覽模式 or 自動建立私人歌單 ===
        songs_html = "\n".join(item_li(i + 1, tr) for i, tr in enumerate(top10))
        preview = (request.args.get("preview") or request.form.get("preview") or "").strip()

        if preview == "1":
            # 預覽：顯示結果 + 兩顆按鈕
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
