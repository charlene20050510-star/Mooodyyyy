from openai import OpenAI

from flask import Flask, request, redirect, session, url_for
import os, random, re
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
app.secret_key = os.environ.get("SECRET_KEY", "devsecret")

# 權限：讀取歌庫 + 建立/修改歌單
SCOPE = "user-library-read playlist-modify-public playlist-modify-private"

def oauth():
    return SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ["SPOTIPY_REDIRECT_URI"],  # 必須與 Spotify Dashboard 完全一致
        scope=SCOPE,
        cache_path=None,
        open_browser=False,
        show_dialog=True
    )

def get_sp():
    """用 session 裡的 access_token 建一個 Spotipy client。"""
    tok = session.get("access_token")
    if not tok:
        return None
    return spotipy.Spotify(auth=tok)

# ---------- 基本路由 ----------
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
    token_info = oauth().get_access_token(code, as_dict=True)
    session["access_token"] = token_info["access_token"]
    return redirect(url_for("welcome"))

@app.route("/ping")
def ping():
    return "PING OK", 200

import time
from collections import defaultdict

# 小快取（重複輸入不上 OpenAI / 重複歌曲不上 audio_features）
CACHE = {"emb": {}, "feat": {}}

def map_text_to_params(text: str):
    """把自然語言變成音樂參數（簡化版規則，可之後換成 embedding 相似度）"""
    t = text.lower()
    params = {
        "target_energy": 0.5,
        "target_valence": 0.5,
        "target_tempo": 110.0,
        "prefer_acoustic": False,
        "prefer_instrumental": False,
        "seed_genres": []
    }
    if any(k in t for k in ["累", "疲", "sad", "lonely", "emo", "哭"]):
        params.update({"target_energy": 0.2, "target_valence": 0.25, "target_tempo": 80})
    if any(k in t for k in ["開心", "爽", "happy", "party", "嗨"]):
        params.update({"target_energy": 0.8, "target_valence": 0.8, "target_tempo": 125})
    if any(k in t for k in ["讀書", "專心", "focus", "工作", "coding"]):
        params.update({"target_energy": 0.3, "target_valence": 0.5, "target_tempo": 90})
    if any(k in t for k in ["爵士", "jazz"]): params["seed_genres"].append("jazz")
    if any(k in t for k in ["lofi", "lo-fi", "lo fi", "輕音"]): params["seed_genres"].append("lo-fi")
    if any(k in t for k in ["鋼琴", "piano", "acoustic"]): params["prefer_acoustic"] = True
    if any(k in t for k in ["純音樂", "instrumental"]): params["prefer_instrumental"] = True
    return params

def fetch_playlist_tracks(sp, playlist_id: str, max_n: int = 100):
    """從指定歌單抓歌（候選集先小：最多 100 首）"""
    tracks = []
    offset = 0
    while len(tracks) < max_n:
        batch = sp.playlist_items(playlist_id, offset=offset, limit=50)
        items = batch.get("items", [])
        if not items: break
        for it in items:
            tr = (it or {}).get("track") or {}
            if tr.get("id"): tracks.append(tr)
            if len(tracks) >= max_n: break
        if not batch.get("next"): break
        offset += 50
    return tracks

def audio_features_map(sp, track_ids):
    """用快取 + 批次查 audio_features"""
    feats = {}
    to_query = []
    for tid in track_ids:
        if tid in CACHE["feat"]:
            feats[tid] = CACHE["feat"][tid]
        else:
            to_query.append(tid)
    for i in range(0, len(to_query), 50):
        chunk = to_query[i:i+50]
        res = sp.audio_features(chunk) or []
        for f in (res or []):
            if not f: continue
            tid = f.get("id")
            if tid:
                CACHE["feat"][tid] = f
                feats[tid] = f
    return feats

def score_track(f, p):
    """簡單打分：越接近目標越好；可再加權"""
    if not f: return 1e9
    def d(a,b): return abs((a or 0) - (b or 0))
    s = 0.0
    s += d(f.get("energy"), p["target_energy"])
    s += d(f.get("valence"), p["target_valence"])
    # tempo 有時為零或 None，先做防呆
    tempo = f.get("tempo") or p["target_tempo"]
    s += min(abs(tempo - p["target_tempo"]) / 60.0, 1.0)
    if p["prefer_acoustic"]:
        s += (1.0 - (f.get("acousticness") or 0)) * 0.5
    if p["prefer_instrumental"]:
        s += (1.0 - (f.get("instrumentalness") or 0)) * 0.5
    return s

def select_top(tracks, feats, params, top_n=10):
    scored = []
    for t in tracks:
        tid = t.get("id")
        f = feats.get(tid)
        s = score_track(f, params)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0])
    return [t for _, t in scored[:top_n]]

@app.route("/recommend", methods=["GET","POST"])
def recommend():
    if "access_token" not in session:
        return redirect(url_for("home"))
    if request.method == "GET":
        return """
        <h2>Mooodyyy 🎵 請輸入一句話描述你的情境</h2>
        <form method="POST">
          <textarea name="text" rows="4" style="width:100%;max-width:720px"
            placeholder="例：下雨的夜晚想聽鋼琴放鬆；或：想專心讀書的純音樂"></textarea><br>
          <button type="submit">生成 Top 10</button>
        </form>
        <p><a href="/welcome">🏠 回首頁</a></p>
        """
    # POST
    text = (request.form.get("text") or "").strip()
    if not text:
        return "請輸入一句話。<a href='/recommend'>返回</a>"
    params = map_text_to_params(text)

    # Spotify 用 timeout，避免卡住
    sp = spotipy.Spotify(auth=session["access_token"], requests_timeout=10)

    # 候選集先用官方 Global Top 50（可用環境變數覆蓋）
    top_id = os.environ.get("GLOBAL_TOP_PLAYLIST_ID", "37i9dQZEVXbMDoHDwVN2tF")
    t0 = time.time()
    tracks = fetch_playlist_tracks(sp, top_id, max_n=100)  # 先抓 100
    ids = [t.get("id") for t in tracks if t.get("id")]
    feats = audio_features_map(sp, ids)
    top10 = select_top(tracks, feats, params, top_n=10)
    dt = time.time() - t0

    if not top10:
        return "暫時挑不到符合的歌，試試別的描述？<a href='/recommend'>返回</a>"

    li = []
    for i, t in enumerate(top10, 1):
        nm = t.get("name","")
        artists = ", ".join(a.get("name","") for a in t.get("artists",[]))
        url = (t.get("external_urls") or {}).get("spotify","#")
        li.append(f"<li>{i:02d}. <a href='{url}' target='_blank'>{artists} — {nm}</a></li>")

    # 兩個建立歌單按鈕（公開/私人）
    html_btn = f"""
    <form method="POST" action="/create_playlist" style="display:inline;margin-right:8px">
      <input type="hidden" name="mode" value="public">
      <input type="hidden" name="text" value="{text}">
      <button type="submit">➕ 建立公開歌單</button>
    </form>
    <form method="POST" action="/create_playlist" style="display:inline">
      <input type="hidden" name="mode" value="private">
      <input type="hidden" name="text" value="{text}">
      <button type="submit">➕ 建立私人歌單</button>
    </form>
    """

    return f"""
    <h2>Top 10 推薦</h2>
    <p>情境：{text}</p>
    <p>（候選集 100 首，花費 {dt:.2f}s）</p>
    <ol>{''.join(li)}</ol>
    {html_btn}
    <p><a href="/recommend">↩︎ 再試一次</a> ｜ <a href="/welcome">🏠 回首頁</a></p>
    """

@app.route("/create_playlist", methods=["POST"])
def create_playlist():
    if "access_token" not in session:
        return redirect(url_for("home"))
    mode = (request.form.get("mode") or "private").strip()  # public/private
    text = (request.form.get("text") or "").strip()
    if not text or mode not in ("public","private"):
        return "參數不完整。<a href='/recommend'>返回</a>"

    sp = spotipy.Spotify(auth=session["access_token"], requests_timeout=10)

    # 和 /recommend 一樣跑一次（簡化：保持一致的 Top10）
    params = map_text_to_params(text)
    top_id = os.environ.get("GLOBAL_TOP_PLAYLIST_ID", "37i9dQZEVXbMDoHDwVN2tF")
    tracks = fetch_playlist_tracks(sp, top_id, max_n=100)
    ids = [t.get("id") for t in tracks if t.get("id")]
    feats = audio_features_map(sp, ids)
    top10 = select_top(tracks, feats, params, top_n=10)
    if not top10:
        return "沒有可加入的歌曲。<a href='/recommend'>返回</a>"

    user_id = sp.current_user()["id"]
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    title = f"Mooodyyy · {ts} UTC"
    desc = f"情境：{text}"

    playlist = sp.user_playlist_create(
        user=user_id,
        name=title,
        public=(mode == "public"),
        description=desc
    )
    sp.playlist_add_items(playlist_id=playlist["id"], items=[t["id"] for t in top10])
    url = (playlist.get("external_urls") or {}).get("spotify","#")

    items_html = []
    for i, t in enumerate(top10, 1):
        nm = t.get("name", "")
        artists = ", ".join(a.get("name","") for a in t.get("artists",[]))
        u = (t.get("external_urls") or {}).get("spotify","#")
        items_html.append(f"<li>{i:02d}. <a href='{u}' target='_blank'>{artists} — {nm}</a></li>")

    return f"""
    <h2>✅ 已建立歌單：<a href="{url}" target="_blank">{title}</a></h2>
    <p>模式：{"公開" if mode=="public" else "私人"}</p>
    <p>情境：{text}</p>
    <h3>曲目：</h3>
    <ol>{''.join(items_html)}</ol>
    <p><a href="/recommend">↩︎ 回推薦頁</a> ｜ <a href="/welcome">🏠 回首頁</a></p>
    """

# （除錯用；需要時保留）
@app.route("/env")
def env_show():
    v = os.environ.get("SPOTIPY_REDIRECT_URI", "<none>")
    return f"[{v}] len={len(v)}", 200

# ---------- 語言判斷（簡易） ----------
_re_zh = re.compile(r"[\u4e00-\u9fff]")
_re_ja = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_re_ko = re.compile(r"[\uac00-\ud7af]")

def detect_lang(s: str) -> set:
    s = s or ""
    langs = set()
    if _re_zh.search(s): langs.add("zh")
    if _re_ja.search(s): langs.add("ja")
    if _re_ko.search(s): langs.add("ko")
    if not langs and re.search(r"[A-Za-z]", s):
        langs.add("en")
    return langs or {"en"}

def track_lang_bucket(track: dict) -> set:
    parts = [track.get("name","")] + [a.get("name","") for a in track.get("artists",[])]
    return detect_lang(" ".join(parts))

# ---------- Spotify 抓資料工具 ----------
def fetch_saved_track_ids(sp, max_n=400):
    ids, offset = [], 0
    while offset < max_n:
        try:
            batch = sp.current_user_saved_tracks(limit=50, offset=offset)
        except Exception as e:
            print("⚠️ saved_tracks fail:", e); break
        items = batch.get("items", [])
        if not items: break
        for it in items:
            tr = (it or {}).get("track") or {}
            tid = tr.get("id")
            if tid: ids.append(tid)
        if not batch.get("next"): break
        offset += 50
    # 去重
    uniq = []
    seen = set()
    for x in ids:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

def fetch_tracks_by_ids(sp, ids):
    tracks = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        try:
            resp = sp.tracks(chunk)
            tracks.extend(resp.get("tracks", []))
        except Exception as e:
            print("⚠️ tracks fail:", e)
    return tracks

def fetch_audio_features(sp, ids):
    feats = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i+100]
        try:
            res = sp.audio_features(chunk)
        except Exception as e:
            print("⚠️ audio_features fail:", e); res = []
        for f in (res or []):
            if f and f.get("id"):
                feats[f["id"]] = f
    return feats

def mood_filter_range(mood):
    presets = {
        "happy": {"energy": (0.6, 1.0), "valence": (0.6, 1.0), "danceability": (0.55, 1.0)},
        "chill": {"energy": (0.1, 0.5), "tempo": (60, 110)},
        "focus": {"energy": (0.25, 0.6), "instrumentalness": (0.15, 1.0), "speechiness": (0.0, 0.35)},
        "sad":   {"energy": (0.0, 0.5), "valence": (0.0, 0.45)}
    }
    return presets.get(mood, presets["chill"])

def pass_range(v, rng):
    lo, hi = rng
    return v is not None and lo <= v <= hi

def match_mood(feat, rule):
    if not feat: return False
    checks = []
    if "energy" in rule:           checks.append(pass_range(feat.get("energy"),          rule["energy"]))
    if "valence" in rule:          checks.append(pass_range(feat.get("valence"),         rule["valence"]))
    if "danceability" in rule:     checks.append(pass_range(feat.get("danceability"),    rule["danceability"]))
    if "speechiness" in rule:      checks.append(pass_range(feat.get("speechiness"),     rule["speechiness"]))
    if "instrumentalness" in rule: checks.append(pass_range(feat.get("instrumentalness"),rule["instrumentalness"]))
    if "tempo" in rule:            checks.append(pass_range(feat.get("tempo"),           rule["tempo"]))
    return all(checks) if checks else True

def filter_by_language(tracks, allow_langs:set):
    if not allow_langs: 
        return tracks
    out = []
    for t in tracks:
        if track_lang_bucket(t) & allow_langs:
            out.append(t)
    return out

def diversify_by_artist(tracks, max_per_artist=2):
    seen, out = {}, []
    for t in tracks:
        artists = t.get("artists", [])
        key = (artists[0].get("id") if artists else t.get("id"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= max_per_artist:
            out.append(t)
    return out

def pick_with_features(tracks, feats, mood, k):
    rule = mood_filter_range(mood)
    good = [t for t in tracks if match_mood(feats.get(t.get("id")), rule)]
    random.shuffle(good)
    return good[:k]

def fill_with_recommendations(sp, have_ids, mood, target_n, seed_pool_ids):
    targets_map = {
        "happy": dict(target_energy=0.8, target_valence=0.8, target_danceability=0.7),
        "chill": dict(target_energy=0.35, target_tempo=90),
        "focus": dict(target_energy=0.45, target_instrumentalness=0.4, target_speechiness=0.2),
        "sad":   dict(target_energy=0.3, target_valence=0.2),
    }
    params = targets_map.get(mood, {})
    out, tries = [], 0
    while len(out) < target_n and tries < 3:
        tries += 1
        seeds = random.sample(seed_pool_ids, k=min(5, len(seed_pool_ids))) if seed_pool_ids else None
        try:
            if seeds:
                rec = sp.recommendations(seed_tracks=seeds[:5], limit=target_n, **params)
            else:
                rec = sp.recommendations(seed_genres=["pop","indie","rock"], limit=target_n, **params)
        except Exception as e:
            print("⚠️ Recommendation failed:", e); rec = {"tracks": []}
        for tr in rec.get("tracks", []) or []:
            tid = tr.get("id")
            if tid and tid not in have_ids:
                out.append(tr); have_ids.add(tid)
            if len(out) >= target_n: break
    return out[:target_n]

# ---------- 登入後主頁 ----------
@app.route("/welcome")
def welcome():
    if "access_token" not in session:
        return redirect(url_for("home"))
    sp = spotipy.Spotify(auth=session["access_token"])
    me = sp.current_user()
    name = me["display_name"]

    # 直接顯示 embedding 的表單
    html = f"""
    <h2>Hello {name} 🎶</h2>
    <p>輸入一段文字情境，我會幫你轉成向量（embedding）：</p>
    <form action="/embed" method="post">
        <textarea name="text" rows="4" cols="50" placeholder="例如：凌晨三點還不想睡"></textarea><br><br>
        <button type="submit">轉換</button>
    </form>
    """
    return html

# ---------- 產生 30 首歌單 ----------
@app.route("/embed", methods=["GET", "POST"])
def embed():
    if request.method == "POST":
        text = request.form.get("text")
        if not text:
            return "請輸入一段文字情境！<br><a href='/embed'>返回</a>"

        try:
            # 呼叫 OpenAI 產生 embedding
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
        except Exception as e:
            return f"❌ 呼叫 OpenAI 失敗：{e}<br><a href='/embed'>返回</a>"

        # 顯示結果（避免太長，只秀前 10 個值）
        preview = ", ".join(f"{x:.6f}" for x in embedding[:10])
        return f"""
        <h2>輸入文字：</h2>
        <p>{text}</p>
        <h2>Embedding 向量</h2>
        <p>維度：{len(embedding)}</p>
        <p>前 10 個數值：</p>
        <code>[{preview}]</code>
        <br><br>
        <a href="/embed">↩️ 再試一次</a>
        <br><a href="/welcome">🏠 回首頁</a>
        """

    # GET 時回傳輸入表單
    return """
        <h2>輸入情境文字，轉成向量（embedding）</h2>
        <form method="post">
          <textarea name="text" rows="4" cols="60" placeholder="例如：凌晨三點還不想睡"></textarea><br><br>
          <button type="submit">產生 Embedding</button>
        </form>
        <p><a href="/welcome">🏠 回首頁</a></p>
    """



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

