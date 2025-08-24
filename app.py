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

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "GET":
        return """
        <h2>Mooodyyy：用一句話描述現在的情境</h2>
        <form method="POST">
          <textarea name="text" rows="4" style="width:100%;max-width:720px" placeholder="例如：下著雨的凌晨兩點，想聽一點鋼琴讓自己安靜下來"></textarea>
          <br><button type="submit">送出</button>
        </form>
        <p><a href="/welcome">🏠 回首頁</a></p>
        """
        # 在大眾 Top10 清單 parts.append(...) 後面，加：
parts.append(f"""
<form method="POST" action="/create_playlist">
  <input type="hidden" name="mode" value="public">
  <input type="hidden" name="text" value="{text}">
  <button type="submit">➕ 建立「大眾 Top10」歌單到我的 Spotify</button>
</form>
""")

# 在我的曲庫 Top10 清單 parts.append(...) 後面，加：
parts.append(f"""
<form method="POST" action="/create_playlist">
  <input type="hidden" name="mode" value="personal">
  <input type="hidden" name="text" value="{text}">
  <button type="submit">➕ 建立「我的曲庫 Top10」歌單到我的 Spotify</button>
</form>
""")

    # POST：使用者送出後
    text = (request.form.get("text") or "").strip()
    if not text:
        return "請輸入一句話描述情境。<br><a href='/recommend'>返回</a>"

    return f"<h3>你剛剛輸入的文字：</h3><p>{text}</p><p><a href='/recommend'>↩︎ 再試一次</a></p>"

@app.route("/create_playlist", methods=["POST"])
def create_playlist():
    # 1) 權限與參數
    if "access_token" not in session:
        return redirect(url_for("home"))
    sp = spotipy.Spotify(auth=session["access_token"])

    mode = (request.form.get("mode") or "").strip()      # "public" 或 "personal"
    text = (request.form.get("text") or "").strip()
    if mode not in ("public", "personal") or not text:
        return "參數不完整。<a href='/recommend'>返回</a>"

    # 2) 把文字轉成音樂參數（你前面已經貼過 map_text_to_params）
    params = map_text_to_params(text)

    # 3) 準備候選歌曲
    tracks = []
    if mode == "public":
        # 用 Spotify 官方 Global Top 50 歌單（或用你的環境變數覆蓋）
        top_id = os.environ.get("GLOBAL_TOP_PLAYLIST_ID", "37i9dQZEVXbMDoHDwVN2tF")
        tracks = fetch_playlist_tracks(sp, top_id, max_n=150)
    else:  # personal
        # 抓你按過 ❤️ 的 Liked Songs（最多 300 首）
        saved_ids, offset = [], 0
        while len(saved_ids) < 300:
            try:
                batch = sp.current_user_saved_tracks(limit=50, offset=offset)
            except Exception as e:
                print("⚠️ saved_tracks fail:", e); break
            items = batch.get("items", [])
            if not items: break
            for it in items:
                tr = (it or {}).get("track") or {}
                tid = tr.get("id")
                if tid: saved_ids.append(tid)
            if not batch.get("next"): break
            offset += 50

        if not saved_ids:
            return "你的曲庫目前沒有收藏歌曲，無法建立個人歌單。<a href='/recommend'>返回</a>"

        # 轉回完整 track 物件
        for i in range(0, len(saved_ids), 50):
            chunk = saved_ids[i:i+50]
            try:
                resp = sp.tracks(chunk)
                tracks.extend(resp.get("tracks", []))
            except Exception as e:
                print("⚠️ tracks fail:", e)

    if not tracks:
        return "找不到候選歌曲，無法建立歌單。<a href='/recommend'>返回</a>"

    # 4) 取出音樂特徵並評分，挑前 10
    ids = [t.get("id") for t in tracks if t.get("id")]
    feats = audio_features_map(sp, ids)
    top10 = select_top(tracks, feats, params, top_n=10)

    if not top10:
        return "沒有符合條件的歌曲。<a href='/recommend'>返回</a>"

    # 5) 建立 Spotify 歌單 + 加入歌曲
    user_id = sp.current_user()["id"]
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    title = f"Mooodyyy · {('大眾Top10' if mode=='public' else '我的曲庫Top10')} · {ts} UTC"
    desc = f"情境：{text} ｜ 參數：energy {params['target_energy']}, valence {params['target_valence']}, tempo {params['target_tempo']}"

    # public 模式設公開，personal 設私人（你可以反過來，隨你）
    is_public = True if mode == "public" else False
    playlist = sp.user_playlist_create(user=user_id, name=title, public=is_public, description=desc)

    track_ids = [t["id"] for t in top10 if t.get("id")]
    # 一次最多 100 首，這裡只有 10 首，直接加
    sp.playlist_add_items(playlist_id=playlist["id"], items=track_ids)

    playlist_url = (playlist.get("external_urls") or {}).get("spotify", "#")

    # 6) 回應頁面
    items_html = []
    for i, t in enumerate(top10, 1):
        nm = t.get("name", "")
        artists = ", ".join(a.get("name","") for a in t.get("artists",[]))
        url = (t.get("external_urls") or {}).get("spotify","#")
        items_html.append(f"<li>{i:02d}. <a href='{url}' target='_blank'>{artists} — {nm}</a></li>")

    return f"""
    <h2>✅ 成功建立歌單！</h2>
    <p>歌單：<a href="{playlist_url}" target="_blank">{title}</a></p>
    <p>模式：{"大眾 Top10" if mode=="public" else "我的曲庫 Top10"}</p>
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

