from flask import Flask, request, redirect, session, url_for
import os, random, re
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)
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
    sp = get_sp()
    if not sp:
        return redirect(url_for("home"))
    try:
        me = sp.current_user()
        name = me.get("display_name") or "there"
    except Exception as e:
        return f"Token 失效或 Spotify 連線失敗，請回首頁重新登入。<br><a href='/'>回首頁</a>"

    html = f"""
    <h2>Hello {name} 🎶</h2>
    <p>選擇你的情緒（可在網址加語言：<code>&lang=zh&lang=en</code>）：</p>
    <ul>
      <li><a href="/generate?mood=happy">🌞 Happy</a></li>
      <li><a href="/generate?mood=chill">🌙 Chill</a></li>
      <li><a href="/generate?mood=focus">🎯 Focus</a></li>
      <li><a href="/generate?mood=sad">🌧️ Sad</a></li>
    </ul>
    """
    return html

# ---------- 產生 30 首歌單 ----------
@app.route("/generate")
def generate():
    sp = get_sp()
    if not sp:
        return redirect(url_for("home"))

    mood = (request.args.get("mood") or "chill").lower()
    allow_langs = set(request.args.getlist("lang"))  # 例：?lang=zh&lang=en

    # 1) 取使用者資訊
    try:
        me = sp.current_user()
        user_id = me["id"]
    except Exception as e:
        return "❌ Spotify 連線失敗，請回首頁重新登入。<br><a href='/'>回首頁</a>"

    # 2) 撈收藏歌曲 ID（無則提示）
    saved_ids = fetch_saved_track_ids(sp, max_n=400)
    if not saved_ids:
        return "❌ 你還沒有收藏任何歌曲，請先在 Spotify 按幾首喜歡的歌再試一次。<br><a href='/welcome'>返回</a>"

    seed_pool_ids = saved_ids[:]
    saved_tracks = fetch_tracks_by_ids(sp, saved_ids)
    feats = fetch_audio_features(sp, saved_ids)

    # 3) 語言過濾 + 多樣化
    base = saved_tracks
    if allow_langs:
        base = filter_by_language(base, allow_langs)
    random.shuffle(base)
    base = diversify_by_artist(base, max_per_artist=2)

    # 4) 先從收藏依情緒挑
    chosen_tracks = pick_with_features(base, feats, mood, k=30)
    have_ids = set(t.get("id") for t in chosen_tracks if t and t.get("id"))

    # 5) 不足用推薦補齊（也做語言過濾）
    need = 30 - len(chosen_tracks)
    if need > 0:
        rec_tracks = fill_with_recommendations(sp, have_ids, mood, need, seed_pool_ids)
        if allow_langs:
            rec_tracks = filter_by_language(rec_tracks, allow_langs)
        chosen_tracks += rec_tracks
        chosen_tracks = chosen_tracks[:30]
        have_ids = set(t.get("id") for t in chosen_tracks if t and t.get("id"))

    # 兜底：還是不夠就再用收藏補滿
    if len(chosen_tracks) < 30 and base:
        for t in base:
            tid = t.get("id")
            if tid and tid not in have_ids:
                chosen_tracks.append(t); have_ids.add(tid)
                if len(chosen_tracks) >= 30: break

    if not chosen_tracks:
        return ("❌ 你的收藏太少，推薦也補不到合適歌曲。"
                "請先收藏一些喜歡的歌再試一次。<br><a href='/welcome'>返回</a>")

    # 6) 建歌單 + 批次加入
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    pl_name = f"Mooodyyy — {mood.capitalize()} ({ts} UTC)"
    pl_desc = "Mood-based playlist powered by Mooodyyy. Have a good vibe! ✨"
    try:
        playlist = sp.user_playlist_create(user=user_id, name=pl_name, public=True, description=pl_desc)
        pl_id = playlist["id"]
        pl_url = playlist["external_urls"]["spotify"]
    except Exception as e:
        return f"❌ 建立歌單失敗：{e}"

    ids = [t.get("id") for t in chosen_tracks if t and t.get("id")]
    for i in range(0, len(ids), 100):
        try:
            sp.playlist_add_items(pl_id, ids[i:i+100])
        except Exception as e:
            print("⚠️ add_items fail:", e)

    # 7) 回傳結果
    lang_note = f"（語言：{', '.join(sorted(allow_langs))}）" if allow_langs else ""
    lines = []
    for idx, t in enumerate(chosen_tracks, 1):
        title = t.get("name","")
        artists = ", ".join(a.get("name","") for a in t.get("artists",[]))
        lines.append(f"<li>{idx:02d}. {artists} — {title}</li>")
    html = f"""
    <h3>✅ 已建立 30 首歌單：{pl_name} {lang_note}</h3>
    <p><a href="{pl_url}" target="_blank">▶ 在 Spotify 開啟播放清單</a></p>
    <details><summary>查看歌曲清單</summary><ol>{"".join(lines)}</ol></details>
    <p><a href="/welcome">↩︎ 回到情緒選單</a></p>
    """
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

