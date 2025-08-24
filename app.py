from flask import Flask, request, redirect, session, url_for
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "devsecret")

@app.route("/env")
def env_show():
    v = os.environ.get("SPOTIPY_REDIRECT_URI", "<none>")
    # 用方括號標示開頭/結尾，順便顯示長度，抓隱藏空白/換行
    return f"[{v}] len={len(v)}", 200

@app.route("/login_debug")
def login_debug():
    url = oauth().get_authorize_url()
    return f'<a href="{url}">{url}</a>', 200


# 權限範圍：讀取歌庫、建立歌單
SCOPE = "user-library-read playlist-modify-public playlist-modify-private"

def oauth():
    return SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ["SPOTIPY_REDIRECT_URI"],  # 必須和 Spotify Dashboard 完全一致
        scope=SCOPE,
        cache_path=None,
        open_browser=False,
        show_dialog=True
    )

# 首頁：顯示登入連結
@app.route("/")
def home():
    return '<a href="/login">🔐 Login with Spotify</a>'

# 跳轉到 Spotify 授權頁
@app.route("/login")
def login():
    return redirect(oauth().get_authorize_url())

# Spotify 登入完成後會回到這裡（callback）
@app.route("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return redirect(url_for("home"))
    token_info = oauth().get_access_token(code, as_dict=True)
    session["access_token"] = token_info["access_token"]
    return redirect(url_for("welcome"))

# 追加在檔案上方 imports 之後
import math, random, re
from datetime import datetime

def get_sp():
    if "access_token" not in session:
        return None
    return spotipy.Spotify(auth=session["access_token"])

# ——— 語言偵測（簡易啟發式） ———
_re_zh = re.compile(r"[\u4e00-\u9fff]")          # 中日韓統一表意文字（中文最常用）
_re_ja = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")  # 平假名/片假名/小書寫
_re_ko = re.compile(r"[\uac00-\ud7af]")          # 韓文音節

def detect_lang(s: str) -> set:
    s = s or ""
    langs = set()
    if _re_zh.search(s): langs.add("zh")
    if _re_ja.search(s): langs.add("ja")
    if _re_ko.search(s): langs.add("ko")
    # 英文：若完全沒有 CJK，且含拉丁字母，就視為 en
    if not (langs) and re.search(r"[A-Za-z]", s):
        langs.add("en")
    return langs or {"en"}

def track_lang_bucket(track: dict) -> set:
    # 綜合歌名與所有藝人名稱
    parts = [track.get("name","")] + [a.get("name","") for a in track.get("artists",[])]
    joined = " ".join(parts)
    return detect_lang(joined)

# ——— 撈收藏歌、特徵、過濾 ———
def fetch_saved_track_ids(sp, max_n=400):
    ids = []
    offset = 0
    while offset < max_n:
        batch = sp.current_user_saved_tracks(limit=50, offset=offset)
        items = batch.get("items", [])
        if not items: break
        for it in items:
            tr = (it or {}).get("track") or {}
            tid = tr.get("id")
            if tid: ids.append(tid)
        if not batch.get("next"): break
        offset += 50
    return list(dict.fromkeys(ids))  # 去重

def fetch_tracks_by_ids(sp, ids):
    tracks = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        resp = sp.tracks(chunk)
        tracks.extend(resp.get("tracks", []))
    return tracks

def fetch_audio_features(sp, ids):
    feats = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i+100]
        res = sp.audio_features(chunk)
        for f in (res or []):
            if f and f.get("id"):
                feats[f["id"]] = f
    return feats

def mood_filter_range(mood):
    # 回傳對 audio features 的篩選條件（最小可用，之後可微調）
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
    # 能量/快樂度/舞動/語音度/樂器性/速度
    checks = []
    if "energy" in rule:          checks.append(pass_range(feat.get("energy"),          rule["energy"]))
    if "valence" in rule:         checks.append(pass_range(feat.get("valence"),         rule["valence"]))
    if "danceability" in rule:    checks.append(pass_range(feat.get("danceability"),    rule["danceability"]))
    if "speechiness" in rule:     checks.append(pass_range(feat.get("speechiness"),     rule["speechiness"]))
    if "instrumentalness" in rule:checks.append(pass_range(feat.get("instrumentalness"),rule["instrumentalness"]))
    if "tempo" in rule:           checks.append(pass_range(feat.get("tempo"),           rule["tempo"]))
    return all(checks) if checks else True

def filter_by_language(tracks, allow_langs:set):
    if not allow_langs: 
        return tracks
    out = []
    for t in tracks:
        langs = track_lang_bucket(t)
        if langs & allow_langs:
            out.append(t)
    return out

def diversify_by_artist(tracks, max_per_artist=2):
    seen = {}
    out = []
    for t in tracks:
        artists = t.get("artists", [])
        key = (artists[0].get("id") if artists else t.get("id"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= max_per_artist:
            out.append(t)
    return out

def pick_with_features(tracks, feats, mood, k):
    rule = mood_filter_range(mood)
    # 先依符合程度排序（完全符合>部分符合>不符合）；此處先做嚴格 all-check
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
    out = []
    tries = 0
    while len(out) < target_n and tries < 3:
        tries += 1
        seeds = random.sample(seed_pool_ids, k=min(5, len(seed_pool_ids))) if seed_pool_ids else None
        rec = None
        try:
            if seeds:
                rec = sp.recommendations(seed_tracks=seeds, limit=target_n, **params)
            else:
                rec = sp.recommendations(seed_genres=["pop","indie","rock"], limit=target_n, **params)
        except Exception:
            rec = {"tracks":[]}
        for tr in rec.get("tracks", []):
            tid = tr.get("id")
            if tid and tid not in have_ids:
                out.append(tr)
                have_ids.add(tid)
            if len(out) >= target_n:
                break
    return out[:target_n]

# ——— 覆蓋 /welcome：顯示基本入口 ———
@app.route("/welcome")
def welcome():
    sp = get_sp()
    if not sp:
        return redirect(url_for("home"))
    me = sp.current_user()
    name = me.get("display_name") or "there"
    html = f"""
    <h2>Hello {name} 🎶</h2>
    <p>選擇你的情緒與語言，讓 Mooodyyy 幫你生成 30 首歌的播放清單：</p>
    <ul>
      <li><a href="/generate?mood=happy">🌞 Happy</a></li>
      <li><a href="/generate?mood=chill">🌙 Chill</a></li>
      <li><a href="/generate?mood=focus">🎯 Focus</a></li>
      <li><a href="/generate?mood=sad">🌧️ Sad</a></li>
    </ul>
    <p>（可在網址後面加語言參數，如 <code>&lang=zh&lang=en</code>）</p>
    """
    return html

# ——— 覆蓋 /generate：產生 30 首歌單 ———
@app.route("/generate")
def generate():
    sp = get_sp()
    if not sp:
        return redirect(url_for("home"))

    mood = request.args.get("mood", "chill").lower()
    allow_langs = set(request.args.getlist("lang"))  # 例：['zh','en']

    me = sp.current_user()
    user_id = me["id"]

    # 1) 撈收藏 + 取特徵
    saved_ids = fetch_saved_track_ids(sp, max_n=400)
    seed_pool_ids = saved_ids[:]  # 之後推薦要用
    saved_tracks = fetch_tracks_by_ids(sp, saved_ids) if saved_ids else []
    feats = fetch_audio_features(sp, saved_ids) if saved_ids else {}

    # 2) 語言過濾 + 情緒過濾 + 去重 + 多樣化
    base = saved_tracks
    if allow_langs:
        base = filter_by_language(base, allow_langs)
    random.shuffle(base)                # 打散
    base = diversify_by_artist(base)    # 同藝人上限 2 首

    # 3) 依情緒挑選，先從收藏裡挑
    chosen_tracks = pick_with_features(base, feats, mood, k=30)
    have_ids = set(t.get("id") for t in chosen_tracks if t.get("id"))

    # 4) 不足 → 用推薦補齊
    need = 30 - len(chosen_tracks)
    if need > 0:
        rec_tracks = fill_with_recommendations(sp, have_ids, mood, need, seed_pool_ids)
        # 語言過濾（補來的也過一下語言條件）
        if allow_langs:
            rec_tracks = filter_by_language(rec_tracks, allow_langs)
        chosen_tracks += rec_tracks
        chosen_tracks = chosen_tracks[:30]
        have_ids = set(t.get("id") for t in chosen_tracks if t.get("id"))

    # 兜底：還是不夠就直接從收藏補滿
    if len(chosen_tracks) < 30 and base:
        for t in base:
            tid = t.get("id")
            if tid and tid not in have_ids:
                chosen_tracks.append(t)
                have_ids.add(tid)
                if len(chosen_tracks) >= 30: break

    if not chosen_tracks:
        return ("你的收藏太少，且推薦也補不到合適的歌曲。"
                "建議先在 Spotify 收藏一些喜歡的歌，再試一次。")

    # 5) 建立歌單 + 寫入歌曲
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    name = f"Mooodyyy — {mood.capitalize()} ({ts} UTC)"
    desc = "Mood-based playlist powered by Mooodyyy. Have a good vibe! ✨"
    playlist = sp.user_playlist_create(user=user_id, name=name, public=True, description=desc)
    pl_id = playlist["id"]
    pl_url = playlist["external_urls"]["spotify"]

    # 批次加入
    track_ids = [t["id"] for t in chosen_tracks if t.get("id")]
    for i in range(0, len(track_ids), 100):
        sp.playlist_add_items(pl_id, track_ids[i:i+100])

    # 6) 回傳結果
    lang_note = f"（語言：{', '.join(sorted(allow_langs))}）" if allow_langs else ""
    lines = []
    for idx, t in enumerate(chosen_tracks, 1):
        title = t.get("name","")
        artists = ", ".join(a.get("name","") for a in t.get("artists",[]))
        lines.append(f"<li>{idx:02d}. {artists} — {title}</li>")
    html = f"""
    <h3>✅ 已建立 30 首歌單：{name} {lang_note}</h3>
    <p><a href="{pl_url}" target="_blank">▶ 在 Spotify 開啟播放清單</a></p>
    <details><summary>查看歌曲清單</summary><ol>{"".join(lines)}</ol></details>
    <p><a href="/welcome">↩︎ 回到情緒選單</a></p>
    """
    return html


# 簡單健康檢查
@app.route("/ping")
def ping():
    return "PING OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
