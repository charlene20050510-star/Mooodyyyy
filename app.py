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
    """從指定歌單抓歌 - 修復版本"""
    tracks = []
    offset = 0
    while len(tracks) < max_n:
        try:
            # 加上 market="from_token" 避免地區限制
            batch = sp.playlist_items(playlist_id, offset=offset, limit=50, market="from_token")
            items = batch.get("items", [])
            if not items: 
                break
            for it in items:
                tr = (it or {}).get("track") or {}
                if tr.get("id") and tr.get("is_playable", True):
                    tracks.append(tr)
                if len(tracks) >= max_n: 
                    break
            if not batch.get("next"): 
                break
            offset += 50
        except Exception as e:
            print(f"⚠️ 抓取歌單失敗: {e}")
            # 如果失敗，嘗試備用歌單
            if playlist_id != "37i9dQZF1DXcBWIGoYBM5M":
                return fetch_playlist_tracks(sp, "37i9dQZF1DXcBWIGoYBM5M", max_n)
            break
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

@app.route("/welcome")
def welcome():
    if "access_token" not in session:
        return redirect(url_for("home"))
    
    try:
        sp = spotipy.Spotify(auth=session["access_token"], requests_timeout=10)
        me = sp.current_user()
        name = me.get("display_name", "音樂愛好者")
    except Exception as e:
        print(f"⚠️ 獲取用戶信息失敗: {e}")
        name = "音樂愛好者"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Mooodyyy - AI 音樂情境推薦</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #1DB954, #1ed760);
                color: white;
                padding: 20px;
                min-height: 100vh;
                margin: 0;
            }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 40px 20px; }}
            .card {{ 
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 16px;
                backdrop-filter: blur(10px);
                margin: 20px 0;
            }}
            textarea {{
                width: 100%;
                padding: 15px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                resize: vertical;
                box-sizing: border-box;
            }}
            button {{
                background: #FF6B6B;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 10px;
            }}
            button:hover {{ background: #ff5252; }}
            .secondary {{ background: rgba(255,255,255,0.2); }}
            .secondary:hover {{ background: rgba(255,255,255,0.3); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎵 Hello {name}</h1>
            <p>歡迎來到 Mooodyyy - 讓 AI 理解你的音樂情境</p>
            
            <div class="card">
                <h2>🎯 情境推薦</h2>
                <p>用一句話描述你現在的心情或場景：</p>
                <form action="/recommend" method="post">
                    <textarea name="text" rows="4" 
                              placeholder="例如：下雨天想聽輕柔的鋼琴曲放鬆..."></textarea>
                    <br>
                    <button type="submit">生成專屬歌單</button>
                </form>
            </div>
            
            <div class="card">
                <h2>🤖 技術展示</h2>
                <p>想看看 AI 如何理解文字？試試 Embedding 轉換：</p>
                <form action="/embed" method="post">
                    <textarea name="text" rows="2" 
                              placeholder="例如：凌晨三點還不想睡..."></textarea>
                    <br>
                    <button type="submit" class="secondary">查看 AI 向量</button>
                </form>
            </div>
            
            <p style="text-align: center; margin-top: 40px; opacity: 0.8;">
                <a href="/logout" style="color: white;">登出</a>
            </p>
        </div>
    </body>
    </html>
    """

@app.route("/recommend", methods=["GET","POST"])
def recommend():
    if "access_token" not in session:
        return redirect(url_for("home"))
    
    # 處理輸入
    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
    else:
        text = (request.args.get("text") or "").strip()
    
    if not text:
        return redirect(url_for("welcome"))
    
    try:
        # Spotify 連接
        sp = spotipy.Spotify(auth=session["access_token"], requests_timeout=15)
        
        # 多個備用歌單 ID
        candidate_playlists = [
            "37i9dQZF1DXcBWIGoYBM5M",  # Today's Top Hits
            "37i9dQZEVXbMDoHDwVN2tF",  # Global Top 50
        ]
        
        tracks = []
        for playlist_id in candidate_playlists:
            try:
                tracks = fetch_playlist_tracks(sp, playlist_id, max_n=100)
                if tracks:
                    print(f"✅ 成功從歌單 {playlist_id} 獲取 {len(tracks)} 首歌")
                    break
            except Exception as e:
                print(f"⚠️ 歌單 {playlist_id} 失敗: {e}")
                continue
        
        if not tracks:
            return f"""
            <h2>❌ 暫時無法獲取歌曲</h2>
            <p>Spotify API 暫時不可用，請稍後再試。</p>
            <a href="/welcome">回首頁</a>
            """
        
        # 使用現有的推薦邏輯
        t0 = time.time()
        params = map_text_to_params(text)
        
        ids = [t.get("id") for t in tracks if t.get("id")]
        feats = audio_features_map(sp, ids)
        top10 = select_top(tracks, feats, params, top_n=10)
        dt = time.time() - t0
        
        if not top10:
            return f"""
            <h2>😅 找不到符合的歌曲</h2>
            <p>試試用不同的描述方式，比如：</p>
            <ul>
                <li>"想要有活力的音樂"</li>
                <li>"適合讀書的輕音樂"</li>
                <li>"傷心時聽的歌"</li>
            </ul>
            <a href="/welcome">重新嘗試</a>
            """
        
        # 建立歌單按鈕
        buttons_html = f"""
        <div style="margin: 20px 0;">
            <form method="POST" action="/create_playlist" style="display:inline; margin-right:10px;">
                <input type="hidden" name="mode" value="public">
                <input type="hidden" name="text" value="{text}">
                <button type="submit" style="background:#1DB954; color:white; border:none; padding:10px 20px; border-radius:6px;">
                    ➕ 建立公開歌單
                </button>
            </form>
            <form method="POST" action="/create_playlist" style="display:inline;">
                <input type="hidden" name="mode" value="private">
                <input type="hidden" name="text" value="{text}">
                <button type="submit" style="background:#FF6B6B; color:white; border:none; padding:10px 20px; border-radius:6px;">
                    ➕ 建立私人歌單
                </button>
            </form>
        </div>
        """
        
        # 歌曲清單
        songs_html = []
        for i, track in enumerate(top10, 1):
            name = track.get("name", "Unknown")
            artists = ", ".join([a.get("name", "") for a in track.get("artists", [])])
            url = (track.get("external_urls") or {}).get("spotify", "#")
            songs_html.append(f"""
                <li style="margin: 8px 0; line-height: 1.4;">
                    {i:02d}. <a href="{url}" target="_blank" style="color: #1DB954; text-decoration: none;">
                        <strong>{artists}</strong> - {name}
                    </a>
                </li>
            """)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>推薦結果 - Mooodyyy</title>
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    background: linear-gradient(135deg, #1DB954, #1ed760);
                    color: white;
                    padding: 20px;
                    margin: 0;
                }}
                .container {{ max-width: 700px; margin: 0 auto; }}
                .result-box {{ 
                    background: rgba(255,255,255,0.1);
                    padding: 30px;
                    border-radius: 16px;
                    backdrop-filter: blur(10px);
                }}
                ol {{ padding-left: 0; }}
                li {{ list-style: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="result-box">
                    <h1>🎯 為你找到了 {len(top10)} 首歌</h1>
                    <p><strong>你的情境：</strong>"{text}"</p>
                    <p style="opacity: 0.8;">從 {len(tracks)} 首候選歌曲中篩選，耗時 {dt:.1f} 秒</p>
                    
                    <h2>🎵 推薦歌單：</h2>
                    <ol>
                        {''.join(songs_html)}
                    </ol>
                    
                    {buttons_html}
                    
                    <p style="margin-top: 30px;">
                        <a href="/welcome" style="color: white;">↩️ 回首頁</a> | 
                        <a href="/recommend" style="color: white;">🔄 重新推薦</a>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
    except Exception as e:
        print(f"❌ 系統錯誤: {e}")
        return f"""
        <h2>❌ 系統暫時出錯</h2>
        <p>錯誤訊息：{str(e)}</p>
        <p>請回首頁重試，或聯繫技術支援。</p>
        <a href="/welcome">回首頁</a>
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
    candidate_playlists = [
        "37i9dQZF1DXcBWIGoYBM5M",  # Today's Top Hits
        "37i9dQZEVXbMDoHDwVN2tF",  # Global Top 50
    ]
    
    tracks = []
    for playlist_id in candidate_playlists:
        try:
            tracks = fetch_playlist_tracks(sp, playlist_id, max_n=100)
            if tracks:
                break
        except Exception as e:
            print(f"⚠️ 歌單 {playlist_id} 失敗: {e}")
            continue
    
    if not tracks:
        return "沒有可加入的歌曲。<a href='/recommend'>返回</a>"
    
    ids = [t.get("id") for t in tracks if t.get("id")]
    feats = audio_features_map(sp, ids)
    top10 = select_top(tracks, feats, params, top_n=10)
    
    if not top10:
        return "沒有可加入的歌曲。<a href='/recommend'>返回</a>"

    user_id = sp.current_user()["id"]
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

# ---------- Embedding 功能 ----------
@app.route("/embed", methods=["GET", "POST"])
def embed():
    if "access_token" not in session:
        return redirect(url_for("home"))
        
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

# 添加登出功能
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ---------- 其他保留的功能 ----------
@app.route("/env")
def env_show():
    v = os.environ.get("SPOTIPY_REDIRECT_URI", "<none>")
    return f"[{v}] len={len(v)}", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
