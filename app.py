from flask import Flask, request, redirect, session, url_for
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "devsecret")

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

# 登入成功後的歡迎頁
@app.route("/welcome")
def welcome():
    if "access_token" not in session:
        return redirect(url_for("home"))
    sp = spotipy.Spotify(auth=session["access_token"])
    me = sp.current_user()
    return f"Hello {me['display_name']} 🎶, welcome to Mooodyyy!"

# 簡單健康檢查
@app.route("/ping")
def ping():
    return "PING OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
