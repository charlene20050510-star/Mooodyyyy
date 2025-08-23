from flask import Flask, request, redirect, session, url_for
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "devsecret")

SCOPE = "user-library-read playlist-modify-public playlist-modify-private"

def oauth():
    return SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ["SPOTIPY_REDIRECT_URI"],  # 我們會設成根網址
        scope=SCOPE,
        cache_path=None,
        open_browser=False,
        show_dialog=True
    )

@app.route("/ping")
def ping():
    return "ok", 200

@app.route("/")
def home():
    # 若 Spotify 授權後把 code 帶回根網址，這裡直接處理
    code = request.args.get("code")
    if code:
        token_info = oauth().get_access_token(code, as_dict=True)
        session["access_token"] = token_info["access_token"]
        return redirect(url_for("welcome"))
    return '<a href="/login">🔐 Login with Spotify</a>'

@app.route("/login")
def login():
    return redirect(oauth().get_authorize_url())

@app.route("/callback")
def callback():
    # 備用路由：如果 Redirect URI 用 /callback 一樣能接住
    code = request.args.get("code")
    if not code:
        return redirect(url_for("home"))
    token_info = oauth().get_access_token(code, as_dict=True)
    session["access_token"] = token_info["access_token"]
    return redirect(url_for("welcome"))

@app.route("/welcome")
def welcome():
    if "access_token" not in session:
        return redirect(url_for("home"))
    sp = spotipy.Spotify(auth=session["access_token"])
    me = sp.current_user()
    return f"Hello {me['display_name']} 🎶, welcome to Mooodyyy!"

if __name__ == "__main__":
    # 本地測試用；雲端會用 gunicorn
    app.run(host="0.0.0.0", port=5000)
