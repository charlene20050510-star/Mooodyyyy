from flask import Flask, request, redirect, session, url_for
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "defaultsecret")

# Spotify OAuth 設定
def oauth():
    return SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ["SPOTIPY_REDIRECT_URI"],
        scope="user-library-read playlist-modify-public"
    )

@app.route("/")
def home():
    # 如果 Spotify 把 code 帶回根網址，直接處理
    code = request.args.get("code")
    if code:
        token_info = oauth().get_access_token(code, as_dict=True)
        session["access_token"] = token_info["access_token"]
        return redirect(url_for("welcome"))

    return '<a href="/login">🔐 Login with Spotify</a>'

@app.route("/login")
def login():
    auth_url = oauth().get_authorize_url()
    return redirect(auth_url)

@app.route("/callback")
def callback():
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
    user = sp.current_user()
    return f"Hello {user['display_name']} 🎶, welcome to Mooodyyy!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

