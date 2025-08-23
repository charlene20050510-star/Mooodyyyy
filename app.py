from flask import Flask, redirect, request, session, url_for
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "devkey")

CLIENT_ID = os.environ["SPOTIPY_CLIENT_ID"]
CLIENT_SECRET = os.environ["SPOTIPY_CLIENT_SECRET"]
REDIRECT_URI = os.environ["SPOTIPY_REDIRECT_URI"]  # https://mooodyyy.onrender.com/callback
SCOPE = "user-library-read playlist-modify-public playlist-modify-private"

def oauth():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=None
    )

@app.route("/")
def home():
    return '<a href="/login">🔐 Login with Spotify</a>'

@app.route("/login")
def login():
    return redirect(oauth().get_authorize_url())

@app.route("/callback")
def callback():
    code = request.args.get("code")
    token_info = oauth().get_access_token(code, as_dict=True)
    session["access_token"] = token_info["access_token"]
    return redirect(url_for("welcome"))

@app.route("/welcome")
def welcome():
    token = session.get("access_token")
    if not token:
        return redirect("/login")
    sp = spotipy.Spotify(auth=token)
    me = sp.current_user()
    return f"Hello {me['display_name']} 🎶, welcome to Mooodyyy!"
