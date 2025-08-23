@app.route("/")
def home():
    # 如果 Spotify 把 code 帶回根網址，就直接在這裡處理
    code = request.args.get("code")
    if code:
        token_info = oauth().get_access_token(code, as_dict=True)
        session["access_token"] = token_info["access_token"]
        return redirect(url_for("welcome"))

    return '<a href="/login">🔐 Login with Spotify</a>'


@app.route("/callback")
def callback():
    # 備用方案，Spotify 也可能導回 /callback
    code = request.args.get("code")
    if not code:
        return redirect(url_for("home"))

    token_info = oauth().get_access_token(code, as_dict=True)
    session["access_token"] = token_info["access_token"]
    return redirect(url_for("welcome"))
