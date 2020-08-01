from app import app
from flask import render_template

@app.route("/")      #default page will be
def index():
    return render_template("index.html")