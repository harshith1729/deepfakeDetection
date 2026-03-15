from flask import Flask, render_template, request, redirect
from .twilio_alert import send_alert

app = Flask(__name__)

@app.route("/")
def login():
    return render_template("login.html")


@app.route("/dashboard", methods=["POST"])
def dashboard():

    username = request.form["username"]
    password = request.form["password"]

    if username == "admin" and password == "1234":
        return render_template("dashboard.html")

    return "Invalid login"


@app.route("/simulate_attack")
def simulate_attack():

    send_alert()
    return "Alert Sent!"


@app.route("/config")
def config():
    return render_template("config.html")


if __name__ == "__main__":
    app.run(debug=True)