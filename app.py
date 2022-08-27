from flask import Flask

app = Flask(__name__)

@app.route("/")

def home():
    return "Flash App"

if __name__ == '__main__':
    app.run()