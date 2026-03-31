
from flask import Flask
from flask_cors import CORS

# Import feature routes
from features.object_detection.routes import object_bp
from features.currency_detection.routes import currency_bp
from features.text_reading.routes import text_bp


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Register feature APIs
app.register_blueprint(object_bp)
app.register_blueprint(currency_bp)
app.register_blueprint(text_bp)


@app.route("/")
def home():
    return "VoiceVision AI Backend Running"


if __name__ == "__main__":
    app.run(host="10.188.82.22", port=5000)