import os
from flask import Flask, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    CORS(app, origins="*", supports_credentials=True)

    # Register blueprints
    from api.documents import documents_bp
    from api.chat import chat_bp

    app.register_blueprint(documents_bp, url_prefix="/api/documents")
    app.register_blueprint(chat_bp, url_prefix="/api/chat")

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_PUBLISHABLE_KEY", ""),
        )

    @app.route("/health")
    def health():
        return {"status": "ok", "version": "1.0.0"}

    @app.route("/api/warmup")
    def warmup():
        """Pre-load embedding model to avoid cold-start delay on first chat."""
        try:
            from utils.embedder import embed_texts
            embed_texts(["warmup"])
            return {"status": "warm", "model": "all-MiniLM-L6-v2"}
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
