"""
JWT authentication middleware for Flask routes.

Validates Supabase JWT tokens from the Authorization header and
attaches user info to Flask's `g` context for downstream handlers.
"""
from functools import wraps
from flask import request, jsonify, g
from utils.supabase_client import verify_jwt


def require_auth(f):
    """
    Decorator that enforces authentication on a route.

    Expects: `Authorization: Bearer <supabase_jwt>` header.
    Sets `g.user_id` and `g.user_email` on success.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid authorization header"}), 401

        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            return jsonify({"error": "Empty token"}), 401

        user = verify_jwt(token)
        if not user:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.user = user
        g.user_id = str(user.id)
        g.user_email = getattr(user, "email", "")
        return f(*args, **kwargs)

    return decorated
