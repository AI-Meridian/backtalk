"""The signed-out preflight (backtalk/auth.py): pure parsing and the
error classifier. No subprocess, no CLI needed."""
import json

from backtalk import auth


def test_parse_logged_in_shape():
    out = auth.parse_status(json.dumps({
        "loggedIn": True, "authMethod": "claude.ai",
        "email": "someone@example.com", "subscriptionType": "pro"}), 0)
    assert out["logged_in"] is True
    assert out["email"] == "someone@example.com"
    assert out["plan"] == "pro"
    assert out["method"] == "claude.ai"
    assert out["error"] == ""


def test_parse_logged_out_shape():
    out = auth.parse_status(json.dumps({"loggedIn": False}), 1)
    assert out["logged_in"] is False
    assert out["email"] == ""
    assert out["error"] == ""        # exit 1 on signed-out is normal


def test_parse_garbage_reads_as_signed_out_with_reason():
    out = auth.parse_status("Segmentation fault", 139)
    assert out["logged_in"] is False
    assert "unreadable" in out["error"]
    assert "139" in out["error"]


def test_parse_empty_stdout():
    out = auth.parse_status("", 0)
    assert out["logged_in"] is False


def test_parse_non_object_json():
    out = auth.parse_status("[1,2,3]", 0)
    assert out["logged_in"] is False
    assert "shape" in out["error"]


def test_auth_error_classifier_matches_cli_wordings():
    for s in ("Not logged in · Please run /login",
              "Error: 401 Invalid API key · Please run /login",
              "OAuth token has expired (run /login to renew)",
              "authentication failed",
              "Authentication error · Try again"):
        assert auth.is_auth_error(s), s


def test_auth_error_classifier_ignores_ordinary_errors():
    for s in ("", None, "You've hit your usage limit",
              "overloaded_error: 529", "Internal server error",
              "File not found"):
        assert not auth.is_auth_error(s), s


def test_find_claude_returns_path_or_none():
    found = auth.find_claude()
    assert found is None or isinstance(found, str)
