"""Unit tests for Frontend UI, Timeline Scrubber & Camera Controller assets."""

from fastapi.testclient import TestClient
from src.server.app import app

client = TestClient(app)

def test_index_html_ui_components():
    """Verifies that index.html contains all required UI elements, controls, and drawers."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # Top Navigation Bar elements
    assert "FRC 2026 REBUILT" in html
    assert "3D Match Digital Twin" in html or "3D Digital Twin" in html
    assert 'id="match-select"' in html
    assert 'id="status-indicator"' in html
    assert 'id="fps-counter"' in html
    assert 'id="frame-counter"' in html

    # Camera Preset Buttons
    assert 'id="preset-free"' in html
    assert 'id="preset-topdown"' in html
    assert 'id="preset-broadcast"' in html
    assert 'id="preset-redwall"' in html
    assert 'id="preset-bluewall"' in html
    assert 'id="preset-follow"' in html
    assert 'id="btn-help"' in html

    # Right Robot Inspector Drawer
    assert 'id="robot-inspector"' in html
    assert 'id="inspector-team-number"' in html
    assert 'id="inspector-alliance-badge"' in html
    assert 'id="inspector-speed-ips"' in html
    assert 'id="inspector-speed-mph"' in html
    assert 'id="inspector-coords"' in html
    assert 'id="inspector-heading-deg"' in html
    assert 'id="inspector-action-badge"' in html
    assert 'id="btn-focus-robot"' in html
    assert 'id="btn-deselect-robot"' in html
    assert 'id="btn-close-inspector"' in html

    # Bottom Playback Dock
    assert 'id="btn-play-pause"' in html
    assert 'id="btn-step-back-10"' in html
    assert 'id="btn-step-back-1"' in html
    assert 'id="btn-step-fwd-1"' in html
    assert 'id="btn-step-fwd-10"' in html
    assert 'data-speed="0.25"' in html
    assert 'data-speed="0.5"' in html
    assert 'data-speed="1.0"' in html
    assert 'data-speed="2.0"' in html
    assert 'data-speed="4.0"' in html
    assert 'id="readout-frame"' in html
    assert 'id="readout-time"' in html
    assert 'id="readout-period"' in html

    # Timeline Scrubber & ActionFormer event markers
    assert 'id="timeline-slider"' in html
    assert 'id="timeline-fill"' in html
    assert 'id="timeline-markers"' in html
    assert 'id="timeline-tooltip"' in html

    # Keyboard Shortcuts Modal
    assert 'id="shortcuts-modal"' in html
    assert "Keyboard Shortcuts" in html
    assert "Play / Pause" in html

    # Vendor & Module Scripts
    assert "/static/vendor/three.min.js" in html
    assert "/static/vendor/OrbitControls.js" in html
    assert "/static/vendor/GLTFLoader.js" in html
    assert "/static/js/app.js" in html

def test_static_css_stylesheet():
    """Verifies style.css is served and contains dark theme, glassmorphism, and color rules."""
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    css = response.text
    assert "--bg-obsidian" in css
    assert "backdrop-filter" in css
    assert "--alliance-red" in css
    assert "--alliance-blue" in css
    assert "--event-score" in css
    assert "--event-climb" in css
    assert "--event-defense" in css
    assert ".playback-dock" in css
    assert ".robot-inspector" in css
    assert ".event-marker" in css

def test_static_javascript_modules():
    """Verifies all JavaScript modules and vendored loaders are served via /static/."""
    for script in ["app.js", "field3d.js", "robot3d.js", "ball3d.js"]:
        response = client.get(f"/static/js/{script}")
        assert response.status_code == 200, f"Failed to retrieve /static/js/{script}"
        assert len(response.text) > 100

    for vendor in ["three.min.js", "OrbitControls.js", "GLTFLoader.js"]:
        response = client.get(f"/static/vendor/{vendor}")
        assert response.status_code == 200, f"Failed to retrieve /static/vendor/{vendor}"
        assert len(response.text) > 1000

def test_field_glb_model_delivery():
    """Verifies official field.glb CAD model delivery via static and direct endpoints."""
    # Test HEAD request
    resp_head = client.head("/model/field.glb")
    assert resp_head.status_code == 200
    assert "model/gltf-binary" in resp_head.headers.get("content-type", "")

    # Test GET endpoint on /api/field/model
    resp_api = client.get("/api/field/model")
    assert resp_api.status_code == 200
    assert len(resp_api.content) > 60 * 1024 * 1024  # > 60 MB

    # Test static assets mount
    resp_static = client.get("/static/models/field.glb")
    assert resp_static.status_code == 200
    assert len(resp_static.content) > 60 * 1024 * 1024

