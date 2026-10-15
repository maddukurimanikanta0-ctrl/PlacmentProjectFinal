"""
app.py
------
PlacementPredict - Flask Web Application & Serving Framework (M1–M6).
Refactored into modular blueprints strictly under 300 lines per file.
Provides Web UI, Dataset Ingestion, Analytics, ML Workbenches, and REST APIs.
"""

from pathlib import Path
from flask import Flask, render_template

from routes.m1_routes import m1_bp, index, data_loading, preprocessing, eda, predict
from routes.m2_routes import m2_bp, linear_regression, logistic_regression
from routes.m3_routes import m3_bp, tree_models
from routes.m4_routes import m4_bp, unsupervised
from routes.m5_routes import m5_bp, validation
from routes.m6_routes import m6_bp
from routes.api_routes import api_bp, health, api_predict, api_validation


def create_app() -> Flask:
    """Factory creating and configuring Flask application with all M1-M6 blueprints."""
    app = Flask(__name__)

    # Register modular blueprints
    app.register_blueprint(m1_bp)
    app.register_blueprint(m2_bp)
    app.register_blueprint(m3_bp)
    app.register_blueprint(m4_bp)
    app.register_blueprint(m5_bp)
    app.register_blueprint(m6_bp)
    app.register_blueprint(api_bp)

    # Register non-namespaced aliases for backward compatibility with tests & templates
    app.add_url_rule("/", endpoint="index", view_func=index)
    app.add_url_rule("/home", endpoint="home", view_func=index)
    app.add_url_rule("/data-loading", endpoint="data_loading", view_func=data_loading)
    app.add_url_rule("/eda", endpoint="eda", view_func=eda)
    app.add_url_rule("/preprocessing", endpoint="preprocessing", view_func=preprocessing)
    app.add_url_rule("/linear-regression", endpoint="linear_regression", view_func=linear_regression, methods=["GET", "POST"])
    app.add_url_rule("/logistic-regression", endpoint="logistic_regression", view_func=logistic_regression, methods=["GET", "POST"])
    app.add_url_rule("/tree-models", endpoint="tree_models", view_func=tree_models)
    app.add_url_rule("/unsupervised", endpoint="unsupervised", view_func=unsupervised)
    app.add_url_rule("/clustering", endpoint="clustering", view_func=unsupervised)
    app.add_url_rule("/validation", endpoint="validation", view_func=validation, methods=["GET", "POST"])
    app.add_url_rule("/model-validation", endpoint="model_validation", view_func=validation, methods=["GET", "POST"])
    app.add_url_rule("/predict", endpoint="predict", view_func=predict, methods=["POST"])
    app.add_url_rule("/health", endpoint="health", view_func=health)
    app.add_url_rule("/api/predict", endpoint="api_predict", view_func=api_predict, methods=["POST"])
    app.add_url_rule("/api/validation", endpoint="api_validation", view_func=api_validation)

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("base.html"), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
