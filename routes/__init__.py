"""
routes package
--------------
Modular Flask Blueprints for Modules M1 to M6 and REST APIs.
Each individual route module strictly contains fewer than 250 lines.
"""

from routes.m1_routes import m1_bp
from routes.m2_routes import m2_bp
from routes.m3_routes import m3_bp
from routes.m4_routes import m4_bp
from routes.m5_routes import m5_bp
from routes.m6_routes import m6_bp
from routes.api_routes import api_bp

__all__ = ["m1_bp", "m2_bp", "m3_bp", "m4_bp", "m5_bp", "m6_bp", "api_bp"]
