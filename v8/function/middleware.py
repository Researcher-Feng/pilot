"""
Middleware module - simplified for raw API calls.
Only ModelConfig is kept as it's used for configuration.
"""
from typing import Optional


class ModelConfig:
    """统一的模型配置类"""

    def __init__(self, model_type: str = "api", **kwargs):
        self.model_type = model_type  # "api" 或 "local"
        self.model_name = kwargs.get("model_name", "")
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 2000)
        self.extra_params = kwargs.get("extra_params", {})
        self.api_key = kwargs.get("api_key", None)


# Legacy classes kept for compatibility but not used with raw API calls
class CustomMiddleware:
    """Placeholder for compatibility - not used with raw API calls"""
    pass


class MiddlewareFunc:
    """Placeholder for compatibility - not used with raw API calls"""
    def __init__(self, basic_model, advanced_model):
        self.basic_model = basic_model
        self.advanced_model = advanced_model