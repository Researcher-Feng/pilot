"""
Raw API client for LLM models without LangChain dependencies.
Supports OpenAI-compatible APIs and Ollama.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI
from types import SimpleNamespace


class RawAPIClient:
    """Raw API client for OpenAI-compatible APIs"""
    
    def __init__(self, model_name: str, base_url: Optional[str] = None, 
                 api_key: Optional[str] = None, temperature: float = 0.7,
                 max_tokens: int = 2000):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client
        client_kwargs = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
            
        self.client = OpenAI(**client_kwargs)
    
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Invoke the API with messages"""
        try:
            if 'deepseek' in self.model_name:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tool_choice='none',
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            # print(f'response message: {response.choices[0].message.content}')
            print(f'total_tokens consumation: {response.usage.total_tokens}')
            
            content = response.choices[0].message.content
            
            return {
                'messages': [{'role': 'assistant', 'content': content}],
                'structured_response': SimpleNamespace(main_response=content)
            }
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}") # e.body['message']
    
    def stream(self, messages: List[Dict[str, str]]):
        """Stream response from API"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"API stream failed: {str(e)}")


class RawOllamaClient:
    """Raw API client for Ollama"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.7, max_tokens: int = 2000):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Invoke Ollama API with messages"""
        # Convert messages to prompt format for Ollama
        prompt = self._messages_to_prompt(messages)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "")
            
            return {
                'messages': [{'role': 'assistant', 'content': content}],
                'structured_response': SimpleNamespace(main_response=content)
            }
        except Exception as e:
            raise Exception(f"Ollama API call failed: {str(e)}")
    
    def stream(self, messages: List[Dict[str, str]]):
        """Stream response from Ollama"""
        prompt = self._messages_to_prompt(messages)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                stream=True,
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
        except Exception as e:
            raise Exception(f"Ollama API stream failed: {str(e)}")
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt format for Ollama"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)


def create_api_client(model_config) -> Any:
    """Create appropriate API client based on model config"""
    if model_config.model_type == "local":
        return RawOllamaClient(
            model_name=model_config.model_name,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens
        )
    else:
        # Determine API key and base URL from config or environment
        api_key = getattr(model_config, 'api_key', None)
        base_url = getattr(model_config, 'base_url', None)
        
        # Try to get from environment if not in config
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not base_url:
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
        
        return RawAPIClient(
            model_name=model_config.model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
        )

