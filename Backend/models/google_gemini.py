import os
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseLanguageModel

from AIChatFramework.Backend.models.model_interface import ModelProviderBase, ModelProviderRegistry


class GoogleGeminiProvider(ModelProviderBase):

    def __init__(self, model_provider_reg: ModelProviderRegistry, api_key: str = None):
        self._name = "GOOGLE_GEMINI"
        self._model_provider_reg = model_provider_reg
        self._api_key = api_key
        self._client = None
        self.model_provider_reg.register(self)

    @property
    def client(self) -> genai.Client | None:
        # Resolve api_key from init param or environment variables
        api_key = self._api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        
        # Initialize or reinitialize client if API key changed
        if self._client is None or getattr(self, "_last_api_key", None) != api_key:
            try:
                self._client = genai.Client(api_key=api_key)
                self._last_api_key = api_key
            except Exception as e:
                print(f"Error initializing Google Gemini Client: {e}")
                self._client = None
        return self._client

    def get_models(self) -> list:
        client = self.client
        if client is None:
            return []
        
        try:
            models = []
            for m in client.models.list():
                if hasattr(m, 'supported_actions') and m.supported_actions and "generateContent" in m.supported_actions:
                    models.append(m.name[7:])
            return sorted(models)
        except Exception as e:
            print(f"Error listing Google Gemini models: {e}")
            return []

    def create_model(self, model: str, model_kwargs: dict, **kwargs) -> BaseLanguageModel:
        api_key = self._api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            model_kwargs.setdefault("api_key", api_key)
        
        if 'streaming' not in kwargs:
            model_kwargs['streaming'] = True
            
        return ChatGoogleGenerativeAI(model=model, **model_kwargs)

    def get_model_ids_by_name(self, name: str) -> list:
        return [name]

    def get_model_name_by_id(self, model_id: str) -> str:
        return model_id

    def get_model_id_from_ui_name(self, ui_name: str) -> str:
        if '/' in ui_name:
            _, model_id = ui_name.split('/', 1)
            return model_id.strip()
        return ui_name.strip()

    def get_ui_name_from_model_id(self, model_id: str) -> str:
        return f"{self._name} / {model_id}"

    def get_list_of_text_model_names_for_ui(self) -> list:
        return [f"{self._name} / {model_id}" for model_id in self.get_models() if model_id]

    def get_suggested_model_config(self, model_id: str) -> dict:
        return {'max_tokens': 8192}

    def get_suggested_app_config(self, model_id: str) -> dict:
        return {'max_tokens_in_history': 8192, 'batch_size': 3}

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_provider_reg(self) -> ModelProviderRegistry:
        return self._model_provider_reg