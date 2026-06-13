from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

from AIChatFramework.Backend.models.model_interface import ModelProviderBase, ModelProviderRegistry


class GenericOpenAIProvider(ModelProviderBase):

    def __init__(
            self,
            model_provider_reg: ModelProviderRegistry,
            base_url: str,
            api_key: str,
            provider_name: str = ""):
        self._model_provider_reg = model_provider_reg
        self._provider_name = provider_name
        self._name = "GenericOpenAIAPI" if not provider_name else provider_name
        self._base_url = base_url
        self._api_key = api_key
        self._openai_instance = self.generate_openai_instance()
        if self._openai_instance is not None:
            self.model_provider_reg.register(self)

    def generate_openai_instance(self) -> OpenAI | None:
        try:
            oai = OpenAI(base_url=self._base_url, api_key=self._api_key)
            oai.models.list()
            return oai
        except Exception as e:
            print(f'OpenAI Error: {e}')
        return None

    def get_models(self) -> list:
        if self._openai_instance is not None:
            list_of_models = self._openai_instance.models.list().to_dict().get('data', [])
            return [m.get('id') for m in list_of_models if 'id' in m]
        else:
            return []

    def create_model(self, model: str, model_kwargs: dict, **kwargs) -> BaseLanguageModel:
        if self._openai_instance is None:
            raise Exception('OpenAI is not initialized')
        model_kwargs['base_url'] = self._base_url
        model_kwargs['api_key'] = self._api_key
        if 'streaming' not in kwargs:
            model_kwargs['streaming'] = True
        return ChatOpenAI(model=model, **model_kwargs)

    def get_model_ids_by_name(self, name: str) -> list:
        return [name]

    def get_model_name_by_id(self, model_id):
        return model_id

    def get_model_id_from_ui_name(self, ui_name: str) -> str:
        _, model_id = ui_name.split('/')
        return model_id.strip()

    def get_ui_name_from_model_id(self, model_id: str) -> str:
        pname = self._provider_name if self._provider_name else self._name
        return f"{pname} / {model_id}"

    def get_list_of_text_model_names_for_ui(self) -> list:
        pname = self._provider_name if self._provider_name else self._name
        return [f"{pname} / {name}" for name in self.get_models() if name]

    def get_suggested_model_config(self, model_id: str) -> dict:
        return {'max_tokens': 16256}

    def get_suggested_app_config(self, model_id: str) -> dict:
        return {'max_tokens_in_history': 16256, 'batch_size': 3}

    @property
    def name(self):
        return self._name

    @property
    def model_provider_reg(self):
        return self._model_provider_reg
