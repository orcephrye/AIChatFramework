from abc import ABC, abstractmethod
from threading import Lock
from langchain_core.language_models import BaseLanguageModel


class ModelProviderBase(ABC):

    @abstractmethod
    def get_models(self) -> list:
        pass

    @abstractmethod
    def create_model(self, model: str, model_kwargs: dict) -> BaseLanguageModel:
        pass

    @abstractmethod
    def get_model_ids_by_name(self, name: str) -> list:
        pass

    @abstractmethod
    def get_model_name_by_id(self, model_id):
        pass

    @abstractmethod
    def get_list_of_text_model_names_for_ui(self) -> list:
        pass

    @abstractmethod
    def get_model_id_from_ui_name(self, ui_name: str) -> str:
        pass

    @abstractmethod
    def get_ui_name_from_model_id(self, model_id: str) -> str:
        pass

    @abstractmethod
    def get_suggested_model_config(self, model_id: str) -> dict:
        pass

    @abstractmethod
    def get_suggested_app_config(self, model_id: str) -> dict:
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def model_provider_reg(self):
        pass


class ModelProviderRegistry:
    _instance = None
    _initialized = False
    _lock = Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelProviderRegistry._initialized:
            self.__registry = {}
            ModelProviderRegistry._initialized = True


    def register(self, model_provider: ModelProviderBase):
        self.__registry[model_provider.name] = model_provider

    def get_provider(self, name: str) -> ModelProviderBase | None:
        return self.__registry.get(name, None)

    def get_providers(self) -> list:
        return list(self.__registry.keys())

    def list_models_for_provider(self, name: str) -> list:
        if name not in self.__registry:
            return []
        return self.__registry[name].get_models()

    def list_all_models(self):
        models = []
        for name in self.__registry:
            models.extend(self.list_models_for_provider(name))
        return models


class ModelsRegistry:
    _instance = None
    _initialized = False
    _lock = Lock()

    def __new__(cls, model_provider_reg: ModelProviderRegistry):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_provider_reg: ModelProviderRegistry):
        if not ModelsRegistry._initialized:
            self.__registry = {}
            self.model_provider_reg = model_provider_reg
            ModelsRegistry._initialized = True

    def register(self, model_id: str, model: BaseLanguageModel):
        self.__registry[model_id] = model

    def get_model_by_id(self, model_id: str) -> BaseLanguageModel | None:
        if model_id in self.__registry:
            return self.__registry[model_id]
        return None

    def get_registered_models(self) -> list:
        return list(self.__registry.keys())

    def list_all_possible_models(self) -> list:
        return self.model_provider_reg.list_all_models()

    def find_provider_for_model_id(self, model_id: str) -> ModelProviderBase | None:
        for provider_name in self.model_provider_reg.get_providers():
            for mid in self.model_provider_reg.list_models_for_provider(provider_name):
                if mid == model_id:
                    return self.model_provider_reg.get_provider(provider_name)
        return None

    def find_provider_for_model_name(self, name: str) -> ModelProviderBase | None:
        for provider_name in self.model_provider_reg.get_providers():
            provider = self.model_provider_reg.get_provider(provider_name)
            if provider.get_model_ids_by_name(name):
                return provider
        return None

    def create_model_with_provider(
            self,
            provider: ModelProviderBase,
            model: str,
            model_kwargs: dict) -> BaseLanguageModel:
        m = provider.create_model(model=model, model_kwargs=model_kwargs)
        self.register(model, m)
        return m

    def get_model_ids_by_name(self, name: str) -> list:
        provider = self.find_provider_for_model_name(name)
        if provider is not None:
            return provider.get_model_ids_by_name(name)
        return []

    def get_model_name_by_id(self, model_id: str) -> str:
        provider = self.find_provider_for_model_id(model_id)
        if provider is not None:
            return provider.get_model_name_by_id(model_id)
        return ""

    def get_model_id_from_ui_name(self, ui_name: str) -> str:
        for provider_name in self.model_provider_reg.get_providers():
            provider = self.model_provider_reg.get_provider(provider_name)
            name = provider.get_model_id_from_ui_name(ui_name)
            if name != "":
                return name
        return ""

    def get_ui_name_from_model_id(self, model_id: str) -> str:
        provider = self.find_provider_for_model_id(model_id)
        if provider is not None:
            return provider.get_ui_name_from_model_id(model_id)
        return ""

    def get_list_of_text_model_names_for_ui(self) -> list:
        model_ui_names = []
        for provider_name in self.model_provider_reg.get_providers():
            provider = self.model_provider_reg.get_provider(provider_name)
            model_ui_names.extend(provider.get_list_of_text_model_names_for_ui())
        return model_ui_names


ModelProviders = ModelProviderRegistry()
ModelRegistry = ModelsRegistry(model_provider_reg=ModelProviders)