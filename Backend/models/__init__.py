from langchain_core.language_models import BaseLanguageModel

from AIChatFramework.Backend.models.model_interface import ModelProviders, ModelRegistry


def get_available_providers():
    return ModelProviders.get_providers()


def get_available_models():
    return ModelProviders.list_all_models()


def get_loaded_model_names():
    return ModelRegistry.get_registered_models()


def get_loaded_model_by_id(model_id: int) -> BaseLanguageModel:
    return ModelRegistry.get_model_by_id(model_id)


def create_model(model: str, **kwargs: object) -> BaseLanguageModel:
    provider = ModelRegistry.find_provider_for_model_id(model)
    if provider is None:
        raise Exception(f"Model with id {model} not found")
    return ModelRegistry.create_model_with_provider(provider, model=model, model_kwargs=kwargs)


def get_model_name_by_id(model_id: str) -> str :
    return ModelRegistry.get_model_name_by_id(model_id)


def get_model_ids_by_name(name: str) -> str:
    return ModelRegistry.get_model_ids_by_name(name)


def get_model_id_from_ui_name(ui_name: str) -> str :
    return ModelRegistry.get_model_id_from_ui_name(ui_name)


def get_ui_name_from_model_id(model_id: int) -> str:
    return ModelRegistry.get_ui_name_from_model_id(model_id)


def get_list_of_text_model_names_for_ui() -> list:
    return ModelRegistry.get_list_of_text_model_names_for_ui()


def get_suggested_model_config(model_id: str) -> dict:
    provider = ModelRegistry.find_provider_for_model_name(model_id)
    return provider.get_suggested_model_config(model_id)


def get_suggested_app_config(model_id: str) -> dict:
    provider = ModelRegistry.find_provider_for_model_name(model_id)
    return provider.get_suggested_app_config(model_id)

