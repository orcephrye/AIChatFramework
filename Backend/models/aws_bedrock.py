import boto3
from langchain_aws import ChatBedrockConverse, ChatBedrock
from functools import cache, partial

from langchain_core.language_models import BaseLanguageModel

from AIChatFramework.Backend.models.model_interface import ModelProviderBase, ModelProviderRegistry


MODEL_TYPES = {'text': ['TEXT'], 'video': ['VIDEO'], 'image': ['IMAGE'], 'embedding': ['EMBEDDING']}


client = boto3.client('sts')
bedrock_client = boto3.client('bedrock')


def filter_by_type(model_type, model_json):
    return model_json.get('outputModalities', []) == MODEL_TYPES.get(model_type, ['TEXT'])

@cache
def get_foundational_models() -> list:
    results = bedrock_client.list_foundation_models()
    if results['ResponseMetadata']['HTTPStatusCode'] != 200:
        return []
    return results['modelSummaries']

@cache
def get_foundational_models_names(type_to_filter = None):
    if type_to_filter is not None:
        f = partial(filter_by_type, type_to_filter)
        return [m['modelName'] for m in filter(f, get_foundational_models())]
    return [m['modelName'] for m in get_foundational_models()]

@cache
def get_model_by_id(model_id):
    for model in get_foundational_models():
        if model['modelId'] == model_id:
            return model
    return None


@cache
def get_foundational_models_ids_names(type_to_filter = None):
    if type_to_filter is not None:
        f = partial(filter_by_type, type_to_filter)
        return [m['modelId'] for m in filter(f, get_foundational_models())]
    return [m['modelId'] for m in get_foundational_models()]

@cache
def get_model_id_by_name(name):
    for model in get_foundational_models():
        if model['modelName'] == name:
            return model['modelId']
    return None


@cache
def get_model_ids_by_name(name: str, type_to_filter = None):
    if type_to_filter is not None:
        f = partial(filter_by_type, type_to_filter)
        return [m['modelId'] for m in filter(f, get_foundational_models()) if m['modelName'] == name]
    return [m['modelId'] for m in get_foundational_models() if m['modelName'] == name]

def get_list_of_text_model_names_for_ui():
    return sorted(list(set([f"{m['providerName']} / {m['modelName']}"
                            for m in filter(partial(filter_by_type, 'text'), get_foundational_models())])))

@cache
def get_model_id_from_ui_name(ui_name: str) -> str:
    _, model_name = ui_name.split('/')
    list_of_names = get_model_ids_by_name(model_name.strip())
    if list_of_names:
        return str(min(list_of_names, key=len))
    return ""

@cache
def get_ui_name_from_model_id(model_id: str):
    m = get_model_by_id(model_id)
    if m is not None:
        return f"{m['providerName']} / {m['modelName']}"
    return ""


@cache
def get_model_name_by_id(model_id):
    for model in get_foundational_models():
        if model['modelId'] == model_id:
            return model['modelName']
    return ""


@cache
def get_model_arn_by_name(name):
    for model in get_foundational_models():
        if model['modelName'] == name:
            return model['modelArn']
    return ""


@cache
def _needs_prefix(model_id):
    model = get_model_by_id(model_id)
    if not model:
        return
    return 'ON_DEMAND' not in model.get('inferenceTypesSupported', [])

def get_iam_user_id():
    # Get the caller identity
    identity = client.get_caller_identity()

    # Extract and return the User ID
    user_id = identity['UserId']
    return user_id

@cache
def get_std_converse(**kwargs):
    kwargs.setdefault('model', "")
    kwargs.setdefault('max_tokens', 2048)
    kwargs.setdefault('temperature', 0.7)
    kwargs.setdefault('top_p', 1)
    kwargs.setdefault('verbose', True)
    if not kwargs['model'].startswith('us'):
        if _needs_prefix(kwargs['model']):
            kwargs['model'] = 'us.' + kwargs['model']
    return ChatBedrockConverse(**kwargs)

@cache
def get_std_bedrock(**kwargs):
    kwargs.setdefault('model_id', "")
    kwargs.setdefault('model_kwargs', {})
    kwargs['model_kwargs'].setdefault('max_tokens', 2048)
    kwargs['model_kwargs'].setdefault('temperature', 0.7)
    kwargs['model_kwargs'].setdefault('top_p', 1)
    if not kwargs['model_id'].startswith('us'):
        if _needs_prefix(kwargs['model_id']):
            kwargs['model_id'] = 'us.' + kwargs['model_id']
    return ChatBedrock(**kwargs)


class Bedrock(ModelProviderBase):

    def __init__(self, model_provider_reg: ModelProviderRegistry):
        self._name = "AWS_BEDROCK"
        self._model_provider_reg = model_provider_reg
        self.model_provider_reg.register(self)

    def get_models(self) -> list:
        return get_foundational_models_ids_names('text')

    def create_model(self, model: str, model_kwargs: dict, **kwargs) -> BaseLanguageModel:
        return get_std_converse(model=model, **model_kwargs)

    def get_model_ids_by_name(self, name: str) -> list:
        return get_model_ids_by_name(name)

    def get_model_name_by_id(self, model_id):
        return get_model_name_by_id(model_id)

    def get_model_id_from_ui_name(self, ui_name: str) -> str:
        return get_model_id_from_ui_name(ui_name)

    def get_ui_name_from_model_id(self, model_id: str) -> str:
        return get_ui_name_from_model_id(model_id)

    def get_list_of_text_model_names_for_ui(self) -> list:
        return get_list_of_text_model_names_for_ui()

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
