from typing import Any, Dict, Callable
from inspect import signature
from kserve import Model, ModelServer


class ModelInputs:

    def __init__(self, method: Callable):
        ModelInputs._parse_method_signatures(method)

    @staticmethod
    def _parse_method_signatures(method):
        t = signature(method)
        print(t)
        print(t.parameters)


class KServeModel(Model):
    MODEL_INPUTS: ModelInputs = None

    def __init__(self, name: str, model_class: Any):
        super().__init__(name)
        KServeModel._build_functions(model_class)
        self.model = model_class()
        self.upload_webhook = None

    def load(self) -> bool:
        method = getattr(self.model, "load", None)
        if callable(method):
            self.model.load()
        self.ready = True
        return self.ready

    @staticmethod
    def _build_functions(model_class):
        # Predict function
        method = getattr(model_class, "predict", None)
        if callable(method):
            KServeModel.MODEL_INPUTS = ModelInputs(method)
            setattr(KServeModel, "predict", KServeModel._predict)

        # Preprocess function
        method = getattr(model_class, "preprocess", None)
        if callable(method):
            KServeModel.MODEL_INPUTS = ModelInputs(method)
            setattr(KServeModel, "preprocess", KServeModel._preprocess)

        # Postprocess function
        method = getattr(model_class, "postprocess", None)
        if callable(method):
            setattr(KServeModel, "postprocess", KServeModel._postprocess)

    @staticmethod
    def _predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        self.upload_webhook = payload.pop("upload_webhook", None)
        return self.model.predict(**payload)

    @staticmethod
    def _preprocess(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        self.upload_webhook = payload.pop("upload_webhook", None)
        return self.model.preprocess(**payload)

    @staticmethod
    def _postprocess(self, infer_response: Dict, headers: Dict[str, str] = None) -> Dict:
        return self.model.postprocess(infer_response)

    @staticmethod
    def serve(name: str, model_class: Any):
        model = KServeModel(name, model_class)
        ModelServer().start([model])
