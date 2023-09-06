from typing import Any, Dict
from kserve import Model, ModelServer


class KServeModel(Model):

    def __init__(self, name: str, model_class: Any):
        super().__init__(name)
        KServeModel._build_functions(model_class)
        self.model = model_class()

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
            setattr(KServeModel, "predict", KServeModel._predict)
        # Preprocess function
        method = getattr(model_class, "preprocess", None)
        if callable(method):
            setattr(KServeModel, "preprocess", KServeModel._preprocess)
        # Postprocess function
        method = getattr(model_class, "postprocess", None)
        if callable(method):
            setattr(KServeModel, "postprocess", KServeModel._postprocess)

    @staticmethod
    def _predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        return self.model.predict(payload, headers)

    @staticmethod
    def _preprocess(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        return self.model.preprocess(payload, headers)

    def _postprocess(self, infer_response: Dict, headers: Dict[str, str] = None) -> Dict:
        return self.model.postprocess(infer_response, headers)

    @staticmethod
    def serve(name: str, model_class: Any):
        model = KServeModel(name, model_class)
        ModelServer().start([model])
