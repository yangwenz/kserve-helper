import typing
import pydantic
import inspect
from collections import OrderedDict
from typing import Any, Dict, Callable
from inspect import signature
from kserve import Model, ModelServer
from kservehelper.types import Path
from kservehelper.utils import upload_files


class ModelIOInfo:

    def __init__(self):
        self._input_info = None
        self._output_info = None

    @staticmethod
    def _fieldinfo2dict(info):
        d = OrderedDict()
        for arg in info.__repr_args__():
            if arg[0] != "extra":
                d[arg[0]] = arg[1]
            else:
                d.update({k: v for k, v in arg[1].items() if v})
        return d

    def set_input_signatures(self, method: Callable):
        t = signature(method)
        input_info = OrderedDict()
        for key, value in t.parameters.items():
            if isinstance(value.default, pydantic.fields.FieldInfo):
                d = ModelIOInfo._fieldinfo2dict(value.default)
                d["type"] = value.annotation
                input_info[key] = d
        self._input_info = input_info

    def set_output_signatures(self, method: Callable):
        t = signature(method)
        output_info = OrderedDict()
        if t.return_annotation == inspect._empty:
            output_info["type"] = None
            output_info["args"] = ()
        else:
            output_info["type"] = typing.get_origin(t.return_annotation)
            if output_info["type"] is None:
                output_info["type"] = t.return_annotation
            output_info["args"] = typing.get_args(t.return_annotation)
        self._output_info = output_info

    @property
    def inputs(self):
        return self._input_info

    @property
    def outputs(self):
        return self._output_info


class KServeModel(Model):
    MODEL_IO_INFO = ModelIOInfo()

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
            KServeModel.MODEL_IO_INFO.set_input_signatures(method)
            KServeModel.MODEL_IO_INFO.set_output_signatures(method)
            setattr(KServeModel, "predict", KServeModel._predict)

        # Preprocess function
        method = getattr(model_class, "preprocess", None)
        if callable(method):
            KServeModel.MODEL_IO_INFO.set_input_signatures(method)
            setattr(KServeModel, "preprocess", KServeModel._preprocess)

        # Postprocess function
        method = getattr(model_class, "postprocess", None)
        if callable(method):
            KServeModel.MODEL_IO_INFO.set_output_signatures(method)
            setattr(KServeModel, "postprocess", KServeModel._postprocess)

    @staticmethod
    def _predict(self, payload: Dict, headers: Dict[str, str] = None):
        self.upload_webhook = payload.pop("upload_webhook", None)
        outputs = self.model.predict(**payload)
        return KServeModel._upload(self.upload_webhook, outputs)

    @staticmethod
    def _preprocess(self, payload: Dict, headers: Dict[str, str] = None):
        self.upload_webhook = payload.pop("upload_webhook", None)
        return self.model.preprocess(**payload)

    @staticmethod
    def _postprocess(self, infer_response: Dict, headers: Dict[str, str] = None):
        outputs = self.model.postprocess(infer_response)
        return KServeModel._upload(self.upload_webhook, outputs)

    @staticmethod
    def _upload(upload_webhook, model_outputs):
        if upload_webhook is None or KServeModel.MODEL_IO_INFO.outputs is None:
            return model_outputs

        if KServeModel.MODEL_IO_INFO.outputs["type"] == Path:
            assert not isinstance(model_outputs, (list, tuple)), \
                "Model output type is `Path`, but the actual output is a List"
            return upload_files(upload_webhook, [model_outputs])

        if KServeModel.MODEL_IO_INFO.outputs["type"] == list:
            if len(KServeModel.MODEL_IO_INFO.outputs["args"]) == 1 and \
                    KServeModel.MODEL_IO_INFO.outputs["args"][0] == Path:
                assert isinstance(model_outputs, (list, tuple)), \
                    "Model output type is `List[Path]`, but the actual output is not a List"
                return upload_files(upload_webhook, model_outputs)
        return model_outputs

    @staticmethod
    def serve(name: str, model_class: Any):
        model = KServeModel(name, model_class)
        ModelServer().start([model])
