import time
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
    HAS_PREPROCESS = False
    HAS_PREDICT = False
    HAS_POSTPROCESS = False

    def __init__(self, name: str, model_class: Any):
        super().__init__(name)
        KServeModel._reset()
        KServeModel._build_functions(model_class)
        self.model = model_class()

        # Only used for transforms
        self.upload_webhook = None
        self.predict_start_time = -1

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
            KServeModel.HAS_PREDICT = True

        # Preprocess function
        method = getattr(model_class, "preprocess", None)
        if callable(method):
            assert not KServeModel.HAS_PREDICT, \
                "`predict` function has been defined already, cannot define `preprocess` separately"
            KServeModel.MODEL_IO_INFO.set_input_signatures(method)
            setattr(KServeModel, "preprocess", KServeModel._preprocess)
            KServeModel.HAS_PREPROCESS = True

        # Postprocess function
        method = getattr(model_class, "postprocess", None)
        if callable(method):
            assert not KServeModel.HAS_PREDICT, \
                "`predict` function has been defined already, cannot define `postprocess` separately"
            KServeModel.MODEL_IO_INFO.set_output_signatures(method)
            setattr(KServeModel, "postprocess", KServeModel._postprocess)
            KServeModel.HAS_POSTPROCESS = True

        if KServeModel.HAS_PREPROCESS or KServeModel.HAS_POSTPROCESS:
            assert KServeModel.HAS_PREPROCESS and KServeModel.HAS_POSTPROCESS, \
                "`preprocess` and `postprocess` must be both defined"

    @staticmethod
    def _predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        start_time = time.time()
        upload_webhook = payload.pop("upload_webhook", None)
        outputs = self.model.predict(**payload)
        results = KServeModel._upload(upload_webhook, outputs)
        results["running_time"] = f"{time.time() - start_time}s"
        return results

    @staticmethod
    def _preprocess(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        self.predict_start_time = time.time()
        self.upload_webhook = payload.pop("upload_webhook", None)
        return self.model.preprocess(**payload)

    @staticmethod
    def _postprocess(self, infer_response: Dict, headers: Dict[str, str] = None) -> Dict:
        outputs = self.model.postprocess(infer_response)
        results = KServeModel._upload(self.upload_webhook, outputs)
        results["running_time"] = f"{time.time() - self.predict_start_time}s"
        self.upload_webhook = None
        return results

    @staticmethod
    def _upload(upload_webhook, model_outputs):
        if KServeModel.MODEL_IO_INFO.outputs is None:
            assert isinstance(model_outputs, dict), "Model output must be a dict"
            return model_outputs

        if KServeModel.MODEL_IO_INFO.outputs["type"] == Path:
            assert upload_webhook is not None, \
                "Model output type is `Path`, but `upload_webhook` is not set"
            assert isinstance(model_outputs, Path), \
                "Model output type is `Path`, but the actual output is not `Path`"
            return upload_files(upload_webhook, [model_outputs])

        if KServeModel.MODEL_IO_INFO.outputs["type"] == list:
            if len(KServeModel.MODEL_IO_INFO.outputs["args"]) == 1 and \
                    KServeModel.MODEL_IO_INFO.outputs["args"][0] == Path:
                assert upload_webhook is not None, \
                    "Model output type is `Path`, but `upload_webhook` is not set"
                assert isinstance(model_outputs, (list, tuple)), \
                    "Model output type is `List[Path]`, but the actual output is not a List"
                return upload_files(upload_webhook, model_outputs)

        assert isinstance(model_outputs, dict), "Model output must be a dict"
        return model_outputs

    @staticmethod
    def _reset():
        KServeModel.MODEL_IO_INFO = ModelIOInfo()
        if KServeModel.HAS_PREPROCESS:
            KServeModel.HAS_PREPROCESS = False
            method = getattr(KServeModel, "preprocess", None)
            if callable(method):
                delattr(KServeModel, "preprocess")
        if KServeModel.HAS_PREDICT:
            KServeModel.HAS_PREDICT = False
            method = getattr(KServeModel, "predict", None)
            if callable(method):
                delattr(KServeModel, "predict")
        if KServeModel.HAS_POSTPROCESS:
            KServeModel.HAS_POSTPROCESS = False
            method = getattr(KServeModel, "postprocess", None)
            if callable(method):
                delattr(KServeModel, "postprocess")

    @staticmethod
    def serve(name: str, model_class: Any):
        model = KServeModel(name, model_class)
        ModelServer().start([model])
