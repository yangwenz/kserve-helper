import time
from kservehelper.model import KServeModel
from kservehelper.types import Input


class Model:

    def load(self):
        pass

    def predict(
            self,
            repeat: int = Input(
                description="The number of repeats",
                default=5
            )
    ):
        def _generator():
            for i in range(repeat):
                yield "Hello World!"
                time.sleep(1)

        return KServeModel.wrap_generator(_generator)


if __name__ == "__main__":
    KServeModel.serve("streaming", Model)
