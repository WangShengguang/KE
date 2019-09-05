from .ConvKB import ConvKB
from .TransE import TransE
from .TransformerKB import TransformerKB

TrainXModels = ["TransE"]
other_models = ["ConvKB", "TransformerKB"]

__all__ = other_models + TrainXModels
