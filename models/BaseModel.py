from abc import ABC, abstractmethod

class BaseModel(ABC):
  @abstractmethod
  def train(self, X, y, **kwargs):
    pass

  @abstractmethod
  def evaluate(self, X, y, **kwargs):
    pass
