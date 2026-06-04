import tensorflow.keras as keras
from models.BaseModel import BaseModel

class MLP(BaseModel):
  """
  Multilayer Perceptron (Feed Forward Network) implementation with Tensorflow Keras.
  Batch normalization placed in every hidden layer.
  """
  def __init__(
    self, 
    in_channels: int, 
    out_channels: int,
    features: list[int] = [32, 64], 
    activation: str | list[str] = "relu", 
    last_activation: str = "relu",
    loss: str = None,
    metrics: list[str] = None
  ):

    super().__init__()
    self.model = keras.Sequential()
    self.model.add(keras.Input(shape=(in_channels,)))
    
    if isinstance(activation, list):
      assert(len(activation) == len(features))
      for i in range(len(features)):
        self.model.add(keras.layers.Dense(units=features[i]))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Activation(activation[i]))
    else:
      for i in range(len(features)):
        self.model.add(keras.layers.Dense(units=features[i]))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Activation(activation))

    self.model.add(keras.layers.Dense(units=out_channels, activation=last_activation))

    self.metrics = metrics
    model_metrics = None
    if metrics != None:
      model_metrics = [keras.metrics.get(m) for m in metrics]
    
    self.model.compile(
      loss=loss,
      metrics=model_metrics
    )

  def train(
    self, 
    X, 
    y,
    lr: float,
    optimizer: str,
    num_epochs: int = 50,
    batch_size: int = 64,
    val_split: float = 0.2,
    callbacks: list[keras.callbacks.Callback] = None,
    starting_epoch: int = 0
  ):

    self.model.optimizer = keras.optimizers.get(optimizer)
    self.model.optimizer.learning_rate.assign(lr)

    history = self.model.fit(
      X,
      y,
      epochs=num_epochs,
      batch_size=batch_size,
      validation_split=val_split,
      callbacks=callbacks,
      initial_epoch = starting_epoch
    )

    return len(history.epoch)

  def evaluate(
    self,
    X,
    y,
    batch_size: int = 64
  ):
    scores = {}
    results = self.model.evaluate(X, y, batch_size=batch_size)
    
    scores["loss"] = results[0]
    for metric, value in zip(self.metrics, results[1:]):
      scores[metric] = value
    
    return scores
