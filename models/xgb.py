import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import get_scorer
from models.BaseModel import BaseModel

class XGBoost(BaseModel):
  def __init__(
    self,
    n_estimators = [100, 1000, 50],
    learning_rate = [0.01, 0.05, 0.1, 0.2],
    max_depth = [3, 4, 5, 6, 8],
    subsample = [0.6, 0.7, 0.8, 0.9, 1.0],
    colsample_bytree = [0.6, 0.7, 0.8, 0.9, 1.0],
    metrics: list[str] = ["neg_mean_squared_error"],
    n_iter = 20,
    cv = 5,
    random_state = 49
  ):
    super().__init__()
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=random_state)
    param_distributions = {
    'n_estimators': np.arange(n_estimators[0], n_estimators[1], n_estimators[2]),
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree
    }
    self.metrics = {f'm{i}' : metrics[i] for i in range(len(metrics))}
    self.model = RandomizedSearchCV(
      estimator=xgb_model,
      param_distributions=param_distributions,
      n_iter=n_iter,
      scoring=self.metrics,
      refit='m0',
      cv=cv,
      verbose=1,
      random_state=random_state,
      n_jobs=-1
    )

  def train(
    self,
    X,
    y,
    tee
  ):
    self.model.fit(X, y.squeeze().values)
    tee.write(f"Best Params: {self.model.best_params_}")

    best_index = self.model.best_index_
    tee.write("Best results:")
    for key, value in self.metrics.items():
      tee.write(f"Best {value}: {self.model.cv_results_['mean_test_{}'.format(key)][best_index]:.4f}")
    
    return self.model.cv_results_

  def evaluate(
    self,
    X,
    y
  ):
    scores = {}
    for metric in self.metrics.values():
      scorer = get_scorer(metric)
      score = scorer(self.model, X, y.squeeze().values)
      if "neg" in metric: score *= -1
      scores[metric] = score
    return scores

