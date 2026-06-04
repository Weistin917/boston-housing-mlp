from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer
from models.BaseModel import BaseModel

class RFRegressor(BaseModel):
  def __init__(
    self,
    n_estimators = [100, 200, 500],
    max_depth = [None, 10, 20, 30],
    min_samples_split = [2, 5, 10],
    max_features = ['sqrt', 'log2', None],
    metrics: list[str] = ["neg_mean_squared_error"],
    cv = 5,
    random_state = 49
  ):
    super().__init__()
    rf = RandomForestRegressor(random_state=random_state)
    param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'max_features': max_features 
    }
    self.metrics = {f'm{i}' : metrics[i] for i in range(len(metrics))}
    self.model = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=self.metrics,
        refit='m0',
        cv=cv,
        return_train_score=True,
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
