from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer
from models.BaseModel import BaseModel

class SVRegressor(BaseModel):
  def __init__(
    self,
    kernel: str = "rbf",
    C = [0.1, 1, 10, 100],
    epsilon = [0.001, 0.1, 1, 2],
    gamma = ['scale', 0.1, 0.01, 0.001],
    metrics: list[str] = ["neg_mean_squared_error"],
    cv = 5
  ):
    super().__init__()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf'))
    ])
    param_grid = {
      'svr__C': C,
      'svr__gamma': gamma,
      'svr__epsilon': epsilon
    }
    self.metrics = {f'm{i}' : metrics[i] for i in range(len(metrics))}
    self.model = GridSearchCV(
        estimator=pipe,
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
