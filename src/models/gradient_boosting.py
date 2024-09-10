import os
import pickle
import sys
import warnings
from pprint import pprint

import optuna
import xgboost as xgb
from omegaconf import OmegaConf
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error


warnings.filterwarnings("ignore", category=FutureWarning)

sys.path[-1] = os.sep.join(__file__.split(os.sep)[:-2])

current_dir = os.getcwd()
os.chdir(os.path.dirname(os.sep.join(__file__.split(os.sep)[:-2])))

from preprocessing import prepare_train_data, split_train_data
from utils import compute_metrics


os.chdir(os.path.dirname(os.sep.join(__file__.split(os.sep)[:-2])))

cfg = OmegaConf.load("src/config/config.yaml")
os.chdir(current_dir)


class OptunaSearchXGB:
    def __init__(self):
        self.train_features, self.val_features, self.train_targets, self.val_targets = (
            self.load_data()
        )

        # Создаем DMatrix для XGBoost
        self.dtrain = xgb.DMatrix(self.train_features, self.train_targets)
        self.dval = xgb.DMatrix(self.val_features, self.val_targets)
        self.get_boosting_model()

    def objective(self, trial):
        # Определяем пространство поиска гиперпараметров
        param = {
            "verbosity": 0,
            "device": "cuda",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0),
        }

        # Обучаем модель
        bst = xgb.train(
            param,
            self.dtrain,
            evals=[(self.dval, "eval")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Прогнозируем и считаем RMSE на валидационной выборке
        preds = bst.predict(self.dval)
        rmse = mean_squared_error(self.val_targets, preds, squared=False)

        return rmse

    def search_best_params(self):
        # Запускаем оптимизацию
        study = optuna.create_study(direction="minimize", sampler=TPESampler())
        study.optimize(self.objective, n_trials=100)

        with open(
            f"{cfg.general.data_dir}/secondary/dataloaders/boosting-study-with-best_params.pkl",
            "wb",
        ) as file:
            pickle.dump(study, file)

        return study

    def get_boosting_model(self):
        study = self.search_best_params()

        """
        # Визуализация результатов
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
        optuna.visualization.plot_contour(study, params=['max_depth', 'learning_rate']).show()
        optuna.visualization.plot_slice(study).show()
        optuna.visualization.plot_edf(study).show()
        optuna.visualization.plot_terminator_improvement(study).show()
        optuna.visualization.plot_rank(study).show()
        optuna.visualization.plot_timeline(study).show()
        """

        self.best_params = study.best_params
        self.best_params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
        self.model = xgb.train(
            self.best_params,
            self.dtrain,
            evals=[(self.dval, "eval")],
            early_stopping_rounds=50,
            verbose_eval=True,
        )

        return self.model

    @staticmethod
    def load_data():
        if not os.path.exists(
            f"{cfg.general.data_dir}/secondary/dataloaders/train_val_x_y.pkl"
        ):
            features, targets = prepare_train_data()
            train_features, val_features, train_targets, val_targets = split_train_data(
                features, targets
            )
            print(
                train_features.shape,
                val_features.shape,
                train_targets.shape,
                val_targets.shape,
            )
            with open(
                f"{cfg.general.data_dir}/secondary/dataloaders/train_val_x_y.pkl", "wb"
            ) as file:
                pickle.dump((train_features, val_features, train_targets, val_targets), file)
        else:
            with open(
                f"{cfg.general.data_dir}/secondary/dataloaders/train_val_x_y.pkl", "rb"
            ) as file:
                train_features, val_features, train_targets, val_targets = pickle.load(file)

        return train_features, val_features, train_targets, val_targets

    def predict(self, dval):
        return self.model.predict(data=xgb.DMatrix(dval))


if __name__ == "__main__":
    train_features, val_features, train_targets, val_targets = OptunaSearchXGB.load_data()

    model = OptunaSearchXGB()
    y_pred = model.predict(val_features)
    pprint(compute_metrics(val_targets, y_pred))
