from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import ConfigurationSpace
from pyrfr import regression
from pyrfr.regression import binary_rss_forest as BinaryForest
from pyrfr.regression import default_data_container as DataContainer
import lightgbm as lgb
import pandas as pd
from typing import Optional, Dict, Any, Tuple

from smac.constants import N_TREES, VERY_SMALL_NUMBER
from smac.model.random_forest import AbstractRandomForest

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class SMACLightGBMSurrogate(AbstractRandomForest):
    """LGM Random forest that takes instance features into account.

    Parameters
    ----------
    n_trees : int, defaults to `N_TREES`
        The number of trees in the random forest.
    n_points_per_tree : int, defaults to -1
        Number of points per tree. If the value is smaller than 0, the number of samples will be used.
    ratio_features : float, defaults to 5.0 / 6.0
        The ratio of features that are considered for splitting.
    min_samples_split : int, defaults to 3
        The minimum number of data points to perform a split.
    min_samples_leaf : int, defaults to 3
        The minimum number of data points in a leaf.
    max_depth : int, defaults to 2**20
        The maximum depth of a single tree.
    eps_purity : float, defaults to 1e-8
        The minimum difference between two target values to be considered.
    max_nodes : int, defaults to 2**20
        The maximum total number of nodes in a tree.
    bootstrapping : bool, defaults to True
        Enables bootstrapping.
    log_y: bool, defaults to False
        The y values (passed to this random forest) are expected to be log(y) transformed.
        This will be considered during predicting.
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : float, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        n_estimators: int = 100,               # Number of boosting rounds (trees)
        learning_rate: float = 0.1,             # Learning rate
        max_depth: int = -1,                    # Depth of trees, -1 means no limit
        num_leaves: int = 127,                   # Number of leaves per tree
        min_data_in_leaf: int = 20,             # Minimum samples per leaf
        feature_fraction: float = 1.0,          # Ratio of features per tree
        bagging_fraction: float = 0.8,          # Ratio of data per tree (bootstrap)
        bagging_freq: int = 5,                  # Frequency for bagging
        log_y: bool = False,                    # Log-transform objective values
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = None,
        seed: int = 0                           # Random seed
    ) -> None:
        # Initialize the superclass
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed
        )

        # LightGBM parameters
        self._lgb_params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': -1,
            'seed': seed,
            "log_y": log_y
        }
        self._log_y = log_y
        self._lgb_model = None  # Placeholder for LightGBM model

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "objective": self._lgb_params['objective'],
                "boosting_type": self._lgb_params['boosting_type'],
                "n_estimators": self._lgb_params['n_estimators'],
                "learning_rate": self._lgb_params['learning_rate'],
                "max_depth": self._lgb_params['max_depth'],
                "num_leaves": self._lgb_params['num_leaves'],
                "min_data_in_leaf": self._lgb_params['min_data_in_leaf'],
                "feature_fraction": self._lgb_params['feature_fraction'],
                "bagging_fraction": self._lgb_params['bagging_fraction'],
                "bagging_freq": self._lgb_params['bagging_freq'],
                "verbose": self._lgb_params['verbose'],
                "seed": self._lgb_params['seed'],
                "log_y": self._log_y,

            }
        )

        return meta

    def _train(
      self,
      X_train: np.ndarray,
      y_train: np.ndarray,
      X_valid: Optional[np.ndarray] = None,
      y_valid: Optional[np.ndarray] = None,
      params: Optional[Dict[str, Any]] = None,
      num_boost_round: int = 100,
      early_stopping_rounds: Optional[int] = None,
      verbose_eval: bool = True
    ) -> lgb.Booster:
      """
      Train a LightGBM model with optional validation data and early stopping.

      Parameters:
      - X_train: Training features.
      - y_train: Training labels.
      - X_valid: Validation features (optional).
      - y_valid: Validation labels (optional).
      - params: Training parameters (optional).
      - num_boost_round: Number of boosting rounds.
      - early_stopping_rounds: Rounds for early stopping (optional).
      - verbose_eval: Verbosity of training output.

      Returns:
      - Trained LightGBM Booster model.
      """
      # Impute inactive features
      X_train = self._impute_inactive(X_train)
      y_train = y_train.flatten()

      # Create Dataset object for training
      train_data = lgb.Dataset(X_train, label=y_train)

      # Create Dataset object for validation if provided
      valid_data = None
      if X_valid is not None and y_valid is not None:
        X_valid = self._impute_inactive(X_valid)
        y_valid = y_valid.flatten()
        valid_data = lgb.Dataset(X_valid, label=y_valid)

      # Define default training parameters if none provided
      if params is None:
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }

      # Train the model with or without validation data
      if valid_data is not None:
        self._rf = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
      else:
        self._rf = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval
        )

      return self._rf
    
    def _init_data_container(
      self,
      X: np.ndarray,
      y: np.ndarray,
      categorical_feature: Optional[List[int]] = None
    ) -> lgb.Dataset:
      """
      Initializes a LightGBM Dataset with specified categorical features.

      Parameters
      ----------
      X : np.ndarray
          Input data points of shape [n_samples, n_features].
      y : np.ndarray
          Corresponding target values of shape [n_samples].
      categorical_feature : list of int, optional
          List of indices for categorical features. If None, no features are considered categorical.

      Returns
      -------
      dataset : lgb.Dataset
          The initialized LightGBM Dataset.
      """
      # Flatten the target array if necessary
      y = y.flatten()

      # Create the LightGBM Dataset
      dataset = lgb.Dataset(
          data=X,
          label=y,
          categorical_feature=categorical_feature
      )

      return dataset


    def _predict(
      self,
      X: np.ndarray,
      covariance_type: Optional[str] = "diagonal",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      """
      Predict mean and variance for the given input data using LightGBM.

      Parameters  
      ----------
      X : np.ndarray
          Input data points of shape [n_samples, n_features].
      covariance_type : str, optional
          Type of covariance to return. Only 'diagonal' is supported.

      Returns
      -------
      means : np.ndarray
          Predicted mean values of shape [n_samples, 1].
      vars_ : np.ndarray or None
          Predicted variances of shape [n_samples, 1] if available; otherwise, None.
      """
      if len(X.shape) != 2:
          raise ValueError(f"Expected 2D array, got {len(X.shape)}D array!")

      if X.shape[1] != self._num_features:
          raise ValueError(f"Rows in X should have {self._num_features} entries but have {X.shape[1]}!")

      if covariance_type != "diagonal":
          raise ValueError("`covariance_type` can only take 'diagonal' for this model.")

      assert self._model is not None, "Model has not been trained."

      # Impute inactive features if necessary
      X = self._impute_inactive(X)

      # Predict mean values
      means = self._model.predict(X, raw_score=False)

      # If log transformation was applied during training, apply inverse transformation
      if self._log_y:
        means = np.exp(means)

      # Estimate variance (uncertainty)
      if self._estimate_variance:
        # Example using quantile regression to estimate variance
        lower_quantile = self._model.predict(X, raw_score=False, pred_contrib=False, num_iteration=None, pred_leaf=False, pred_early_stop=False, pred_parameter={"quantile_alpha": 0.025})
        upper_quantile = self._model.predict(X, raw_score=False, pred_contrib=False, num_iteration=None, pred_leaf=False, pred_early_stop=False, pred_parameter={"quantile_alpha": 0.975})
        vars_ = ((upper_quantile - lower_quantile) / 2) ** 2
      else:
        vars_ = None

      means = means.reshape((-1, 1))
      if vars_ is not None:
        vars_ = vars_.reshape((-1, 1))

      return means, vars_

    def predict_marginalized(
      self,
      X: np.ndarray,
      instance_features: Optional[np.ndarray] = None,
      quantiles: Tuple[float, float] = (0.025, 0.975)
    ) -> Tuple[np.ndarray, np.ndarray]:
      """
      Predicts mean and variance marginalized over all instances using LightGBM.

      Parameters
      ----------
      X : np.ndarray
          Input data points of shape [n_samples, n_features].
      instance_features : np.ndarray, optional
          Instance-specific features of shape [n_instances, n_instance_features].
      quantiles : tuple of float
          Quantiles for uncertainty estimation (e.g., (0.025, 0.975) for 95% prediction interval).

      Returns
      -------
      means : np.ndarray
          Predicted mean values of shape [n_samples, 1].
      vars_ : np.ndarray
          Predicted variances of shape [n_samples, 1].
      """
      if len(X.shape) != 2:
        raise ValueError(f"Expected 2D array, got {len(X.shape)}D array!")

      if instance_features is not None:
        if len(instance_features.shape) != 2:
            raise ValueError(f"Expected 2D array for instance_features, got {len(instance_features.shape)}D array!")
        if X.shape[0] != instance_features.shape[0]:
            raise ValueError("Number of samples in X and instance_features must match.")
        # Concatenate instance-specific features with X
        X = np.hstack((X, instance_features))

      assert self._model is not None, "Model has not been trained."

      # Predict mean values
      means = self._model.predict(X, raw_score=False)

      # Estimate variance using quantile regression
      lower_quantile = self._model.predict(X, raw_score=False, pred_contrib=False, num_iteration=None, pred_leaf=False, pred_early_stop=False, pred_parameter={"quantile_alpha": quantiles[0]})
      upper_quantile = self._model.predict(X, raw_score=False, pred_contrib=False, num_iteration=None, pred_leaf=False, pred_early_stop=False, pred_parameter={"quantile_alpha": quantiles[1]})
      vars_ = ((upper_quantile - lower_quantile) / 2) ** 2

      means = means.reshape((-1, 1))
      vars_ = vars_.reshape((-1, 1))

      return means, vars_


