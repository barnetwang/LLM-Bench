#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
import logging
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')

class BayesianOptimizer:
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                 n_initial_points: int = 5, n_iterations: int = 20,
                 acquisition_function: str = 'ei'):
        """
        Args:
            param_bounds: 參數邊界，格式為 {'param_name': (min_val, max_val)}
            n_initial_points: 初始隨機點數量
            n_iterations: 優化迭代次數
            acquisition_function: 採集函數類型 ('ei', 'pi', 'ucb')
        """
        self.param_bounds = param_bounds
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.acquisition_function = acquisition_function
        
        self.param_names = list(param_bounds.keys())
        self.bounds_array = np.array([param_bounds[name] for name in self.param_names])
        self.X = []
        self.y = [] 
        self.best_score = -np.inf
        self.best_params = None
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * len(self.param_names), (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        self.logger = logging.getLogger(__name__)
        
    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        normalized = []
        for name in self.param_names:
            min_val, max_val = self.param_bounds[name]
            val = params[name]
            normalized_val = (val - min_val) / (max_val - min_val)
            normalized.append(normalized_val)
        return np.array(normalized)
    
    def _denormalize_params(self, normalized_params: np.ndarray) -> Dict[str, float]:
        params = {}
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[name]
            normalized_val = normalized_params[i]
            val = min_val + normalized_val * (max_val - min_val)
            params[name] = val
        return params
    
    def _generate_random_params(self) -> Dict[str, float]:
        params = {}
        for name in self.param_names:
            min_val, max_val = self.param_bounds[name]
            params[name] = np.random.uniform(min_val, max_val)
        return params
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-8)
        
        improvement = mu - self.best_score - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _probability_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-8)
        
        improvement = mu - self.best_score - xi
        Z = improvement / sigma
        pi = norm.cdf(Z)
        
        return pi
    
    def _upper_confidence_bound(self, X: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        mu, sigma = self.gp.predict(X, return_std=True)
        ucb = mu + kappa * sigma
        return ucb
    
    def _acquisition_function_value(self, X: np.ndarray) -> np.ndarray:
        if self.acquisition_function == 'ei':
            return self._expected_improvement(X)
        elif self.acquisition_function == 'pi':
            return self._probability_improvement(X)
        elif self.acquisition_function == 'ucb':
            return self._upper_confidence_bound(X)
        else:
            raise ValueError(f"不支援的採集函數: {self.acquisition_function}")
    
    def _optimize_acquisition(self, n_restarts: int = 10) -> Dict[str, float]:
        best_acq = -np.inf
        best_params = None
        
        for _ in range(n_restarts):
            x0 = np.random.uniform(0, 1, len(self.param_names))
            for _ in range(50):
                x_candidate = x0 + np.random.normal(0, 0.1, len(self.param_names))
                x_candidate = np.clip(x_candidate, 0, 1)
                
                acq_val = self._acquisition_function_value(x_candidate.reshape(1, -1))[0]
                
                if acq_val > best_acq:
                    best_acq = acq_val
                    best_params = x_candidate.copy()
                    x0 = x_candidate.copy()
        
        if best_params is None:
            best_params = np.random.uniform(0, 1, len(self.param_names))
        
        return self._denormalize_params(best_params)
    
    def suggest_next_point(self) -> Dict[str, float]:
        if len(self.X) < self.n_initial_points:
            return self._generate_random_params()
        else:
            return self._optimize_acquisition()
    
    def update(self, params: Dict[str, float], score: float):
        """     
        Args:
            params: 測試的參數組合
            score: 獲得的評分
        """
        X_normalized = self._normalize_params(params)
        self.X.append(X_normalized)
        self.y.append(score)
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            self.logger.info(f"發現新的最佳結果: {score:.4f} with params: {params}")
        if len(self.X) >= 2:
            X_array = np.array(self.X)
            y_array = np.array(self.y)
            self.gp.fit(X_array, y_array)
    
    def get_best_result(self) -> Tuple[Dict[str, float], float]:
        return self.best_params, self.best_score
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        history = []
        for i, (X_normalized, score) in enumerate(zip(self.X, self.y)):
            params = self._denormalize_params(X_normalized)
            history.append({
                'iteration': i + 1,
                'parameters': params,
                'score': score,
                'is_best': score == self.best_score
            })
        return history
    
    def predict_score(self, params: Dict[str, float]) -> Tuple[float, float]:
        """
        Returns:
            (預測評分, 不確定性)
        """
        if len(self.X) < 2:
            return 0.0, 1.0
        
        X_normalized = self._normalize_params(params)
        mu, sigma = self.gp.predict(X_normalized.reshape(1, -1), return_std=True)
        return mu[0], sigma[0]
    
    def get_parameter_importance(self) -> Dict[str, float]:
        if len(self.X) < 2:
            return {name: 1.0 / len(self.param_names) for name in self.param_names}
        kernel_params = self.gp.kernel_.get_params()
        if 'k2__length_scale' in kernel_params:
            length_scales = kernel_params['k2__length_scale']
            if isinstance(length_scales, (list, np.ndarray)):
                importance = 1.0 / np.array(length_scales)
                importance = importance / importance.sum()
                return {name: importance[i] for i, name in enumerate(self.param_names)}
        
        return {name: 1.0 / len(self.param_names) for name in self.param_names}

class AdaptiveBayesianOptimizer(BayesianOptimizer):
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                 n_initial_points: int = 5, n_iterations: int = 20,
                 early_stopping_patience: int = 5, improvement_threshold: float = 0.01):
        """
        Args:
            early_stopping_patience: 早停耐心值
            improvement_threshold: 改進閾值
        """
        super().__init__(param_bounds, n_initial_points, n_iterations)
        self.early_stopping_patience = early_stopping_patience
        self.improvement_threshold = improvement_threshold
        self.no_improvement_count = 0
        self.last_best_score = -np.inf
        
    def update(self, params: Dict[str, float], score: float):
        super().update(params, score)
        if score > self.last_best_score + self.improvement_threshold:
            self.no_improvement_count = 0
            self.last_best_score = score
        else:
            self.no_improvement_count += 1
    
    def should_stop_early(self) -> bool:
        return self.no_improvement_count >= self.early_stopping_patience
    
    def get_remaining_iterations(self) -> int:
        if self.should_stop_early():
            return 0
        return max(0, self.n_iterations - len(self.X))
    
    def suggest_next_point(self) -> Optional[Dict[str, float]]:
        if self.should_stop_early():
            return None
        return super().suggest_next_point()
DEFAULT_PARAM_BOUNDS = {
    'temperature': (0.1, 1.5),
    'top_p': (0.5, 1.0),
    'top_k': (5, 150)
}

def create_optimizer_for_quality_tuning() -> AdaptiveBayesianOptimizer:
    return AdaptiveBayesianOptimizer(
        param_bounds=DEFAULT_PARAM_BOUNDS,
        n_initial_points=8,
        n_iterations=25,
        early_stopping_patience=5,
        improvement_threshold=0.02
    )
