import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.datasets import load_breast_cancer, load_diabetes
import optuna
from optuna.pruners import BasePruner
from typing import Dict, List, Optional, Tuple, Union
import time
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FidelityConfig:
    """Configuration for fidelity levels"""
    n_estimators_ratios: List[float] = None
    data_sampling_ratios: List[float] = None
    min_n_estimators: int = 10
    
    def __post_init__(self):
        if self.n_estimators_ratios is None:
            self.n_estimators_ratios = [0.1, 0.2, 0.5, 0.75, 1.0]
        if self.data_sampling_ratios is None:
            self.data_sampling_ratios = [0.25, 0.5, 0.75, 1.0]

class PerformancePredictor(ABC):
    """Abstract base class for performance prediction models"""
    
    @abstractmethod
    def predict(self, current_performance: float, current_fidelity: float, 
                trial_data: Dict) -> float:
        """Predict final performance from current partial evaluation"""
        pass
    
    @abstractmethod
    def update(self, trial_history: List[Dict]):
        """Update predictor with new trial data"""
        pass

class ExponentialPerformancePredictor(PerformancePredictor):
    """Exponential decay model for performance prediction"""
    
    def __init__(self, alpha: float = 0.1, beta: float = 2.0):
        self.alpha = alpha
        self.beta = beta
        self.history = []
        self.fitted_params = {'alpha': alpha, 'beta': beta}
    
    def predict(self, current_performance: float, current_fidelity: float, 
                trial_data: Dict) -> float:
        """
        Predict final performance using exponential model:
        Performance_final ≈ Performance_current + α * exp(-β * fidelity)
        """
        if current_fidelity >= 1.0:
            return current_performance
        
        # Use fitted parameters if available, otherwise use defaults
        alpha = self.fitted_params['alpha']
        beta = self.fitted_params['beta']
        
        # Predict improvement potential
        improvement_potential = alpha * np.exp(-beta * current_fidelity)
        predicted_final = current_performance + improvement_potential
        
        return predicted_final
    
    def update(self, trial_history: List[Dict]):
        """Update model parameters based on historical data"""
        self.history = trial_history
        
        # Simple parameter updating based on recent trends
        if len(trial_history) > 5:
            recent_trials = trial_history[-5:]
            improvements = []
            
            for trial in recent_trials:
                if 'intermediate_scores' in trial and len(trial['intermediate_scores']) > 1:
                    scores = trial['intermediate_scores']
                    fidelities = trial.get('fidelities', [])
                    if len(scores) == len(fidelities) and len(scores) > 1:
                        final_improvement = scores[-1] - scores[0]
                        initial_fidelity = fidelities[0]
                        improvements.append((final_improvement, initial_fidelity))
            
            # Update alpha based on observed improvements
            if improvements:
                avg_improvement = np.mean([imp[0] for imp in improvements])
                self.fitted_params['alpha'] = max(0.01, min(0.5, avg_improvement))

class RFMultiFidelityPruner(BasePruner):
    """Custom Optuna pruner for Random Forest multi-fidelity optimization"""
    
    def __init__(self, 
                 performance_predictor: PerformancePredictor,
                 fidelity_config: FidelityConfig,
                 safety_margin: float = 0.05,
                 min_trials_for_pruning: int = 10,
                 confidence_level: float = 0.95):
        self.performance_predictor = performance_predictor
        self.fidelity_config = fidelity_config
        self.safety_margin = safety_margin
        self.min_trials_for_pruning = min_trials_for_pruning
        self.confidence_level = confidence_level
        self.trial_history = []
    
    def prune(self, study: optuna.Study, trial: optuna.Trial) -> bool:
        """Decide whether to prune the current trial"""
        try:
            # Don't prune if we don't have enough trials yet
            if len(study.trials) < self.min_trials_for_pruning:
                return False
            
            # Get current trial's intermediate values
            intermediate_values = trial.intermediate_values
            if not intermediate_values:
                return False
            
            # Get the latest reported value and its corresponding step
            latest_step = max(intermediate_values.keys())
            current_performance = intermediate_values[latest_step]
            current_fidelity = self._step_to_fidelity(latest_step)
            
            # Predict final performance
            trial_data = {
                'params': trial.params,
                'intermediate_scores': list(intermediate_values.values()),
                'fidelities': [self._step_to_fidelity(step) for step in intermediate_values.keys()]
            }
            
            predicted_final = self.performance_predictor.predict(
                current_performance, current_fidelity, trial_data
            )
            
            # Get current best value from completed trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                return False
            
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                current_best = max(t.value for t in completed_trials)
                should_prune = predicted_final < (current_best - self.safety_margin)
            else:
                current_best = min(t.value for t in completed_trials)
                should_prune = predicted_final > (current_best + self.safety_margin)
            
            # Log pruning decision
            if should_prune:
                logger.info(f"Pruning trial {trial.number}: predicted={predicted_final:.4f}, "
                           f"best={current_best:.4f}, margin={self.safety_margin:.4f}")
            
            return should_prune
            
        except Exception as e:
            logger.warning(f"Error in pruning decision for trial {trial.number}: {e}")
            return False
    
    def _step_to_fidelity(self, step: int) -> float:
        """Convert step number to fidelity level"""
        if step < len(self.fidelity_config.n_estimators_ratios):
            return self.fidelity_config.n_estimators_ratios[step]
        return 1.0

class MultiFidelityRandomForestOptimizer:
    """Main optimizer class combining all components"""
    
    def __init__(self, 
                 task_type: str = 'classification',
                 fidelity_config: FidelityConfig = None,
                 performance_predictor: PerformancePredictor = None,
                 n_trials: int = 100,
                 random_state: int = 42):
        
        self.task_type = task_type
        self.fidelity_config = fidelity_config or FidelityConfig()
        self.performance_predictor = performance_predictor or ExponentialPerformancePredictor()
        self.n_trials = n_trials
        self.random_state = random_state
        
        # Initialize pruner
        self.pruner = RFMultiFidelityPruner(
            performance_predictor=self.performance_predictor,
            fidelity_config=self.fidelity_config
        )
        
        # Results storage
        self.study = None
        self.best_trial = None
        self.optimization_time = None
        self.trial_results = []
    
    def _create_objective(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray):
        """Create the objective function for Optuna optimization"""
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            max_n_estimators = trial.suggest_int('n_estimators', 50, 1000)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
            # Initialize model based on task type
            if self.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=1,  # Start with 1, will be increased
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=self.random_state,
                    warm_start=True,
                    oob_score=True,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=1,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=self.random_state,
                    warm_start=True,
                    oob_score=True,
                    n_jobs=-1
                )
            
            # Multi-fidelity evaluation
            for step, fidelity_ratio in enumerate(self.fidelity_config.n_estimators_ratios):
                # Calculate current number of estimators
                current_n_estimators = max(
                    self.fidelity_config.min_n_estimators,
                    int(max_n_estimators * fidelity_ratio)
                )
                
                # Update model with new number of estimators
                model.n_estimators = current_n_estimators
                
                # Subsample data if specified
                if len(self.fidelity_config.data_sampling_ratios) > step:
                    data_ratio = self.fidelity_config.data_sampling_ratios[min(step, len(self.fidelity_config.data_sampling_ratios)-1)]
                    if data_ratio < 1.0:
                        n_samples = int(len(X_train) * data_ratio)
                        indices = np.random.choice(len(X_train), n_samples, replace=False)
                        X_train_sub = X_train[indices]
                        y_train_sub = y_train[indices]
                    else:
                        X_train_sub = X_train
                        y_train_sub = y_train
                else:
                    X_train_sub = X_train
                    y_train_sub = y_train
                
                # Train model
                model.fit(X_train_sub, y_train_sub)
                
                # Get performance score
                if self.task_type == 'classification':
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                    # Use OOB score if available and better
                    if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
                        score = max(score, model.oob_score_)
                else:
                    y_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, y_pred)  # Negative for maximization
                    if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
                        score = max(score, -model.oob_score_)
                
                # Report intermediate result
                trial.report(score, step)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return score
        
        return objective
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                test_size: float = 0.2) -> Dict:
        """Run the optimization process"""
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if self.task_type == 'classification' else None
        )
        
        # Create study
        direction = 'maximize' if self.task_type == 'classification' else 'maximize'  # Both maximize (accuracy or negative MSE)
        self.study = optuna.create_study(
            direction=direction,
            pruner=self.pruner,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Create objective function
        objective = self._create_objective(X_train, y_train, X_val, y_val)
        
        # Run optimization
        start_time = time.time()
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[self._trial_callback]
        )
        
        self.optimization_time = time.time() - start_time
        self.best_trial = self.study.best_trial
        
        # Update performance predictor with all trial data
        self.performance_predictor.update(self.trial_results)
        
        return self._get_results()
    
    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback function to collect trial data"""
        trial_data = {
            'number': trial.number,
            'params': trial.params,
            'value': trial.value,
            'state': trial.state,
            'intermediate_values': trial.intermediate_values,
            'datetime_start': trial.datetime_start,
            'datetime_complete': trial.datetime_complete
        }
        self.trial_results.append(trial_data)
    
    def _get_results(self) -> Dict:
        """Compile optimization results"""
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        results = {
            'best_params': self.best_trial.params if self.best_trial else None,
            'best_value': self.best_trial.value if self.best_trial else None,
            'optimization_time': self.optimization_time,
            'n_completed_trials': len(completed_trials),
            'n_pruned_trials': len(pruned_trials),
            'total_trials': len(self.study.trials),
            'pruning_rate': len(pruned_trials) / len(self.study.trials) if self.study.trials else 0,
            'study': self.study,
            'trial_results': self.trial_results
        }
        
        return results

class BaselineOptimizer:
    """Baseline optimizer without early stopping for comparison"""
    
    def __init__(self, task_type: str = 'classification', n_trials: int = 100, random_state: int = 42):
        self.task_type = task_type
        self.n_trials = n_trials
        self.random_state = random_state
        self.study = None
        self.optimization_time = None
    
    def optimize(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """Run baseline optimization without early stopping"""
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if self.task_type == 'classification' else None
        )
        
        def objective(trial):
            # Sample hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 50, 1000)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
            # Create and train model
            if self.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            if self.task_type == 'classification':
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
            else:
                y_pred = model.predict(X_val)
                score = -mean_squared_error(y_val, y_pred)
            
            return score
        
        # Create study without pruner
        direction = 'maximize'
        self.study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Run optimization
        start_time = time.time()
        self.study.optimize(objective, n_trials=self.n_trials)
        self.optimization_time = time.time() - start_time
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'optimization_time': self.optimization_time,
            'n_completed_trials': len(self.study.trials),
            'n_pruned_trials': 0,
            'total_trials': len(self.study.trials),
            'pruning_rate': 0.0,
            'study': self.study
        }

def run_comparative_experiment(X: np.ndarray, y: np.ndarray, 
                             task_type: str = 'classification',
                             n_trials: int = 100,
                             n_runs: int = 5) -> Dict:
    """Run comparative experiment between baseline and multi-fidelity approach"""
    
    baseline_results = []
    multifidelity_results = []
    
    for run in range(n_runs):
        print(f"Running experiment {run + 1}/{n_runs}...")
        
        # Baseline
        baseline_opt = BaselineOptimizer(task_type=task_type, n_trials=n_trials, random_state=42+run)
        baseline_result = baseline_opt.optimize(X, y)
        baseline_results.append(baseline_result)
        
        # Multi-fidelity
        multifidelity_opt = MultiFidelityRandomForestOptimizer(
            task_type=task_type, 
            n_trials=n_trials, 
            random_state=42+run
        )
        multifidelity_result = multifidelity_opt.optimize(X, y)
        multifidelity_results.append(multifidelity_result)
    
    # Aggregate results
    def aggregate_results(results_list):
        metrics = ['best_value', 'optimization_time', 'pruning_rate']
        aggregated = {}
        
        for metric in metrics:
            values = [r[metric] for r in results_list if r[metric] is not None]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_ci'] = 1.96 * np.std(values) / np.sqrt(len(values))
        
        return aggregated
    
    comparison_results = {
        'baseline': aggregate_results(baseline_results),
        'multifidelity': aggregate_results(multifidelity_results),
        'raw_baseline': baseline_results,
        'raw_multifidelity': multifidelity_results
    }
    
    # Calculate speedup
    baseline_time = comparison_results['baseline']['optimization_time_mean']
    multifidelity_time = comparison_results['multifidelity']['optimization_time_mean']
    speedup = baseline_time / multifidelity_time if multifidelity_time > 0 else 0
    
    comparison_results['speedup_factor'] = speedup
    
    return comparison_results

# Example usage and testing
if __name__ == "__main__":
    # Load sample data
    from sklearn.datasets import load_breast_cancer, load_diabetes
    
    # Classification example
    print("Running classification example...")
    X_clf, y_clf = load_breast_cancer(return_X_y=True)
    
    clf_results = run_comparative_experiment(
        X_clf, y_clf, 
        task_type='classification',
        n_trials=50,
        n_runs=3
    )
    
    print("\nClassification Results:")
    print(f"Baseline - Best Score: {clf_results['baseline']['best_value_mean']:.4f} ± {clf_results['baseline']['best_value_ci']:.4f}")
    print(f"Baseline - Time: {clf_results['baseline']['optimization_time_mean']:.2f} ± {clf_results['baseline']['optimization_time_ci']:.2f} seconds")
    
    print(f"Multi-fidelity - Best Score: {clf_results['multifidelity']['best_value_mean']:.4f} ± {clf_results['multifidelity']['best_value_ci']:.4f}")
    print(f"Multi-fidelity - Time: {clf_results['multifidelity']['optimization_time_mean']:.2f} ± {clf_results['multifidelity']['optimization_time_ci']:.2f} seconds")
    print(f"Multi-fidelity - Pruning Rate: {clf_results['multifidelity']['pruning_rate_mean']:.2f}")
    
    print(f"Speedup Factor: {clf_results['speedup_factor']:.2f}x")
    
    # Regression example
    print("\n" + "="*50)
    print("Running regression example...")
    X_reg, y_reg = load_diabetes(return_X_y=True)
    
    reg_results = run_comparative_experiment(
        X_reg, y_reg,
        task_type='regression', 
        n_trials=50,
        n_runs=3
    )
    
    print("\nRegression Results:")
    print(f"Baseline - Best Score: {reg_results['baseline']['best_value_mean']:.4f} ± {reg_results['baseline']['best_value_ci']:.4f}")
    print(f"Baseline - Time: {reg_results['baseline']['optimization_time_mean']:.2f} ± {reg_results['baseline']['optimization_time_ci']:.2f} seconds")
    
    print(f"Multi-fidelity - Best Score: {reg_results['multifidelity']['best_value_mean']:.4f} ± {reg_results['multifidelity']['best_value_ci']:.4f}")
    print(f"Multi-fidelity - Time: {reg_results['multifidelity']['optimization_time_mean']:.2f} ± {reg_results['multifidelity']['optimization_time_ci']:.2f} seconds")
    print(f"Multi-fidelity - Pruning Rate: {reg_results['multifidelity']['pruning_rate_mean']:.2f}")
    
    print(f"Speedup Factor: {reg_results['speedup_factor']:.2f}x")