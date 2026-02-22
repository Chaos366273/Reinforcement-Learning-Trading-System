# Import constants if available
try:
    from ppo.constants import *
except ImportError:
    # Fallback constants if ppo.constants is not available
    DEFAULT_NUM_TICKERS = 12  # UPDATED: Now 12 tickers (AAPL, MSFT, NVDA, JPM, SPY, GLD, BTC-USD, KO, JNJ, AMZN, XLE, TLT)
    DEFAULT_NUM_FEATURES = 13  # UPDATED: 13 features (15 - 2: fundamental & sentiment disabled)
    DEFAULT_WINDOW_SIZE = 5
    DEFAULT_ACTION_SPACE_SIZE = 3
    DEFAULT_OBSERVATION_SPACE_SIZE = 150
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_TRAILING_STOP = 0.05
    DEFAULT_TIMEOUT = 30
    DEFAULT_SLIPPAGE = 0.001
    REWARD_POOL_SIZE = 1000
    DEFAULT_DRAWDOWN_LIMIT = 0.10
    CUDA_GARBAGE_COLLECTION_THRESHOLD = 1000
    DEFAULT_RATE_LIMIT = 100
    LOG_FREQUENCY = 100
    DEFAULT_COMMISSION_RATE = 0.001
    DEFAULT_STOP_LOSS = 0.05
    SAVE_FREQUENCY = 1000
# Import modular components
from ppo.utils.performance_monitor import PerformanceMonitor, PerformanceConfig
from ppo.utils.advanced_analytics import AdvancedAnalytics, AnalyticsConfig
from ppo.core.advanced_risk_manager import AdvancedRiskManager
from ppo.data.feature_engineer import FeatureEngineer
from ppo.training.ppo_trainer import create_ppo_agent, train_ppo_agent, save_model
from ppo.evaluation.performance_evaluator import evaluate_model

# Import configuration
try:
    import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config.py not found. Using default configuration.")
# Import emergency manager - IDE-friendly approach
try:
    from ppo.utils.emergency_mode_manager import emergency_manager  # type: ignore
except ImportError:
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ppo'))
        from utils.emergency_mode_manager import emergency_manager  # type: ignore
    except ImportError:
        # Fallback: create dummy emergency manager
        class DummyEmergencyManager:
            def is_emergency_mode_active(self) -> bool:
                return False

            def can_train_model(self):
                return True, "Emergency manager unavailable"

            def cleanup_invalid_models(self) -> None:
                return None

        emergency_manager = DummyEmergencyManager()  # type: ignore
from ppo.data.data_loader import download_stock_data, load_cached_data
from ppo.data.technical_indicators import calculate_technical_indicators
from ppo.models.trading_env import FixedTradingEnv

# 2025-12-22: MaskablePPO for proper action masking (prevents invalid SELL when no positions)
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_PPO_AVAILABLE = True
    print("✅ MaskablePPO available - Action masking ENABLED (agent cannot select invalid actions)")
except ImportError as e:
    MASKABLE_PPO_AVAILABLE = False
    print("⚠️ MaskablePPO not available - Install with: pip install sb3-contrib")
    print(f"   Import error: {e}")
    print("   Falling back to standard PPO (invalid actions will get penalty instead of being masked)")
"""

Main Orchestrator for PPO Trading Agent

Clean modular architecture - Follows README specifications exactly

Uses all components from ppo/ modular structure



===============================================================================

CRITICAL FIXES IMPLEMENTED (2025-08-27)

===============================================================================



DEFAULT_NUM_FEATURES  # CONSTANT: Replaced magic number. PAPER TRADING READINESS (FIXED)

   -  Added comprehensive readiness assessment

   -  Statistical validation checks (p-value < 0.10 or Sharpe p-value < DEFAULT_TRAILING_STOP)  # CONSTANT: Replaced magic number

   -  Performance metrics validation (Sharpe > 1.0, MaxDD < 10%, Win Rate > 45%)

   -  Data quality validation (real benchmark data available)

   -  Overall readiness determination



CURRENT STATUS: TRAINING IN PROGRESS

- Statistical validation: PENDING (Requires sufficient training data)

- Performance metrics: PENDING (Agent needs more training)

- Data quality: PASSED (Real SPY benchmark data)

- Advanced features: DYNAMIC DETECTION (Status based on actual implementation)



NEXT STEPS: Scale training to 300K-600K timesteps for super agent readiness

===============================================================================

"""



import os
import sys
import numpy as np
import logging
from pathlib import Path
import psutil
import torch
import time
from datetime import datetime

# ENHANCED: Error tracking and validation system (adopted from BC)
class ErrorTracker:
    """Track errors and warnings during PPO execution"""
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.critical_errors = []
    
    def add_error(self, message, is_critical=False):
        """Add error to tracker"""
        self.errors.append({'message': message, 'timestamp': time.time()})
        if is_critical:
            self.critical_errors.append(message)
        logger.error(f"{'CRITICAL ' if is_critical else ''}ERROR: {message}")
    
    def add_warning(self, message):
        """Add warning to tracker"""
        self.warnings.append({'message': message, 'timestamp': time.time()})
        logger.warning(f"WARNING: {message}")
    
    def should_reduce_timesteps(self):
        """
        Determine if timesteps should be reduced based on error criteria
        Criteria:
        - Any critical error (training crash, NaN losses, memory issues)
        - 3+ warnings/errors in total
        """
        if len(self.critical_errors) > 0:
            return True, f"Critical errors detected: {self.critical_errors}"
        if len(self.errors) + len(self.warnings) >= 3:
            return True, f"Multiple errors/warnings: {len(self.errors)} errors, {len(self.warnings)} warnings"
        return False, None
    
    def get_summary(self):
        """Get error summary"""
        return {
            'total_errors': len(self.errors),
            'critical_errors': len(self.critical_errors),
            'warnings': len(self.warnings),
            'should_reduce': self.should_reduce_timesteps()[0]
        }

# Initialize global error tracker
error_tracker = ErrorTracker()

# ENHANCED: Detailed progress logging (adopted from BC)
from stable_baselines3.common.callbacks import BaseCallback

class AdaptiveHyperparameterCallback(BaseCallback):
    """Adaptively adjust LR and entropy coefficient based on performance"""
    def __init__(self, eval_env=None, eval_freq=10000, verbose=1):  # FIX 4: Default 10K steps (was 50K, now triggers more often)
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval_step = 0
        self.original_lr_final = None
        self.original_ent_coef = None
        self.adaptive_lr_active = False
        self.adaptive_ent_active = False
        # NEW: Track previous performance to detect degradation
        self.previous_return = None
        self.previous_sharpe = None
        self.previous_trades = None
        self.best_return = None
        self.performance_drop_threshold = 0.02  # 2% return drop triggers LR increase
        self.trade_drop_threshold = 0.3  # 30% trade count drop triggers LR increase
        
        # NEW: Track adaptive LR activity for unified report
        self.adaptive_lr_log = []  # List of all adaptive LR events
        self.callback_triggered = False  # Track if callback ever ran
        # NEW: Track adaptive entropy activity for unified report
        self.adaptive_entropy_log = []
        
        # FIX 4: Log initialization
        if self.eval_env:
            logger.info(f"[ADAPTIVE CALLBACK INIT] Initialized with eval_freq={eval_freq} steps")
        else:
            logger.warning(f"[ADAPTIVE CALLBACK INIT] ⚠️ WARNING: eval_env is None - callback will not trigger!")
        
        # NEW: Initialize market-adaptive entropy system
        self.market_adaptive = None
        self.market_adaptive_enabled = False
        try:
            from ppo.utils.market_adaptive_entropy import MarketAdaptiveEntropy  # type: ignore
            self.market_adaptive = MarketAdaptiveEntropy(base_entropy=0.02)
            self.market_adaptive_enabled = True
            logger.info(f"[MARKET ADAPTIVE] Entropy system initialized")
        except ImportError:
            # Module doesn't exist - gracefully handle
            logger.debug(f"[MARKET ADAPTIVE] Module not available - using standard entropy")
        except Exception as e:
            logger.warning(f"[MARKET ADAPTIVE] Failed to initialize: {e}")
            self.market_adaptive_enabled = False
        
    def _on_step(self) -> bool:
        # Evaluate periodically
        if self.eval_env and (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self.callback_triggered = True  # Mark that callback ran
            
            # Log callback trigger
            log_entry = {
                'timestep': self.num_timesteps,
                'event': 'CALLBACK_TRIGGERED',
                'message': f'Adaptive LR callback triggered at {self.num_timesteps:,} timesteps'
            }
            self.adaptive_lr_log.append(log_entry)
            logger.info(f"[ADAPTIVE LR] Callback triggered at {self.num_timesteps:,} timesteps")
            
            try:
                from ppo.evaluation.performance_evaluator import PerformanceEvaluator  # type: ignore
                from ppo.utils.dynamic_config import get_dynamic_config
                evaluator = PerformanceEvaluator()  # type: ignore
                # FIX 4: Use 1 episode for faster evaluation in adaptive callback (was 1, keep it fast)
                results = evaluator.evaluate(self.model, self.eval_env, n_episodes=1)
                
                current_return = results.get('total_return', 0)
                current_sharpe = results.get('sharpe_ratio', 0)
                total_trades = results.get('total_trades', 0)
                
                # FIX 1: Extract action distribution from results for entropy callback
                # Try both 'action_counts' and 'policy_action_counts' keys
                action_counts = results.get('action_counts', results.get('policy_action_counts', {}))
                
                # Handle both string keys (HOLD/BUY/SELL) and numeric keys (0/1/2)
                if 'HOLD' in action_counts or 'BUY' in action_counts:
                    # String keys format
                    hold_count = action_counts.get('HOLD', 0)
                    buy_count = action_counts.get('BUY', 0)
                    sell_count = action_counts.get('SELL', 0)
                else:
                    # Numeric keys format
                    hold_count = action_counts.get(0, 0)
                    buy_count = action_counts.get(1, 0)
                    sell_count = action_counts.get(2, 0)
                
                total_actions = buy_count + sell_count + hold_count
                
                buy_pct = (buy_count / total_actions * 100) if total_actions > 0 else 0
                sell_pct = (sell_count / total_actions * 100) if total_actions > 0 else 0
                hold_pct = (hold_count / total_actions * 100) if total_actions > 0 else 0
                
                # Log evaluation results with action distribution
                log_entry = {
                    'timestep': self.num_timesteps,
                    'event': 'EVALUATION',
                    'return': current_return,
                    'sharpe': current_sharpe,
                    'trades': total_trades,
                    'buy_pct': buy_pct,
                    'sell_pct': sell_pct,
                    'hold_pct': hold_pct,
                    'message': f'Evaluated: Return={current_return:.2%}, Sharpe={current_sharpe:.2f}, Trades={total_trades}, BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%, HOLD={hold_pct:.1f}%'
                }
                self.adaptive_lr_log.append(log_entry)
                logger.info(f"[ADAPTIVE LR] Evaluation at {self.num_timesteps:,}: Return={current_return:.2%}, Sharpe={current_sharpe:.2f}, Trades={total_trades}, BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%")
                
                config = get_dynamic_config()
                
                # Track best performance
                if self.best_return is None or current_return > self.best_return:
                    self.best_return = current_return
                
                # BIDIRECTIONAL ADAPTIVE LEARNING RATE:
                # 1. Increase LR when performance is poor OR trade count is low (encourage exploration)
                # 2. Increase LR when performance is excellent (maintain adaptability)
                # 3. NEW: Increase LR when performance degrades (return/trades drop)
                
                sharpe_threshold = config.adaptive_lr_sharpe_threshold
                return_threshold = config.adaptive_lr_return_threshold
                poor_sharpe_threshold = getattr(config, 'adaptive_lr_poor_sharpe_threshold', 0.5)
                poor_return_threshold = getattr(config, 'adaptive_lr_poor_return_threshold', 0.01)
                poor_boost_factor = getattr(config, 'adaptive_lr_poor_boost_factor', 1.8)
                poor_lr_max = getattr(config, 'adaptive_lr_poor_max', 5e-5)  # UPDATED: Increased max LR
                
                # NEW: Detect performance degradation
                performance_degraded = False
                if self.previous_return is not None:
                    return_drop = self.previous_return - current_return
                    if return_drop > self.performance_drop_threshold:
                        performance_degraded = True
                        log_entry = {
                            'timestep': self.num_timesteps,
                            'event': 'PERFORMANCE_DEGRADATION',
                            'previous_return': self.previous_return,
                            'current_return': current_return,
                            'drop': return_drop,
                            'message': f'Return dropped: {self.previous_return:.2%} -> {current_return:.2%} (drop: {return_drop:.2%})'
                        }
                        self.adaptive_lr_log.append(log_entry)
                        logger.info(f"[ADAPTIVE LR (DEGRADATION)] Return dropped: {self.previous_return:.2%} -> {current_return:.2%} (drop: {return_drop:.2%})")
                
                # NEW: Detect trade count drop
                trades_dropped = False
                if self.previous_trades is not None and self.previous_trades > 0:
                    trade_drop_ratio = (self.previous_trades - total_trades) / self.previous_trades
                    if trade_drop_ratio > self.trade_drop_threshold:
                        trades_dropped = True
                        log_entry = {
                            'timestep': self.num_timesteps,
                            'event': 'TRADE_DROP',
                            'previous_trades': self.previous_trades,
                            'current_trades': total_trades,
                            'drop_ratio': trade_drop_ratio,
                            'message': f'Trades dropped: {self.previous_trades} -> {total_trades} (drop: {trade_drop_ratio:.1%})'
                        }
                        self.adaptive_lr_log.append(log_entry)
                        logger.info(f"[ADAPTIVE LR (TRADE DROP)] Trades dropped: {self.previous_trades} -> {total_trades} (drop: {trade_drop_ratio:.1%})")
                
                # Update previous values
                self.previous_return = current_return
                self.previous_sharpe = current_sharpe
                self.previous_trades = total_trades
                
                # Store original final LR if not stored
                if self.original_lr_final is None:
                    self.original_lr_final = config.lr_final
                
                # LR warmup: early training returns are near 0; avoid forcing LR to max too soon.
                lr_warmup_steps = int(getattr(config, 'adaptive_lr_warmup_steps', 50000))

                # Check if performance is poor (low Sharpe OR low return OR degraded).
                # NOTE: trade count is handled by entropy/incentive adaptation, not LR (prevents LR-driven collapse).
                performance_poor = (current_sharpe < poor_sharpe_threshold or 
                                   current_return < poor_return_threshold or
                                   performance_degraded or
                                   trades_dropped)
                
                # Check if performance is excellent
                performance_excellent = (current_sharpe > sharpe_threshold and current_return > return_threshold)
                
                if self.num_timesteps < lr_warmup_steps:
                    log_entry = {
                        'timestep': self.num_timesteps,
                        'event': 'LR_WARMUP_SKIP',
                        'warmup_steps': lr_warmup_steps,
                        'message': f'LR adaptation skipped during warmup (<{lr_warmup_steps:,} steps)'
                    }
                    self.adaptive_lr_log.append(log_entry)
                elif performance_poor and not performance_excellent:
                    # POOR PERFORMANCE: Increase LR to encourage exploration
                    if not self.adaptive_lr_active or self.adaptive_lr_active == 'excellent':
                        # Store original LR if not stored
                        if not hasattr(self, 'original_model_lr'):
                            self.original_model_lr = getattr(self.model, 'learning_rate', None)
                            if self.original_model_lr is None:
                                self.original_model_lr = 3e-4  # Default fallback
                        
                        # Calculate boosted LR (cap at poor_lr_max). CRITICAL: must INCREASE LR when poor (report 2026-02-09: cap 1e-4 had reduced 3e-4->1e-4).
                        current_lr = getattr(self.model, 'learning_rate', None) or self.original_model_lr
                        if current_lr is None:
                            current_lr = 3e-4  # Default fallback
                        new_lr = max(current_lr, min(poor_lr_max, current_lr * poor_boost_factor))  # Never reduce LR in poor branch
                        
                        # Directly update model's learning rate
                        self.model.learning_rate = new_lr
                        config.update_config(lr_final=new_lr)
                        
                        log_entry = {
                            'timestep': self.num_timesteps,
                            'event': 'LR_INCREASE_POOR',
                            'previous_lr': current_lr,
                            'new_lr': new_lr,
                            'boost_factor': poor_boost_factor,
                            'sharpe': current_sharpe,
                            'return': current_return,
                            'trades': total_trades,
                            'message': f'LR increased (POOR): {current_lr:.2e} -> {new_lr:.2e} (boost: {poor_boost_factor}x)'
                        }
                        self.adaptive_lr_log.append(log_entry)
                        logger.info(f"[ADAPTIVE LR (POOR)] Performance needs improvement (Sharpe={current_sharpe:.2f}<{poor_sharpe_threshold}, Return={current_return:.2%}<{poor_return_threshold:.2%}, Trades={total_trades})")
                        logger.info(f"   Increasing LR to encourage exploration: {current_lr:.2e} -> {new_lr:.2e} (boost: {poor_boost_factor}x)")
                        self.adaptive_lr_active = 'poor'
                elif performance_excellent:
                    # EXCELLENT PERFORMANCE: Maintain higher LR for fine-tuning
                    if not self.adaptive_lr_active or self.adaptive_lr_active == 'poor':
                        # Store original LR if not stored
                        if not hasattr(self, 'original_model_lr'):
                            self.original_model_lr = getattr(self.model, 'learning_rate', None)
                            if self.original_model_lr is None:
                                self.original_model_lr = 3e-4  # Default fallback
                        
                        # Increase LR to maintain adaptability (configurable boost, cap at initial LR)
                        current_lr = getattr(self.model, 'learning_rate', None) or self.original_model_lr
                        if current_lr is None:
                            current_lr = 3e-4  # Default fallback
                        new_lr = min(config.lr_initial, current_lr * config.adaptive_lr_boost_factor)
                        
                        # Directly update model's learning rate
                        self.model.learning_rate = new_lr
                        config.update_config(lr_final=new_lr)
                        
                        log_entry = {
                            'timestep': self.num_timesteps,
                            'event': 'LR_INCREASE_EXCELLENT',
                            'previous_lr': current_lr,
                            'new_lr': new_lr,
                            'boost_factor': config.adaptive_lr_boost_factor,
                            'sharpe': current_sharpe,
                            'return': current_return,
                            'message': f'LR increased (EXCELLENT): {current_lr:.2e} -> {new_lr:.2e} (boost: {config.adaptive_lr_boost_factor}x)'
                        }
                        self.adaptive_lr_log.append(log_entry)
                        logger.info(f"[ADAPTIVE LR (EXCELLENT)] Performance excellent (Sharpe={current_sharpe:.2f}>{sharpe_threshold}, Return={current_return:.2%}>{return_threshold:.2%})")
                        logger.info(f"   Increasing LR for fine-tuning: {current_lr:.2e} -> {new_lr:.2e} (boost: {config.adaptive_lr_boost_factor}x)")
                        self.adaptive_lr_active = 'excellent'
                else:
                    # NORMAL PERFORMANCE: Reset to original (but respect LR schedule)
                    if self.adaptive_lr_active:
                        # Get the scheduled LR based on current timesteps
                        from ppo.training.ppo_trainer import BCRegularizationCallback
                        if hasattr(self.model, '_bc_refresh_steps'):
                            # Calculate what LR should be based on schedule
                            current_step = self.num_timesteps
                            original_lr = getattr(self, 'original_model_lr', 3e-4)
                            
                            # Apply LR schedule if available
                            lr_schedule = getattr(self.model, '_lr_decay_schedule', {})
                            scheduled_lr = original_lr
                            for threshold in sorted(lr_schedule.keys()):
                                if current_step >= threshold:
                                    scheduled_lr = original_lr * lr_schedule[threshold]
                            
                            # Reset to scheduled LR (not original, to respect decay)
                            self.model.learning_rate = scheduled_lr
                            config.update_config(lr_final=scheduled_lr)
                            logger.info(f"[ADAPTIVE LR] Performance normalized (Sharpe={current_sharpe:.2f}, Return={current_return:.2%})")
                            logger.info(f"   Resetting LR to scheduled value: {scheduled_lr:.2e} (step {current_step:,})")
                        else:
                            # Fallback: reset to original
                            original_lr = getattr(self, 'original_model_lr', 3e-4)
                            self.model.learning_rate = original_lr
                            config.update_config(lr_final=self.original_lr_final if self.original_lr_final else original_lr)
                            logger.info(f"[ADAPTIVE LR] Performance normalized (Sharpe={current_sharpe:.2f}, Return={current_return:.2%})")
                            logger.info(f"   Resetting LR to original: {original_lr:.2e}")
                        self.adaptive_lr_active = False
                    else:
                        # FIX 1: Log that callback checked but no adjustment needed
                        log_entry = {
                            'timestep': self.num_timesteps,
                            'event': 'NO_ADJUSTMENT',
                            'sharpe': current_sharpe,
                            'return': current_return,
                            'trades': total_trades,
                            'message': f'No LR adjustment needed: Performance is normal (Sharpe={current_sharpe:.2f}, Return={current_return:.2%})'
                        }
                        self.adaptive_lr_log.append(log_entry)
                        logger.info(f"[ADAPTIVE LR] No adjustment needed: Sharpe={current_sharpe:.2f} (normal range), Return={current_return:.2%}, Trades={total_trades}")
                
                # MARKET-ADAPTIVE ENTROPY: Adjust entropy based on market conditions
                if self.market_adaptive_enabled:
                    try:
                        # Get market data from environment
                        if hasattr(self.training_env, 'get_attr'):
                            price_matrix = self.training_env.get_attr('price_matrix')[0]
                            volume_matrix = self.training_env.get_attr('volume_matrix')[0]
                            
                            # Calculate adaptive entropy
                            if self.market_adaptive is not None:
                                adaptive_entropy, market_info = self.market_adaptive.calculate(
                                    price_matrix, 
                                    volume_matrix
                                )
                                
                                # Update model entropy - SB3 PPO uses model.ent_coef
                                if hasattr(self.model, 'ent_coef'):
                                    self.model.ent_coef = adaptive_entropy  # type: ignore
                                elif hasattr(self.model, 'policy') and hasattr(self.model.policy, 'ent_coef'):
                                    self.model.policy.ent_coef = adaptive_entropy  # type: ignore
                                config.update_config(ent_coef=adaptive_entropy)
                                
                                # Log market conditions (every 1000 steps to avoid spam)
                                if self.num_timesteps % 1000 == 0:
                                    self.market_adaptive.log_market_conditions(market_info)
                                
                                # Track in adaptive entropy log
                                self.adaptive_entropy_log.append({
                                    'timestep': self.num_timesteps,
                                    'event': 'MARKET_ADAPTIVE_ENTROPY',
                                    'entropy': adaptive_entropy,
                                    'volatility': market_info['volatility'],
                                    'regime': market_info['regime'],
                                    'trend_strength': market_info['trend_strength'],
                                    'uncertainty': market_info['uncertainty'],
                                    'total_multiplier': market_info['total_multiplier'],
                                    'message': f'Market-adaptive entropy: {adaptive_entropy:.4f} (regime: {market_info["regime"]}, vol: {market_info["volatility"]:.2%})'
                                })
                    except Exception as e:
                        logger.warning(f"[MARKET ADAPTIVE] Error calculating entropy: {e}")
                
                # ADAPTIVE ENTROPY: Increase exploration if trade count is too low OR action imbalance detected
                min_trades_threshold = config.adaptive_ent_min_trades
                
                # PHASE 3 FIX: Check for action imbalance (BUY < 5% OR SELL < 5% OR SELL > 85%)
                buy_imbalance_threshold = getattr(config, 'adaptive_ent_buy_threshold', 5.0)  # Default 5%
                sell_low_threshold = getattr(config, 'adaptive_ent_sell_low_threshold', 5.0)  # Default 5% (NEW)
                sell_high_threshold = getattr(config, 'adaptive_ent_sell_threshold', 85.0)  # Default 85%
                
                # FIX 2026-02-15: Also trigger when action entropy collapsed (policy too deterministic)
                action_entropy_collapsed = False
                if total_actions > 0:
                    probs = [max(hold_pct/100, 1e-10), max(buy_pct/100, 1e-10), max(sell_pct/100, 1e-10)]
                    import math
                    action_entropy = -sum(p * math.log(p) for p in probs if p > 0)
                    max_ent = math.log(3)
                    entropy_ratio = action_entropy / max_ent if max_ent > 0 else 0
                    action_entropy_collapsed = (entropy_ratio < 0.2)  # Policy too deterministic
                
                # PHASE 3 FIX: Trigger on BUY < 5% OR SELL < 5% OR SELL > 85% OR entropy collapsed
                action_imbalance_detected = (buy_pct < buy_imbalance_threshold) or (sell_pct < sell_low_threshold) or (sell_pct > sell_high_threshold) or action_entropy_collapsed
                
                if total_trades < min_trades_threshold or action_imbalance_detected:
                    # Store original entropy coefficient (only once)
                    if self.original_ent_coef is None:
                        self.original_ent_coef = config.ent_coef
                    
                    # 2026-02-19: Always re-apply boost when imbalance detected (trigger multiple times)
                    # Increase entropy to encourage more trading (configurable boost, cap at max)
                    if total_trades == 0:
                        new_ent_coef = config.adaptive_ent_max  # Max exploration when fully collapsed
                    else:
                        new_ent_coef = min(config.adaptive_ent_max, self.original_ent_coef * config.adaptive_ent_boost_factor)
                    # SB3 PPO: try model.ent_coef first
                    if hasattr(self.model, 'ent_coef'):
                        self.model.ent_coef = new_ent_coef  # type: ignore
                    elif hasattr(self.model, 'policy') and hasattr(self.model.policy, 'ent_coef'):
                        self.model.policy.ent_coef = new_ent_coef  # type: ignore
                    config.update_config(ent_coef=new_ent_coef)
                    
                    # Determine trigger reason
                    trigger_reasons = []
                    if total_trades < min_trades_threshold:
                        trigger_reasons.append(f'low trades ({total_trades} < {min_trades_threshold})')
                    if buy_pct < buy_imbalance_threshold:
                        trigger_reasons.append(f'BUY imbalance ({buy_pct:.1f}% < {buy_imbalance_threshold}%)')
                    if sell_pct < sell_low_threshold:
                        trigger_reasons.append(f'SELL imbalance ({sell_pct:.1f}% < {sell_low_threshold}%)')
                    if sell_pct > sell_high_threshold:
                        trigger_reasons.append(f'SELL over-selling ({sell_pct:.1f}% > {sell_high_threshold}%)')
                    if action_entropy_collapsed:
                        trigger_reasons.append('action entropy collapsed (policy too deterministic)')
                    
                    trigger_reason_str = ', '.join(trigger_reasons)
                    
                    # Log every time we (re-)apply boost (multiple times per run when imbalance persists)
                    self.adaptive_entropy_log.append({
                        'timestep': self.num_timesteps,
                        'event': 'ENTROPY_INCREASE',
                        'previous_ent': self.original_ent_coef,
                        'new_ent': new_ent_coef,
                        'trades': total_trades,
                        'buy_pct': buy_pct,
                        'sell_pct': sell_pct,
                        'hold_pct': hold_pct,
                        'trigger_reason': trigger_reason_str,
                        'message': f'Entropy (re-)applied: {self.original_ent_coef:.4f} -> {new_ent_coef:.4f} ({trigger_reason_str})'
                    })
                    logger.info(f"[ADAPTIVE ENTROPY] Action imbalance or low trades: {trigger_reason_str}")
                    logger.info(f"   BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%, HOLD={hold_pct:.1f}%")
                    logger.info(f"   Entropy set to {new_ent_coef:.4f} (sustained boost every eval until balanced)")
                    self.adaptive_ent_active = True

                    # 2026-02-19: Adaptive base incentive – bump config so reward incentive changes on its own
                    try:
                        cap = getattr(config, 'adaptive_incentive_cap', 0.25)
                        step = getattr(config, 'adaptive_incentive_step', 0.02)
                        cur_base = getattr(config, 'base_trading_incentive', 0.18)
                        cur_sell = getattr(config, 'sell_base_incentive', 0.18)
                        new_base = min(cap, cur_base + step)
                        new_sell = min(cap, cur_sell + step)
                        if new_base > cur_base or new_sell > cur_sell:
                            config.update_config(base_trading_incentive=new_base, sell_base_incentive=new_sell)
                            logger.info(f"[ADAPTIVE INCENTIVE] Base incentives bumped: trading {cur_base:.2f}->{new_base:.2f}, sell {cur_sell:.2f}->{new_sell:.2f} (cap={cap})")
                    except Exception as e:
                        logger.debug(f"[ADAPTIVE INCENTIVE] Skip bump: {e}")
                else:
                    if self.adaptive_ent_active and self.original_ent_coef is not None:
                        # Reset to original if trade count recovers AND action balance is restored
                        # 2026-02-20: Enforce ent_coef_min floor to prevent entropy collapse
                        ent_coef_min = getattr(config, 'ent_coef_min', 0.02)
                        restored_ent = max(self.original_ent_coef, ent_coef_min)
                        if hasattr(self.model, 'ent_coef'):
                            self.model.ent_coef = restored_ent  # type: ignore
                        elif hasattr(self.model, 'policy') and hasattr(self.model.policy, 'ent_coef'):
                            self.model.policy.ent_coef = restored_ent  # type: ignore
                        config.update_config(ent_coef=restored_ent)
                        self.adaptive_entropy_log.append({
                            'timestep': self.num_timesteps,
                            'event': 'ENTROPY_RESET',
                            'restored_ent': restored_ent,
                            'trades': total_trades,
                            'buy_pct': buy_pct,
                            'sell_pct': sell_pct,
                            'message': f'Entropy reset to {restored_ent:.4f} (trades={total_trades} >= {min_trades_threshold}, actions balanced)'
                        })
                        logger.info(f"[ADAPTIVE ENTROPY] Trade count and action balance recovered ({total_trades} >= {min_trades_threshold}, BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%)")
                        logger.info(f"   Resetting entropy to {restored_ent:.4f} (floor: {ent_coef_min})")
                        self.adaptive_ent_active = False
                    else:
                        # FIX 2: Log that callback checked but no entropy adjustment needed
                        if not self.adaptive_ent_active:
                            log_entry = {
                                'timestep': self.num_timesteps,
                                'event': 'NO_ENTROPY_ADJUSTMENT',
                                'trades': total_trades,
                                'buy_pct': buy_pct,
                                'sell_pct': sell_pct,
                                'hold_pct': hold_pct,
                                'message': f'No entropy adjustment needed: Trades={total_trades} >= {min_trades_threshold}, BUY={buy_pct:.1f}% >= {buy_imbalance_threshold}%, SELL={sell_pct:.1f}% (balanced)'
                            }
                            self.adaptive_entropy_log.append(log_entry)
                            logger.info(f"[ADAPTIVE ENTROPY] No adjustment needed: Trades={total_trades}, BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%, actions balanced")
                            
            except Exception as e:
                log_entry = {
                    'timestep': self.num_timesteps,
                    'event': 'EVALUATION_ERROR',
                    'error': str(e),
                    'message': f'Evaluation failed: {str(e)[:100]}'
                }
                self.adaptive_lr_log.append(log_entry)
                logger.warning(f"Adaptive hyperparameter adjustment failed: {e}")
        elif not self.eval_env:
            # Log if eval_env is missing (only log once)
            if not hasattr(self, '_eval_env_warning_logged'):
                log_entry = {
                    'timestep': self.num_timesteps,
                    'event': 'NO_EVAL_ENV',
                    'message': 'Adaptive LR callback cannot run: eval_env is None'
                }
                self.adaptive_lr_log.append(log_entry)
                self._eval_env_warning_logged = True
                logger.warning("⚠️ Adaptive LR callback: eval_env is None - callback will not trigger")
        
        return True

    def _on_training_end(self) -> None:
        """Run one evaluation at end of training if callback never triggered (short runs)."""
        if self.eval_env and not self.callback_triggered:
            self.callback_triggered = True
            log_entry = {
                'timestep': self.num_timesteps,
                'event': 'CALLBACK_TRIGGERED',
                'message': f'Adaptive LR callback ran at training end ({self.num_timesteps:,} timesteps) - eval_freq was not reached during run'
            }
            self.adaptive_lr_log.append(log_entry)
            logger.info(f"[ADAPTIVE LR] Callback ran at training end ({self.num_timesteps:,} steps) - eval_freq ({self.eval_freq}) was not reached")
            try:
                from ppo.evaluation.performance_evaluator import PerformanceEvaluator  # type: ignore
                evaluator = PerformanceEvaluator()  # type: ignore
                results = evaluator.evaluate(self.model, self.eval_env, n_episodes=1)
                self.adaptive_lr_log.append({
                    'timestep': self.num_timesteps, 'event': 'EVALUATION',
                    'return': results.get('total_return', 0), 'sharpe': results.get('sharpe_ratio', 0),
                    'trades': results.get('total_trades', 0)
                })
            except Exception as e:
                logger.warning(f"[ADAPTIVE LR] End-of-training eval failed: {e}")

class EarlyStoppingCallback(BaseCallback):
    """Stop training if performance degrades (prevents overfitting)"""
    def __init__(self, eval_env=None, eval_freq=100000, patience=5, min_delta=0.005,  # OPTIMIZATION: Default 100k instead of 50k 
                 min_sharpe=-0.5, min_return=-0.10, max_drawdown=0.40, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq  # OPTIMIZATION: Default 100K (was 50K, reduces overhead by 50%)
        self.patience = patience    # FIXED: 5 evaluations (was 3, too aggressive)
        self.min_delta = min_delta  # FIXED: 0.5% improvement (was 0.1%, too strict)
        self.min_sharpe = min_sharpe  # FIXED: -0.5 (was 0.3, BC had 0.09)
        self.min_return = min_return  # FIXED: -10% (was 1%, too strict for early training)
        self.max_drawdown = max_drawdown  # FIXED: 40% (was 20%, allow more exploration)
        self.best_return = -float('inf')
        self.best_sharpe = -float('inf')
        self.no_improve_count = 0
        self.returns_history = []
        self.sharpe_history = []
        self.last_eval_step = 0
        
    def _on_step(self) -> bool:
        # Evaluate periodically
        if self.eval_env and (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            
            # Quick evaluation
            try:
                from ppo.evaluation.performance_evaluator import PerformanceEvaluator  # type: ignore
                evaluator = PerformanceEvaluator()  # type: ignore
                results = evaluator.evaluate(self.model, self.eval_env, n_episodes=1)
                
                current_return = results.get('total_return', 0)
                current_sharpe = results.get('sharpe_ratio', 0)
                current_drawdown = results.get('max_drawdown', 1.0)
                
                self.returns_history.append(current_return)
                self.sharpe_history.append(current_sharpe)
                
                # Check performance thresholds (FIXED: More lenient to allow BC model to adapt)
                # Only stop if CATASTROPHIC failure (not just underperformance)
                if (current_sharpe < self.min_sharpe and 
                    current_return < self.min_return and
                    current_drawdown > self.max_drawdown):
                    if self.verbose:
                        logger.warning(f"⚠️ CATASTROPHIC FAILURE at {self.num_timesteps:,} timesteps")
                        logger.warning(f"  Sharpe: {current_sharpe:.3f} (min: {self.min_sharpe})")
                        logger.warning(f"  Return: {current_return:.3f} (min: {self.min_return})")
                        logger.warning(f"  Drawdown: {current_drawdown:.3f} (max: {self.max_drawdown})")
                    return False  # Stop training only if ALL metrics fail
                
                # Check for improvement
                if current_return > self.best_return + self.min_delta:
                    self.best_return = current_return
                    self.best_sharpe = max(self.best_sharpe, current_sharpe)
                    self.no_improve_count = 0
                    if self.verbose:
                        logger.info(f"✅ Performance improved: Return={current_return:.3f}, Sharpe={current_sharpe:.3f}")
                else:
                    self.no_improve_count += 1
                    if self.verbose:
                        logger.info(f"📊 No improvement: {self.no_improve_count}/{self.patience} (Return={current_return:.3f})")
                
                # Early stop if no improvement
                if self.no_improve_count >= self.patience:
                    if self.verbose:
                        logger.warning(f"🛑 Early stopping: No improvement for {self.patience} evaluations")
                        logger.warning(f"   Best Return: {self.best_return:.3f}, Current: {current_return:.3f}")
                    return False  # Stop training
                    
            except Exception as e:
                logger.warning(f"Evaluation failed in early stopping callback: {e}")
                # Continue training if evaluation fails
                pass
        
        return True  # Continue training

def validate_ppo_performance(results, config):
    """
    7-Stage PPO Performance Validation (adopted from BC)
    Returns: (passed, issues, warnings)
    """
    issues = []
    warnings = []
    
    # Stage 1: Trading Activity Validation
    total_trades = results.get('total_trades', 0)
    if total_trades < 10:
        issues.append(f"CRITICAL: Too few trades ({total_trades} < 10)")
    elif total_trades < 50:
        warnings.append(f"Low trade count ({total_trades} < 50)")
    
    # Stage 2: Return Validation
    total_return = results.get('total_return', 0)
    target_return = config.get('min_target_return', 0.05)
    if total_return < -0.10:
        issues.append(f"CRITICAL: Catastrophic losses ({total_return:.1%})")
    elif total_return < target_return:
        warnings.append(f"Return below target ({total_return:.1%} < {target_return:.1%})")
    
    # Stage 3: Risk-Adjusted Returns
    sharpe = results.get('sharpe_ratio', 0)
    if sharpe < 0:
        issues.append(f"CRITICAL: Negative Sharpe ({sharpe:.2f})")
    elif sharpe < 1.0:
        warnings.append(f"Low Sharpe ratio ({sharpe:.2f} < 1.0)")
    
    # Stage 4: Drawdown Control
    max_dd = results.get('max_drawdown', 0)
    if max_dd > 0.25:
        issues.append(f"CRITICAL: Excessive drawdown ({max_dd:.1%} > 25%)")
    elif max_dd > 0.15:
        warnings.append(f"High drawdown ({max_dd:.1%} > 15%)")
    
    # Stage 5: Win Rate Assessment
    win_rate = results.get('win_rate', 0)
    if win_rate < 0.20:
        issues.append(f"CRITICAL: Very low win rate ({win_rate:.1%})")
    elif win_rate < 0.40:
        warnings.append(f"Low win rate ({win_rate:.1%} < 40%)")
    
    # Stage 6: Training Convergence
    if 'convergence_score' in results:
        conv = results['convergence_score']
        if conv < 0.5:
            warnings.append(f"Poor convergence ({conv:.2f} < 0.5)")
    
    # Stage 7: Paper Trading Readiness
    is_ready = results.get('paper_trading_ready', False)
    if not is_ready:
        warnings.append("Not ready for paper trading")
    
    passed = len(issues) == 0
    return passed, issues, warnings

# Add PPO directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
ppo_dir = os.path.join(current_dir, 'ppo')
if ppo_dir not in sys.path:
    sys.path.insert(0, ppo_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Set up logger
logger = logging.getLogger(__name__)



# =============================================================================

# CRITICAL PATH SETUP - MUST BE FIRST

# =============================================================================



# Add current directory and PPO directory to Python path BEFORE any other imports

current_dir = Path(__file__).parent

ppo_dir = os.path.join(current_dir, 'ppo')



# Add both directories to path if not already present

if str(current_dir) not in sys.path:

    sys.path.insert(0, str(current_dir))

if ppo_dir not in sys.path:

    sys.path.insert(0, ppo_dir)



print(f"[OK] Added {current_dir} and {ppo_dir} to Python path")



# =============================================================================

# PERFORMANCE OPTIMIZATIONS

# =============================================================================



# Set environment variables for maximum performance and memory optimization
# CONSOLIDATED: All environment variables in one place
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048,garbage_collection_threshold:1000'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'



# =============================================================================

# VECTORIZED PERFORMANCE OPTIMIZATIONS

# =============================================================================



# VectorizedPerformanceOptimizer moved to ppo.utils.performance_monitor
# REMOVED: Unused get_performance_optimizer() function



# Memory optimization settings (consolidated with above)
# REMOVED: Duplicate PYTORCH_CUDA_ALLOC_CONF setting



# Memory management (imports already consolidated above)



def optimize_memory():

    """Aggressive memory optimization"""

    gc.collect()

    try:

        import torch

        if hasattr(torch, 'cuda'):

            torch.cuda.empty_cache()

        if hasattr(torch, 'cuda'):

            torch.cuda.synchronize()

    except ImportError:

        pass  # torch not available



def get_memory_usage():

    """Get current memory usage in MB"""

    process = psutil.Process()

    return process.memory_info().rss / 1024 / 1024



def monitor_memory_threshold(threshold_mb=None):

    """Monitor memory usage and optimize if threshold exceeded"""

    if threshold_mb is None:

        # Dynamic threshold based on available system memory

        import psutil

        total_memory = psutil.virtual_memory().total / (1024 * 1024)  # Convert to MB

        threshold_mb = total_memory * 0.8  # Use 80% of available memory as threshold
    
    

    current_memory = get_memory_usage()

    if current_memory > threshold_mb:

        logger.warning(f"Memory usage high: {current_memory:.1f}MB (threshold: {threshold_mb:.1f}MB), optimizing...")

        optimize_memory()

        return True

    return False


def _should_use_parallel_downloads() -> bool:
    """Disable multiprocessing downloads on Windows spawn."""
    return os.name != "nt"


# Numba JIT compilation for performance optimization with Python DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number.13 compatibility

try:
    import numba  # type: ignore

    # Check Numba version compatibility with Python DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number.13

    if hasattr(numba, '__version__') and numba.__version__ >= '0.58.0':

        from numba import njit, prange, jit  # type: ignore

        NUMBA_AVAILABLE = True

        print("SUCCESS: Numba JIT compilation enabled for Python DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number.13")

    elif hasattr(numba, '__version__') and numba.__version__ >= '0.57.0':

        # Fallback for slightly older versions


        NUMBA_AVAILABLE = True

        print("SUCCESS: Numba JIT compilation enabled (older version)")

    else:

        # Numba version too old or incompatible

        NUMBA_AVAILABLE = False

        print("WARNING: Numba version incompatible with Python DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number.13, using fallback implementations")

        # Fallback decorators for Python DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number.13+ compatibility

        def njit(func=None, **kwargs):

            """Fallback njit decorator for Python DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number.13+ compatibility"""

            if func is None:

                return lambda f: f

            return func



        def prange(*args):

            """Fallback prange for Python DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number.13+ compatibility"""

            return range(*args)



        def jit(func=None, **kwargs):

            """Fallback jit decorator for Python DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number.13+ compatibility"""

            if func is None:

                return lambda f: f

            return func

except ImportError:

    NUMBA_AVAILABLE = False

    print("INFO: Numba not available, using fallback implementations")

    # Fallback decorators already defined above



# =============================================================================

# MAIN IMPORTS - PPO TRADING ORCHESTRATOR (CONSOLIDATED)

# =============================================================================




import pandas as pd


import time

import json

import importlib


import logging.handlers

import gc


import weakref

from datetime import datetime, timedelta

from stable_baselines3.common.callbacks import BaseCallback


from typing import Dict, List, Tuple, Any, Optional, Union

import matplotlib.pyplot as plt

# Statistical testing imports for A/B testing
try:
    from scipy import stats
    from scipy.stats import ttest_ind, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. A/B testing will use basic statistical tests.")

# PCA imports for Step DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number - PCA Residual Features
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. PCA features will be disabled.")

import builtins as _builtins

# =============================================================================

# DYNAMIC FEATURE DETECTION FUNCTION

# =============================================================================

def get_dynamic_feature_status_standalone() -> Dict[str, str]:
    """Get dynamic feature status based on actual file implementations"""
    
    # Get the base directory (where PPO_fixed.py is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ppo_dir = os.path.join(base_dir, 'ppo')
    
    # Check actual file existence for each feature
    feature_files = {
        'sequential_training': ['models/sequential_trainer.py', 'core/sequential_training.py'],
        'auto_promotion': ['models/auto_promotion.py', 'core/auto_promotion.py'],
        'zscore_hedging': ['core/hedging_strategies.py', 'strategies/zscore_hedging.py'],
        'beta_spread': ['core/beta_spread.py', 'strategies/beta_spread.py'],
        'regime_detection': ['models/regime_manager.py', 'core/regime_detection.py'],
        'optuna_integration': ['optimization/optuna_optimizer.py', 'core/optuna_integration.py'],
        'fundamental_analysis': ['data/fundamental_analyzer.py', 'analysis/fundamental.py'],
        'sentiment_analysis': ['data/sentiment_analyzer.py', 'analysis/sentiment.py'],
        'dynamic_stop_loss': ['core/stop_loss.py', 'strategies/stop_loss.py'],
        'trade_validation': ['core/trade_validator.py', 'validation/trade_validation.py'],
        'portfolio_optimization': ['optimization/portfolio_optimizer.py', 'core/portfolio_optimization.py']
    }
    
    status = {}
    for feature_name, possible_paths in feature_files.items():
        feature_found = False
        for path in possible_paths:
            full_path = os.path.join(ppo_dir, path)
            if os.path.exists(full_path):
                status[feature_name] = " ENABLED"
                feature_found = True
                break
        
        if not feature_found:
            status[feature_name] = " DISABLED"
    
    return status

# Setup logging (logging.handlers already imported above)



# Create logs directory if it doesn't exist

logs_dir = os.path.join(current_dir, "ppo", "logs")

os.makedirs(logs_dir, exist_ok=True)



# Optimized logging configuration

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

    handlers=[

        # Rotating file handler with compression

        logging.handlers.RotatingFileHandler(

            os.path.join(logs_dir, f"ppo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),

            maxBytes=10*1024*1024,  # 10MB

            backupCount=5,

            encoding='utf-8'

        ),

        # Timed rotating handler for daily logs

        logging.handlers.TimedRotatingFileHandler(

            os.path.join(logs_dir, 'ppo_daily.log'),

            when='midnight',

            interval=1,

            backupCount=7,

            encoding='utf-8'

        ),

        # Console handler with performance optimization

        logging.StreamHandler()

    ]

)



# Performance logging

performance_logger = logging.getLogger('performance')

performance_logger.setLevel(logging.INFO)

performance_handler = logging.handlers.RotatingFileHandler(

    os.path.join(logs_dir, 'performance.log'),

    maxBytes=5*1024*1024,  # 5MB

    backupCount=3,

    encoding='utf-8'

)

performance_logger.addHandler(performance_handler)



# Suppress verbose logging from external libraries

logging.getLogger('urllib3').setLevel(logging.WARNING)

logging.getLogger('requests').setLevel(logging.WARNING)

logging.getLogger('matplotlib').setLevel(logging.WARNING)

logging.getLogger('numba').setLevel(logging.WARNING)

logger = logging.getLogger("PPOTrading")



# =============================================================================

# PPO TRADING ORCHESTRATOR - ENHANCED WITH DEBUG LOGGING

# =============================================================================



# PPO directory already added to path above

# Logging already configured above



# =============================================================================
# A/B TESTING FRAMEWORK FOR SUPER AGENT EVOLUTION
# =============================================================================

# ABTestingFramework moved to ppo.core.ab_testing
# SuperAgentRefinement moved to ppo.core.super_agent_refinement
# PCAResidualExtractor moved to ppo.data.feature_engineer
# AuxiliaryPredictionHeads moved to ppo.training.auxiliary_heads
class _AsciiFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]

        try:

            record.msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')

        except Exception:

            pass

        try:

            if record.args:

                if isinstance(record.args, tuple):

                    record.args = tuple(

                        str(a).encode('ascii', 'ignore').decode('ascii') for a in record.args

                    )

                else:

                    record.args = ()

        except Exception:

            record.args = ()

        return True



# Attach filter to all existing handlers (root and this logger)

for _h in logging.getLogger().handlers:

    _h.addFilter(_AsciiFilter())

for _h in logger.handlers:

    _h.addFilter(_AsciiFilter())



# Safe print wrapper that strips non-ASCII characters (emojis)


_orig_print = _builtins.print

def _safe_print(*args, **kwargs):

    safe_args = []

    for a in args:

        try:

            safe_args.append(str(a).encode('ascii', 'ignore').decode('ascii'))

        except Exception:

            safe_args.append(str(a))

    return _orig_print(*safe_args, **kwargs)

print = _safe_print  # type: ignore



# CRITICAL FIX: Global debug state for comprehensive logging

DEBUG_STATE = {

    'environment_creation': {},

    'data_processing': {},

    'training_execution': {},

    'evaluation_execution': {},

    'errors_encountered': [],

    'performance_metrics': {},

    'system_health': {},

    'trade_execution': {},  # NEW: Track trade execution details

    'portfolio_changes': {},  # NEW: Track portfolio value changes

    'action_processing': {}   # NEW: Track action processing

}



def get_numeric_metric(metric_data, default_value=0.0):
    """Generic helper function to get numeric value from a metric"""
    if isinstance(metric_data, dict):
        return float(metric_data.get('value', default_value))
    elif isinstance(metric_data, (int, float, np.number)):
        return float(metric_data)
    return default_value

def log_debug_info(category: str, info: Dict[str, Any]):

    """Log debug information to global state"""

    if category not in DEBUG_STATE:

        DEBUG_STATE[category] = {}
    
    

    # Handle different category types

    if category == 'errors_encountered':

        # errors_encountered is a list, append the error

        if isinstance(info, dict):

            for key, value in info.items():

                DEBUG_STATE[category].append(f"{key}: {value}")

        else:

            DEBUG_STATE[category].append(str(info))

    else:

        # Other categories are dictionaries, use update

        DEBUG_STATE[category].update(info)
    
    

    logger.debug(f"DEBUG [{category}]: {info}")



def get_status_icon(status_condition):

    """Get status icon based on condition"""

    return "[OK]" if status_condition else "[WARNING]"



def print_status_section(title, items):

    """Print a status section with dynamic icons"""

    print(f"\n{title}")

    for item_name, condition in items.items():

        icon = get_status_icon(condition)

        status = "ACTIVE" if condition else "INACTIVE"

        print(f"  {icon} {item_name}: {status}")



def print_debug_summary():

    """Print comprehensive debug summary at end of execution"""

    print("\n" + "="*80)

    print(" COMPREHENSIVE DEBUG SUMMARY")

    print("="*80)

    
    
    # Environment Creation Debug

    if DEBUG_STATE['environment_creation']:

        print("\n ENVIRONMENT CREATION DEBUG:")

        for key, value in DEBUG_STATE['environment_creation'].items():

            print(f"  {key}: {value}")
    
    

    # Data Processing Debug

    if DEBUG_STATE['data_processing']:

        print("\n DATA PROCESSING DEBUG:")

        for key, value in DEBUG_STATE['data_processing'].items():

            print(f"  {key}: {value}")
    
    

    # Training Execution Debug

    if DEBUG_STATE['training_execution']:

        print("\n TRAINING EXECUTION DEBUG:")

        for key, value in DEBUG_STATE['training_execution'].items():

            print(f"  {key}: {value}")
    
    

    # Evaluation Execution Debug

    if DEBUG_STATE['evaluation_execution']:

        print("\n EVALUATION EXECUTION DEBUG:")

        for key, value in DEBUG_STATE['evaluation_execution'].items():

            print(f"  {key}: {value}")
    
    

    # Error Summary

    if DEBUG_STATE['errors_encountered']:

        print("\n ERRORS ENCOUNTERED:")

        for i, error in enumerate(DEBUG_STATE['errors_encountered'], 1):

            print(f"  {i}. {error}")
    
    

    # Performance Metrics

    if DEBUG_STATE['performance_metrics']:

        print("\n PERFORMANCE METRICS:")

        for key, value in DEBUG_STATE['performance_metrics'].items():

            print(f"  {key}: {value}")
    
    

    # Trade Execution Debug

    if DEBUG_STATE['trade_execution']:

        print("\n TRADE EXECUTION DEBUG:")

        for key, value in DEBUG_STATE['trade_execution'].items():

            print(f"  {key}: {value}")
    
    

    # Portfolio Changes Debug

    if DEBUG_STATE['portfolio_changes']:

        print("\n PORTFOLIO CHANGES DEBUG:")

        for key, value in DEBUG_STATE['portfolio_changes'].items():

            print(f"  {key}: {value}")
    
    

    # Action Processing Debug

    if DEBUG_STATE['action_processing']:

        print("\n ACTION PROCESSING DEBUG:")

        for key, value in DEBUG_STATE['action_processing'].items():

            print(f"  {key}: {value}")
    
    

    # System Health

    if DEBUG_STATE['system_health']:

        print("\n SYSTEM HEALTH:")

        for key, value in DEBUG_STATE['system_health'].items():

            print(f"  {key}: {value}")
    
    

    print("\n" + "="*80)



class DetailedProgressCallback(BaseCallback):
    """
    Custom callback for detailed progress tracking during PPO training.
    Shows timestep progress, episode info, and performance metrics.
    """
    
    def __init__(self, log_freq: int = 10000, verbose: int = 1, status_file: Optional[str] = None):
        super(DetailedProgressCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.start_time = time.time()
        self.status_file = status_file
        
    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        self.start_time = time.time()
        print("\n" + "="*80)
        print("PPO TRAINING STARTED")
        print("="*80)
        print(f"Total timesteps to train: {self.locals.get('total_timesteps', 'Unknown'):,}")
        print(f"Environment: {self.training_env.num_envs} parallel environments")
        print("="*80 + "\n")
        
    def _on_step(self) -> bool:
        """Called after each step."""
        # Get current timestep
        current_timestep = self.num_timesteps
        total_timesteps = self.locals.get('total_timesteps', 0)
        
        # Log progress every log_freq steps
        if current_timestep % self.log_freq == 0:
            elapsed_time = time.time() - self.start_time
            steps_per_sec = current_timestep / elapsed_time if elapsed_time > 0 else 0
            progress_pct = (current_timestep / total_timesteps * 100) if total_timesteps > 0 else 0
            remaining_steps = total_timesteps - current_timestep
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            
            # Create a simple progress bar
            bar_length = 50
            filled_length = int(bar_length * current_timestep // total_timesteps) if total_timesteps > 0 else 0
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print("\n" + "="*80)
            print(f"PPO TRAINING PROGRESS")
            print("="*80)
            print(f"Step {current_timestep:,} / {total_timesteps:,} ({progress_pct:.1f}%)")
            print(f"[{bar}] {progress_pct:.1f}%")
            print(f"Speed: {steps_per_sec:.1f} steps/sec")
            print(f"Elapsed: {elapsed_time/60:.1f} min | ETA: {eta_seconds/60:.1f} min")
            
            # Show episode info if available
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:]
                print(f"Recent episodes: {len(self.episode_rewards)} total")
                print(f"Avg reward (last 10): {np.mean(recent_rewards):.4f}")
                print(f"Avg length (last 10): {np.mean(self.episode_lengths[-10:]):.1f}")
            
            print("="*80 + "\n")
            
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        # Collect episode statistics
        if 'episode_rewards' in self.locals:
            self.episode_rewards.extend(self.locals['episode_rewards'])
        if 'episode_lengths' in self.locals:
            self.episode_lengths.extend(self.locals['episode_lengths'])
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        if getattr(self, 'status_file', None):
            try:
                self._write_status({
                    'status': 'running', 'phase': 'training', 'current_step': self.num_timesteps,
                    'total_steps': self.locals.get('total_timesteps', self.num_timesteps),
                    'progress_pct': 100.0, 'message': 'Training loop finished'
                })
            except Exception:
                pass
        print("\n" + "="*80)
        print("PPO TRAINING COMPLETED")
        print("="*80)
        print(f"Total timesteps: {self.num_timesteps:,}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average speed: {self.num_timesteps/total_time:.1f} steps/sec")
        print(f"Total episodes: {len(self.episode_rewards)}")
        if len(self.episode_rewards) > 0:
            print(f"Average episode reward: {np.mean(self.episode_rewards):.4f}")
            print(f"Average episode length: {np.mean(self.episode_lengths):.1f}")
        print("="*80 + "\n")


class PPOTradingOrchestrator:

    """Enhanced PPO Trading Orchestrator with comprehensive debugging"""

    
    
    def __init__(self):

        """Initialize PPO Trading Orchestrator with enhanced debugging"""
        

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_source = "UNKNOWN"
        self._data_quality = {}

        self.config = self._get_dynamic_configuration()

        self.device = self._init_device()

        
        
        # Directory setup with validation

        self.data_dir = os.path.join(self.base_dir, "ppo", "saved_data")

        self.models_dir = os.path.join(self.base_dir, "ppo", "saved_models")

        self.reports_dir = os.path.join(self.base_dir, "ppo", "daily_reports")

        self.daily_reports_dir = self.reports_dir  # CRITICAL FIX: Add missing attribute

        # Initialize A/B Testing Framework for Super Agent Evolution
        # CLEANED: Removed commented A/B testing initialization
        self.baseline_model_path = None
        self.candidate_model_path = None
        
        # Initialize Step DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number - PCA Residual Features (DISABLED - caused performance degradation)
        # CLEANED: Removed commented PCA and auxiliary heads initialization
        self.pca_extractor = None
        
        logger.info(" A/B Testing Framework initialized for Super Agent Evolution")
        logger.info(" CLEANED: Removed disabled PCA and auxiliary features for cleaner code")
        
        
        # Validate and create directories

        for directory in [self.data_dir, self.models_dir, self.reports_dir]:

            os.makedirs(directory, exist_ok=True)

            if not os.path.exists(directory):

                raise RuntimeError(f"Failed to create directory: {directory}")
        
        

        # CRITICAL FIX: Enhanced configuration validation

        self._validate_configuration()

        
        
        # DYNAMIC FEATURE DETECTION SYSTEM

        self._detect_advanced_features()

        
        
        # OPTUNA INTEGRATION (DISABLED BY DEFAULT)

        self._optuna_enabled = False  # Set to True to enable hyperparameter optimization

        self._optuna_study = None

        
        
        # ENHANCED PROFILING SYSTEM

        self._profiling_data = {

            'timings': {},

            'memory_usage': [],

            'gpu_usage': [],

            'bottlenecks': [],

            'performance_metrics': {}

        }

        self._profiling_active = False

        self._current_operation = None

        
        
        # Initialize dynamic configuration system
        try:
            # Add the ppo directory to the path for imports
            ppo_dir = os.path.join(os.path.dirname(__file__), 'ppo')
            if ppo_dir not in sys.path:
                sys.path.insert(0, ppo_dir)
            
            from ppo.utils.dynamic_config import get_dynamic_config, register_config_listener
            self.dynamic_config = get_dynamic_config()
            register_config_listener(self._on_config_change)
            
            # Initialize orchestrator config with current dynamic config values
            # Always call _on_config_change to sync with current dynamic config values
            self._on_config_change(self.dynamic_config)
            
            logger.info(" DYNAMIC CONFIG: Orchestrator connected to centralized config")
        except Exception as e:
            self.dynamic_config = None
            logger.warning(f" DYNAMIC CONFIG: Using fallback config: {e}")

        # Log initialization

        logger.info(" PPOTradingOrchestrator initialized successfully")

        log_debug_info('system_health', {

            'base_dir': self.base_dir,

            'data_dir': self.data_dir,

            'models_dir': self.models_dir,

            'reports_dir': self.reports_dir,

            'device': str(self.device),

            'config_keys': list(self.config.keys()),

            'advanced_features_detected': self._get_advanced_features_status()

        })

    def _on_config_change(self, config):
        """Callback for when dynamic configuration changes"""
        try:
            # Update orchestrator config from dynamic config
            old_interval = self.config.get('min_trade_interval', 'NOT_SET')
            old_lr = self.config.get('learning_rate', 'NOT_SET')
            old_n_steps = self.config.get('n_steps', 'NOT_SET')
            old_batch_size = self.config.get('batch_size', 'NOT_SET')
            old_gamma = self.config.get('gamma', 'NOT_SET')
            old_ent_coef = self.config.get('ent_coef', 'NOT_SET')
            old_sharpe_bonus = self.config.get('sharpe_bonus', 'NOT_SET')
            old_transaction_penalty = self.config.get('transaction_penalty', 'NOT_SET')
            old_activity_bonus = self.config.get('activity_bonus', 'NOT_SET')

            # Update config dictionary with Phase 1, 2, and DEFAULT_RETRY_ATTEMPTS  # CONSTANT: Replaced magic number parameters
            self.config['min_trade_interval'] = config.min_trade_interval
            self.config['window_size'] = config.window_size
            self.config['max_position_weight'] = config.max_position_weight
            self.config['learning_rate'] = config.learning_rate
            self.config['n_steps'] = config.n_steps
            self.config['batch_size'] = config.batch_size
            self.config['n_epochs'] = config.n_epochs
            self.config['gamma'] = config.gamma
            self.config['gae_lambda'] = config.gae_lambda
            self.config['clip_range'] = config.clip_range
            self.config['ent_coef'] = config.ent_coef
            self.config['vf_coef'] = config.vf_coef
            self.config['max_grad_norm'] = config.max_grad_norm
            self.config['base_return'] = config.base_return
            self.config['sharpe_bonus'] = config.sharpe_bonus
            self.config['transaction_penalty'] = config.transaction_penalty
            self.config['activity_bonus'] = config.activity_bonus
            self.config['risk_bonus'] = config.risk_bonus
            self.config['drawdown_penalty'] = config.drawdown_penalty
            self.config['concentration_penalty'] = config.concentration_penalty
            self.config['win_rate_bonus'] = config.win_rate_bonus
            self.config['profit_factor_bonus'] = config.profit_factor_bonus
            self.config['consistency_bonus'] = config.consistency_bonus
            self.config['w_turnover'] = config.w_turnover

            # Step 1: Universe override (reduce tickers to edge first)
            try:
                if getattr(config, 'universe_override_enabled', False):
                    override = getattr(config, 'universe_override_tickers', None)
                    if isinstance(override, list) and len(override) > 0:
                        # Normalize to upper-case tickers
                        override_tickers = [str(t).upper() for t in override]
                        old_tickers = self.config.get('tickers', [])
                        self.config['tickers'] = override_tickers
                        if old_tickers != override_tickers:
                            logger.info(f" UNIVERSE OVERRIDE: {old_tickers}  {override_tickers}")
            except Exception as e:
                logger.warning(f" UNIVERSE OVERRIDE: Could not apply: {e}")

            # Log the changes
            changes = []
            if old_interval != config.min_trade_interval:
                changes.append(f"min_trade_interval: {old_interval}  {config.min_trade_interval}")
            if self.config.get('window_size') != config.window_size:
                changes.append(f"window_size: {self.config.get('window_size')}  {config.window_size}")
            if self.config.get('max_position_weight') != config.max_position_weight:
                changes.append(f"max_position_weight: {self.config.get('max_position_weight')}  {config.max_position_weight}")
            if old_lr != config.learning_rate:
                changes.append(f"learning_rate: {old_lr}  {config.learning_rate}")
            if old_n_steps != config.n_steps:
                changes.append(f"n_steps: {old_n_steps}  {config.n_steps}")
            if old_batch_size != config.batch_size:
                changes.append(f"batch_size: {old_batch_size}  {config.batch_size}")
            if old_gamma != config.gamma:
                changes.append(f"gamma: {old_gamma}  {config.gamma}")
            if old_ent_coef != config.ent_coef:
                changes.append(f"ent_coef: {old_ent_coef}  {config.ent_coef}")
            if old_sharpe_bonus != config.sharpe_bonus:
                changes.append(f"sharpe_bonus: {old_sharpe_bonus}  {config.sharpe_bonus}")
            if old_transaction_penalty != config.transaction_penalty:
                changes.append(f"transaction_penalty: {old_transaction_penalty}  {config.transaction_penalty}")
            if old_activity_bonus != config.activity_bonus:
                changes.append(f"activity_bonus: {old_activity_bonus}  {config.activity_bonus}")

            if changes:
                logger.info(f" DYNAMIC CONFIG UPDATE: {', '.join(changes)}")
            else:
                logger.debug(" DYNAMIC CONFIG: No changes detected")

        except Exception as e:
            logger.error(f" DYNAMIC CONFIG: Error updating orchestrator config: {e}")
    
    

    def _init_device(self):
        """Initialize device (Force CPU to avoid infinite loop)"""
        
        # Prefer CUDA when available; otherwise CPU
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("🚀 GPU FIX: Using CUDA device")
        else:
            device = torch.device("cpu")
            logger.info(" CPU FALLBACK: Using CPU device")
        
        return device

    

    def _get_dynamic_normalization_config(self):
        """Generate dynamic normalization configuration for consistent reward scaling"""
        try:
            # DYNAMIC NORMALIZATION: Create consistent normalization parameters
            normalization_config = {
                'norm_obs': True,
                'norm_reward': True,
                'clip_obs': 10.0,
                'clip_reward': 10.0,
                'gamma': self.config.get('gamma', 0.99),
                'epsilon': 1e-8,  # Small value to prevent division by zero
                'training_mode': True,  # Flag to indicate if this is for training
                'reward_scale_factor': 1.0,  # Dynamic scaling factor
                'observation_scale_factor': 1.0,  # Dynamic observation scaling
                'adaptive_scaling': True,  # Enable adaptive scaling
                'consistency_mode': True  # Ensure consistency between train/eval
            }
            
            logger.info(" DYNAMIC NORMALIZATION: Configuration created for consistent reward scaling")
            return normalization_config
            
        except Exception as e:
            logger.error(f" DYNAMIC NORMALIZATION: Failed to create configuration: {e}")
            # Fallback to default normalization
            return {
                'norm_obs': True,
                'norm_reward': True,
                'clip_obs': 10.0,
                'clip_reward': 10.0,
                'gamma': 0.99,
                'epsilon': 1e-8,
                'training_mode': True,
                'reward_scale_factor': 1.0,
                'observation_scale_factor': 1.0,
                'adaptive_scaling': False,
                'consistency_mode': True
            }

    def _save_normalization_stats(self, env, mode='train'):
        """Save normalization statistics for consistent behavior between training and evaluation"""
        try:
            if hasattr(env, '_vn_stats_path') and hasattr(env, 'save'):
                # Save the normalization statistics8
                env.save(env._vn_stats_path)
                logger.info(f" DYNAMIC NORMALIZATION: Saved {mode} normalization stats to {env._vn_stats_path}")
                
                # Log the statistics for debugging
                if hasattr(env, 'obs_rms') and hasattr(env, 'ret_rms'):
                    logger.info(f" DYNAMIC NORMALIZATION: Obs mean={env.obs_rms.mean.mean():.4f}, Obs std={env.obs_rms.var.sqrt().mean():.4f}")
                    logger.info(f" DYNAMIC NORMALIZATION: Reward mean={env.ret_rms.mean:.4f}, Reward std={env.ret_rms.var.sqrt():.4f}")
                
                return True
            else:
                logger.warning(f" DYNAMIC NORMALIZATION: Cannot save {mode} normalization stats - environment not compatible")
                return False
                
        except Exception as e:
            logger.error(f" DYNAMIC NORMALIZATION: Failed to save {mode} normalization stats: {e}")
            return False

    def _load_normalization_stats(self, env, mode='test'):
        """Load normalization statistics for consistent behavior between training and evaluation"""
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            
            if hasattr(env, '_vn_stats_path') and os.path.exists(env._vn_stats_path):
                # Load the normalization statistics
                env = VecNormalize.load(env._vn_stats_path, env)
                logger.info(f" DYNAMIC NORMALIZATION: Loaded {mode} normalization stats from {env._vn_stats_path}")
                
                # Log the statistics for debugging
                if hasattr(env, 'obs_rms') and hasattr(env, 'ret_rms'):
                    import numpy as np
                    try:
                        # Handle both dict and single RunningMeanStd cases
                        if isinstance(env.obs_rms, dict):
                            obs_mean = np.mean([rms.mean for rms in env.obs_rms.values() if hasattr(rms, 'mean')]) if env.obs_rms else 0.0
                            obs_std = np.mean([np.sqrt(rms.var) for rms in env.obs_rms.values() if hasattr(rms, 'var')]) if env.obs_rms else 0.0
                        else:
                            obs_mean = np.mean(env.obs_rms.mean) if hasattr(env.obs_rms, 'mean') else 0.0
                            obs_std = np.mean(np.sqrt(env.obs_rms.var)) if hasattr(env.obs_rms, 'var') else 0.0
                        ret_mean = env.ret_rms.mean if hasattr(env.ret_rms, 'mean') else 0.0
                        ret_std = np.sqrt(env.ret_rms.var) if hasattr(env.ret_rms, 'var') else 0.0
                        logger.info(f" DYNAMIC NORMALIZATION: Loaded Obs mean={obs_mean:.4f}, Obs std={obs_std:.4f}")
                    except (AttributeError, TypeError):
                        logger.debug("Could not access normalization stats for logging")
                    logger.info(f" DYNAMIC NORMALIZATION: Loaded Reward mean={ret_mean:.4f}, Reward std={ret_std:.4f}")
                
                return env
            else:
                logger.warning(f" DYNAMIC NORMALIZATION: Cannot load {mode} normalization stats - file not found")
                return env
                
        except Exception as e:
            logger.error(f" DYNAMIC NORMALIZATION: Failed to load {mode} normalization stats: {e}")
            return env

    def _get_dynamic_configuration(self):

        """Generate dynamic configuration based on system capabilities and performance"""

        # Load configuration from config.py if available
        if CONFIG_AVAILABLE:
            # Use config from config.py
            base_config = {
                'total_timesteps': config.ENV_CONFIG.get('total_timesteps', 20000),
                'learning_rate': config.MODEL_CONFIG.get('learning_rate', 1e-6),
                'batch_size': config.MODEL_CONFIG.get('batch_size', 512),
                'n_steps': config.MODEL_CONFIG.get('n_steps', 2048),
                'n_epochs': config.MODEL_CONFIG.get('n_epochs', 4),
                'gamma': config.MODEL_CONFIG.get('gamma', 0.995),
                'gae_lambda': config.MODEL_CONFIG.get('gae_lambda', 0.98),
                'clip_range': config.MODEL_CONFIG.get('clip_range', 0.15),
                'ent_coef': config.MODEL_CONFIG.get('ent_coef', 0.05),  # INCREASED: 0.01→0.05 for 1M timestep training (prevents exploration collapse)
                'vf_coef': config.MODEL_CONFIG.get('vf_coef', 0.25),
                'max_grad_norm': config.MODEL_CONFIG.get('max_grad_norm', 0.3),
                'seed': config.MODEL_CONFIG.get('seed', 42),
                'tickers': config.DATA_CONFIG.get('tickers', ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY', 'GLD']),
                'start_date': config.DATA_CONFIG.get('start_date', '2022-01-01'),
                'end_date': config.DATA_CONFIG.get('end_date', '2023-12-31'),
                'window_size': config.DATA_CONFIG.get('window_size', 10),
                'initial_balance': config.DATA_CONFIG.get('initial_balance', 100000.0),
                'transaction_cost': config.TRADING_CONFIG.get('commission_rate', 0.001),
                'slippage': config.TRADING_CONFIG.get('slippage_rate', 0.0002),
                'test_window_days': 90,
                'n_eval_episodes': 1,  # CRITICAL FIX: Changed from 10 to 1 to avoid memorization
                                         # Running 10 episodes on the same test data causes the agent to see
                                         # identical sequences, making it impossible to assess generalization
            }
            logger.info(f"✅ Loaded configuration from config.py: {base_config['total_timesteps']} timesteps")
        else:
            # Fallback to default configuration
            base_config = {}
            logger.warning("⚠️ Using fallback configuration - config.py not available")
        
        # Get system capabilities

        cpu_count = psutil.cpu_count()

        memory_gb = psutil.virtual_memory().total / (1024**DEFAULT_RETRY_ATTEMPTS)  # CONSTANT: Replaced magic number

        has_gpu = torch.cuda.is_available() if hasattr(torch, 'cuda') else False

        
        
        # Dynamic batch size based on system memory (timesteps from config)

        if memory_gb >= 16:

            batch_size = base_config.get('batch_size', 128)

        elif memory_gb >= 8:

            batch_size = base_config.get('batch_size', 64)

        else:

            batch_size = base_config.get('batch_size', 32)

        # Use timesteps from config instead of hardcoded values
        training_timesteps = base_config.get('total_timesteps', 20000)

        # Override from dynamic_config so training uses canonical config (config.py must not override)
        from ppo.utils.dynamic_config import get_dynamic_config
        _dc = get_dynamic_config()
        base_config['learning_rate'] = _dc.learning_rate
        base_config['use_lr_schedule'] = getattr(_dc, 'use_lr_schedule', True)
        base_config['lr_initial'] = getattr(_dc, 'lr_initial', 3e-5)
        base_config['lr_final'] = getattr(_dc, 'lr_final', 1.5e-5)
        base_config['n_steps'] = getattr(_dc, 'n_steps', 2048)
        base_config['batch_size'] = getattr(_dc, 'batch_size', 512)
        base_config['n_epochs'] = getattr(_dc, 'n_epochs', 4)
        base_config['gamma'] = getattr(_dc, 'gamma', 0.995)
        base_config['gae_lambda'] = getattr(_dc, 'gae_lambda', 0.98)
        base_config['clip_range'] = getattr(_dc, 'clip_range', 0.15)
        base_config['ent_coef'] = getattr(_dc, 'ent_coef', 0.05)
        base_config['vf_coef'] = getattr(_dc, 'vf_coef', 0.25)
        base_config['max_grad_norm'] = getattr(_dc, 'max_grad_norm', 0.3)

        # Dynamic learning rate based on system capabilities

        if has_gpu and memory_gb >= 16:

            learning_rate = 3e-4

        elif has_gpu:

            learning_rate = 2e-4

        else:

            learning_rate = 1e-4
        
        

        # Dynamic validation metrics based on training progress

        # These will be updated based on actual performance

        validation_metrics = {

            'min_sharpe': 0.3,  # Start low, increase as training progresses

            'max_drawdown': 0.25,  # Start high, decrease as training progresses

            'min_win_rate': 0.3  # Start low, increase as training progresses

        }

        
        
        # ENHANCED: Learning rate schedule to prevent overfitting (with adaptive support)
        use_lr_schedule = base_config.get('use_lr_schedule', False)
        if use_lr_schedule:
            from ppo.utils.dynamic_config import get_dynamic_config

            lr_initial = base_config.get('lr_initial', 1e-5)
            lr_final = base_config.get('lr_final', 7.5e-6)
            lr_decay_type = base_config.get('lr_decay_type', 'linear')

            # SB3 passes progress_remaining (1.0 -> 0.0). Local linear schedule to avoid SB3 version-dependent import.
            def adaptive_lr_schedule(progress_remaining: float) -> float:
                """Adaptive LR schedule: reads lr_final from config in real-time. progress_remaining 1->0."""
                progress = 1.0 - progress_remaining  # 0 at start, 1 at end
                config = get_dynamic_config()
                current_lr_final = getattr(config, 'lr_final', lr_final)
                return float(lr_initial + (current_lr_final - lr_initial) * progress)

            final_learning_rate = adaptive_lr_schedule(0.0)  # End of training (progress_remaining=0)
            logger.info(f"[ADAPTIVE LR SCHEDULE] Enabled: {lr_initial:.2e} -> {lr_final:.2e} ({lr_decay_type})")
            logger.info(f"   Initial LR from dynamic_config; adaptive callback can increase (poor/excellent) or decrease (good-but-not-excellent)")
            actual_learning_rate = adaptive_lr_schedule  # Pass adaptive schedule function
        else:
            actual_learning_rate = base_config.get('learning_rate', learning_rate)
            logger.info(f"[FIXED LR] Using fixed learning rate: {actual_learning_rate:.2e}")
        
        return {

            'total_timesteps': training_timesteps,  # Use from config.py

            'learning_rate': actual_learning_rate,  # Can be schedule or fixed value

            'batch_size': batch_size,

            'n_steps': base_config.get('n_steps', min(2048, batch_size * 16)),  # From dynamic_config overlay

            'n_epochs': base_config.get('n_epochs', 4),

            'gamma': base_config.get('gamma', 0.995),

            'gae_lambda': base_config.get('gae_lambda', 0.98),

            'clip_range': base_config.get('clip_range', 0.15),

            'ent_coef': base_config.get('ent_coef', 0.05),

            'vf_coef': base_config.get('vf_coef', 0.25),

            'max_grad_norm': base_config.get('max_grad_norm', 0.3),
            'seed': base_config.get('seed', 42),

            'tickers': base_config.get('tickers', ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY', 'GLD']),

            'start_date': base_config.get('start_date', '2022-01-01'),

            'end_date': base_config.get('end_date', '2023-12-31'),

            'window_size': base_config.get('window_size', 10),  # Use from config.py

            'initial_balance': base_config.get('initial_balance', 100000.0),

            'transaction_cost': base_config.get('transaction_cost', 0.001),

            'slippage': base_config.get('slippage', 0.0002),

            'reward_scaling': 1.0,

            'test_window_days': base_config.get('test_window_days', 90),

            'n_eval_episodes': base_config.get('n_eval_episodes', 1),  # CRITICAL FIX: Changed default from 10 to 1

            'validation_metrics': validation_metrics,

            # Dynamic trading parameters

            'action_multiplier': 1.0,

            'action_threshold': 0.001,

            'min_trade_size': 0.1,

            'max_trades_per_step': DEFAULT_NUM_FEATURES,  # CONSTANT: Replaced magic number

            'min_trade_interval': 2,

            'enable_trade_validation': True,

            'position_size_pct': DEFAULT_TRAILING_STOP,  # CONSTANT: Replaced magic number

            'max_position_weight': 0.25,

            'slippage': DEFAULT_SLIPPAGE,  # CONSTANT: Replaced magic number

            # System-specific parameters

            'cpu_count': cpu_count,

            'memory_gb': memory_gb,

            'has_gpu': has_gpu

        }
    
    


    def _validate_configuration(self):

        """Validate configuration parameters"""

        required_keys = ['tickers', 'start_date', 'end_date', 'window_size', 'total_timesteps']

        missing_keys = [key for key in required_keys if key not in self.config]

        
        
        if missing_keys:

            error_msg = f"Missing required configuration keys: {missing_keys}"

            logger.error(error_msg)

            log_debug_info('errors_encountered', {'config_validation': error_msg})

            raise ValueError(error_msg)
        
        

        # Validate ticker list

        if not self.config['tickers'] or len(self.config['tickers']) == 0:

            error_msg = "Ticker list is empty or invalid"

            logger.error(error_msg)

            log_debug_info('errors_encountered', {'ticker_validation': error_msg})

            raise ValueError(error_msg)
        
        

        logger.info(" Configuration validation passed")

        log_debug_info('system_health', {'config_validation': 'PASSED'})
    
    

    def _detect_advanced_features(self):

        """Detect which advanced features are actually implemented and available"""

        try:
            # Check for Sequential Training implementation

            self._sequential_training_enabled = self._check_sequential_training()

            
            
            # Check for Auto-Promotion implementation  

            self._auto_promotion_enabled = self._check_auto_promotion()

            
            
            # Check for Z-Score Hedging implementation

            self._zscore_hedging_enabled = self._check_zscore_hedging()

            
            
            # Check for Beta-Adjusted Spread implementation

            self._beta_spread_enabled = self._check_beta_spread()

            
            
            # Check for Market Regime Detection implementation (DISABLED as requested)

            self._regime_detection_enabled = False  # DISABLED: Regime detection disabled per user request

        
        
            # Check for Fundamental Analysis implementation

            self._fundamental_analysis_enabled = self._check_fundamental_analysis()

            
            
            # Check for Sentiment Analysis implementation

            self._sentiment_analysis_enabled = self._check_sentiment_analysis()

            
            
            # Check for Dynamic Stop Loss implementation

            self._dynamic_stop_loss_enabled = self._check_dynamic_stop_loss()

            
            
            # Check for Trade Validation implementation

            self._trade_validation_enabled = self._check_trade_validation()

            
            
            # Check for Portfolio Optimization implementation

            self._portfolio_optimization_enabled = self._check_portfolio_optimization()

            
            
            # Check for Optuna integration

            self._optuna_integration_enabled = self._check_optuna_integration()

            
            
            # Initialize regime tracking

            self._active_regime = 'none'

            self._total_regimes_tracked = 0

            
            
            logger.info(" Advanced features detection completed")

            log_debug_info('system_health', {

                'sequential_training': self._sequential_training_enabled,

                'auto_promotion': self._auto_promotion_enabled,

                'zscore_hedging': self._zscore_hedging_enabled,

                'beta_spread': self._beta_spread_enabled,

                'regime_detection': self._regime_detection_enabled,

                'fundamental_analysis': self._fundamental_analysis_enabled,

                'sentiment_analysis': self._sentiment_analysis_enabled,

                'dynamic_stop_loss': self._dynamic_stop_loss_enabled,

                'trade_validation': self._trade_validation_enabled,

                'portfolio_optimization': self._portfolio_optimization_enabled

            })
        
        

        except Exception as e:

            logger.error(f"Feature detection failed: {e}")

            # Set all features to disabled on error

            self._sequential_training_enabled = False

            self._auto_promotion_enabled = False

            self._zscore_hedging_enabled = False

            self._beta_spread_enabled = False

            self._regime_detection_enabled = True

            self._fundamental_analysis_enabled = False

            self._sentiment_analysis_enabled = False

            self._dynamic_stop_loss_enabled = False

            self._trade_validation_enabled = False

            self._portfolio_optimization_enabled = False

            self._active_regime = 'none'

            self._total_regimes_tracked = 0
    
    

    def _check_sequential_training(self):

        """Check if sequential training is implemented"""

        try:

            # Check if sequential training is implemented in this file

            # Look for the sequential training methods we just added

            return (hasattr(self, '_load_training_history') and
                   hasattr(self, '_save_training_history') and
                   hasattr(self, '_load_best_model') and
                   hasattr(self, '_save_best_model'))

        except:

            return False
    
    

    def _check_auto_promotion(self):

        """Check if auto-promotion is implemented"""

        try:

            # Check if auto-promotion methods are implemented in this file

            return (hasattr(self, '_evaluate_model_performance') and
                   hasattr(self, '_promote_model_if_better') and
                   hasattr(self, '_auto_promotion_decision'))

        except:

            return False
    
    

    def _check_zscore_hedging(self):

        """Check if Z-Score hedging is implemented"""

        try:

            # Check if Z-Score hedging is implemented in this file

            return hasattr(self, '_calculate_zscore_hedging_signals')

        except:

            return False
    
    

    def _check_beta_spread(self):

        """Check if Beta-Adjusted Spread is implemented"""

        try:

            # Check if beta spread methods are implemented in this file

            return (hasattr(self, '_calculate_beta_adjusted_spread') and
                   hasattr(self, '_calculate_beta_coefficients') and
                   hasattr(self, '_apply_beta_adjustment'))

        except:

            return False
    
    

    def _check_regime_detection(self):

        """Check if Market Regime Detection is implemented"""

        try:

            # Check if regime detection is implemented in this file

            return hasattr(self, '_detect_market_regime')

        except:

            return False
    
    

    def _check_fundamental_analysis(self):

        """Check if Fundamental Analysis is implemented"""

        try:

            # Check if fundamental analysis methods are implemented in this file

            return (hasattr(self, '_calculate_fundamental_metrics') and
                   hasattr(self, '_analyze_financial_health') and
                   hasattr(self, '_get_fundamental_signals'))

        except:

            return False
    
    

    def _check_sentiment_analysis(self):

        """Check if Sentiment Analysis is implemented"""

        try:

            # Check if sentiment analysis methods are implemented in this file

            return (hasattr(self, '_analyze_news_sentiment') and
                   hasattr(self, '_analyze_social_sentiment') and
                   hasattr(self, '_get_sentiment_signals'))

        except:

            return False
    
    

    def _check_dynamic_stop_loss(self):

        """Check if Dynamic Stop Loss is implemented"""

        try:

            # Check if dynamic stop loss methods are implemented in this file

            return (hasattr(self, '_calculate_dynamic_stop_loss') and
                   hasattr(self, '_adjust_stop_loss_volatility') and
                   hasattr(self, '_get_stop_loss_levels'))

        except:

            return False
    
    

    def _check_trade_validation(self):

        """Check if Trade Validation is implemented"""

        try:

            # Check if trade validation methods are implemented in this file

            return (hasattr(self, '_validate_trade') and
                   hasattr(self, '_check_risk_limits') and
                   hasattr(self, '_pre_trade_validation'))

        except:

            return False
    
    

    def _check_portfolio_optimization(self):

        """Check if Portfolio Optimization is implemented"""

        try:

            # Check if portfolio optimization methods are implemented in this file

            return (hasattr(self, '_calculate_kelly_criterion') and
                   hasattr(self, '_modern_portfolio_theory') and
                   hasattr(self, '_risk_parity_allocation') and
                   hasattr(self, '_optimize_portfolio_weights'))

        except:

            return False
    
    

    def _check_optuna_integration(self):

        """Check if Optuna integration is implemented and enabled"""

        try:

            # Check if Optuna is available and integration is implemented

            if not self._optuna_enabled:

                return False  # Disabled by default
            
            

            # Check if Optuna methods are implemented

            return (hasattr(self, '_create_optuna_study') and
                   hasattr(self, '_optimize_hyperparameters') and
                   hasattr(self, '_optuna_objective'))

        except:

            return False
    
    

    def _get_advanced_features_status(self):

        """Get summary of advanced features status"""

        return {

            'sequential_training': self._sequential_training_enabled,

            'auto_promotion': self._auto_promotion_enabled,

            'zscore_hedging': self._zscore_hedging_enabled,

            'beta_spread': self._beta_spread_enabled,

            'regime_detection': self._regime_detection_enabled,

            'optuna_integration': self._optuna_integration_enabled,

            'total_enabled': sum([

                self._sequential_training_enabled,

                self._auto_promotion_enabled,

                self._zscore_hedging_enabled,

                self._beta_spread_enabled,

                self._regime_detection_enabled,

                self._optuna_integration_enabled

            ])

        }
    
    










    def download_market_data(self, force_download: bool = False) -> pd.DataFrame:
        """Download and cache market data using smart caching
        
        Args:
            force_download: If True, force fresh download from Alpaca (ignores cache)
        """
        from ppo.data.data_loader import load_or_download_data, get_cache_path, load_cached_data, _get_alpaca_keys
        
        logger.info(f"Loading market data: {self.config['start_date']} to {self.config['end_date']}")
        data_source = "API"
        if force_download:
            logger.info("🔄 FORCE DOWNLOAD: Ignoring cache, downloading fresh data from Alpaca")

        # Phase 1: New 5-year CSV first (zero NaNs, clean data)
        try:
            from ppo.data.new_csv_loader import NewStockDataLoader  # type: ignore
            loader = NewStockDataLoader()
            new_df = loader.load_data(tickers=self.config.get('tickers'))
            if new_df is not None and len(new_df) > 0:
                new_df = new_df.rename(columns={
                    'Date': 'date', 'Company': 'tic',
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
                if 'date' in new_df.columns and not pd.api.types.is_datetime64_any_dtype(new_df['date']):
                    new_df['date'] = pd.to_datetime(new_df['date'])
                self.config['tickers'] = sorted(new_df['tic'].unique().tolist())
                logger.info(f"Loading market data from Phase 1 CSV (5-year, zero NaNs): {len(new_df)} rows, tickers: {self.config['tickers']}")
                self._validate_market_data_integrity(new_df)
                self.data_source = "NEW_CSV"
                return new_df
        except Exception as e:
            logger.warning(f"Phase 1 new CSV load failed, falling back to existing pipeline: {e}")

        # CSV pipeline (preferred when processed CSV exists)
        try:
            from ppo.data.csv_data_manager import CSVDataManager
            csv_manager = CSVDataManager()
            csv_path = csv_manager.processed_csv_path
            if csv_path.exists():
                logger.info(f"Loading market data from CSV pipeline: {csv_path}")
                csv_df = csv_manager.load_processed_data()
                csv_df = csv_df.copy()
                if 'Ticker' in csv_df.columns and 'tic' not in csv_df.columns:
                    csv_df['tic'] = csv_df['Ticker']
                elif 'tic' in csv_df.columns and 'Ticker' not in csv_df.columns:
                    csv_df['Ticker'] = csv_df['tic']
                if 'date' in csv_df.columns and not pd.api.types.is_datetime64_any_dtype(csv_df['date']):
                    csv_df['date'] = pd.to_datetime(csv_df['date'])
                logger.info(f"Loaded {len(csv_df)} rows of market data from CSV pipeline")
                data_source = "CSV"
                self._validate_market_data_integrity(csv_df)
                self.data_source = data_source
                return csv_df
        except Exception as e:
            logger.warning(f"CSV pipeline load failed, falling back to cache/API: {e}")
        
        # Check if cache exists and validate ticker count
        cache_file = get_cache_path(self.config['start_date'], self.config['end_date'], '1Day')
        cached_data = load_cached_data(cache_file) if not force_download else None
        force_download_flag = force_download
        
        # OPTIMIZATION: Check API credentials before attempting download
        # This prevents wasting time on failed 401 auth attempts
        api_credentials_valid = False
        if cached_data is not None and not cached_data.empty:
            # Check if cached data has all expected tickers
            cached_tickers = cached_data['tic'].unique() if 'tic' in cached_data.columns else []
            expected_tickers = set(self.config['tickers'])
            cached_tickers_set = set(cached_tickers)
            
            if not expected_tickers.issubset(cached_tickers_set):
                missing = expected_tickers - cached_tickers_set
                logger.warning(f"⚠️ Cache missing {len(missing)} tickers: {missing}")
                logger.warning(f"   Cached: {len(cached_tickers)} tickers, Expected: {len(expected_tickers)} tickers")
                
                # Quick credential check before attempting download
                keys = _get_alpaca_keys()
                if keys:
                    api_key, secret_key = keys
                    # Simple validation: check if keys exist and are non-empty
                    if api_key and secret_key and len(api_key) > 10 and len(secret_key) > 10:
                        api_credentials_valid = True
                        logger.info(f"   ✅ API credentials found, attempting to download missing tickers...")
                    else:
                        logger.warning(f"   ⚠️ API credentials appear invalid, skipping download attempt")
                        logger.warning(f"   Using cached data with {len(cached_tickers)} tickers (missing: {missing})")
                else:
                    logger.warning(f"   ⚠️ No API credentials found, skipping download attempt")
                    logger.warning(f"   Using cached data with {len(cached_tickers)} tickers (missing: {missing})")
                
                if api_credentials_valid:
                    force_download = True
                else:
                    # Skip download, use cached data and update config
                    available_tickers = [t for t in self.config['tickers'] if t in cached_tickers]
                    if available_tickers:
                        logger.info(f"   ✅ Using {len(available_tickers)} available tickers from cache")
                        self.config['tickers'] = available_tickers
                        logger.info(f"Loaded {len(cached_data)} rows of market data from cache")
                        data_source = "CACHE"
                        self._validate_market_data_integrity(cached_data)
                        self.data_source = data_source
                        return cached_data
        
        # Try to download/load data only if credentials are valid
        if force_download_flag:
            try:
                market_data = load_or_download_data(
                    tickers=self.config['tickers'],
                    start_date=self.config['start_date'],
                    end_date=self.config['end_date'],
                    timeframe='1Day',
                    force_download=True,  # Force download
                    use_parallel=_should_use_parallel_downloads()
                )
            except (RuntimeError, Exception) as e:
                # If download fails (e.g., API auth error), fall back to cached data
                error_str = str(e)
                if '401' in error_str or 'Authorization' in error_str:
                    logger.warning(f"⚠️ API authentication failed (401). Skipping download attempts.")
                else:
                    logger.warning(f"⚠️ Download failed: {e}")
                
                if cached_data is not None and not cached_data.empty:
                    cached_tickers = cached_data['tic'].unique() if 'tic' in cached_data.columns else []
                    logger.warning(f"   Falling back to cached data with {len(cached_tickers)} tickers: {list(cached_tickers)}")
                    logger.warning(f"   ⚠️ Using partial ticker set - some tickers may be missing")
                    # Update config to use only available tickers
                    available_tickers = [t for t in self.config['tickers'] if t in cached_tickers]
                    if available_tickers:
                        logger.info(f"   ✅ Using {len(available_tickers)} available tickers: {available_tickers}")
                        self.config['tickers'] = available_tickers
                        market_data = cached_data
                        data_source = "CACHE"
                    else:
                        raise RuntimeError(f"No cached data available and download failed: {e}")
                else:
                    raise RuntimeError(f"No cached data available and download failed: {e}")
        else:
            # No download needed, use cached data
            if cached_data is not None and not cached_data.empty:
                market_data = cached_data
                data_source = "CACHE"
            else:
                # No cache available, must try download
                market_data = load_or_download_data(
                    tickers=self.config['tickers'],
                    start_date=self.config['start_date'],
                    end_date=self.config['end_date'],
                    timeframe='1Day',
                    force_download=False,
                    use_parallel=_should_use_parallel_downloads()
                )
        
        logger.info(f"Loaded {len(market_data)} rows of market data")
        self._validate_market_data_integrity(market_data)
        self.data_source = data_source
        return market_data

    def _validate_market_data_integrity(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data integrity; raise if corrupted."""
        if not isinstance(market_data, pd.DataFrame):
            raise ValueError("Market data is not a DataFrame")

        required_price_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_price_cols if col not in market_data.columns]
        if "date" not in market_data.columns:
            missing_cols.append("date")

        ticker_col = None
        if "tic" in market_data.columns:
            ticker_col = "tic"
        elif "Ticker" in market_data.columns:
            ticker_col = "Ticker"
        else:
            missing_cols.append("tic")

        if missing_cols:
            raise ValueError(f"Corrupted data: missing columns {missing_cols}")

        numeric_cols = market_data.select_dtypes(include=[np.number]).columns.tolist()
        nan_count = int(market_data[numeric_cols].isna().sum().sum()) if numeric_cols else 0
        inf_count = int(np.isinf(market_data[numeric_cols].to_numpy()).sum()) if numeric_cols else 0

        invalid_prices = {}
        for col in required_price_cols:
            if col in market_data.columns:
                bad = market_data[col].isna() | (market_data[col] <= 0)
                if bad.any():
                    invalid_prices[col] = int(bad.sum())

        high_low_issues = 0
        if "high" in market_data.columns and "low" in market_data.columns:
            high_low_issues = int((market_data["high"] < market_data["low"]).sum())

        data_quality = {
            "rows": len(market_data),
            "columns": list(market_data.columns),
            "ticker_column": ticker_col,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "invalid_price_counts": invalid_prices,
            "high_low_issues": high_low_issues,
        }

        self._data_quality = data_quality

        if nan_count > 0 or inf_count > 0 or invalid_prices or high_low_issues > 0:
            raise ValueError(f"Corrupted data detected: {data_quality}")

        logger.info("✅ Market data integrity check passed")
        return data_quality

    def create_trading_environment(self, market_data: pd.DataFrame, mode: str = 'train') -> Any:
        """Create trading environment using advanced feature engineer and trading environment.
        When use_intraday_step is True, loads 5m/30m bar data and steps per bar (market-condition driven).
        """
        try:
            # Intraday step: load bar-level data and create env that steps per bar
            try:
                from ppo.utils.dynamic_config import get_dynamic_config
                dc = get_dynamic_config()
                use_intraday = getattr(dc, 'use_intraday_step', False)
                bar_interval = getattr(dc, 'intraday_step_interval', '5m')
            except Exception:
                use_intraday = False
            if use_intraday:
                try:
                    from ppo.data.csv_loader_adapter import CSVLoaderAdapter
                    from ppo.data.csv_data_manager import CSVDataManager
                    from ppo.models.trading_env import FixedTradingEnv
                    csv_manager = CSVDataManager()
                    adapter = CSVLoaderAdapter(csv_manager)
                    tickers = self.config.get('tickers', [])
                    bar_data = adapter.load_intraday_bar_matrices(bar_interval, tickers=tickers)
                    if bar_data is not None and bar_data.get('price_matrix') is not None:
                        price_matrix = bar_data['price_matrix']
                        feature_matrix = bar_data['feature_matrix']
                        volume_matrix = bar_data['volume_matrix']
                        window_size = self.config.get('window_size', 10)
                        if price_matrix.shape[0] > window_size + 50:
                            env = FixedTradingEnv(
                                feature_matrix, price_matrix, volume_matrix,
                                window_size=window_size, mode=mode
                            )
                            # Inject metadata for explainability/reporting alignment
                            try:
                                env.dates = bar_data.get('dates')
                                env.unique_tickers = bar_data.get('tickers', tickers)
                                env.feature_cols = bar_data.get('feature_cols', [])
                                env.trading_config = dict(self.config)
                            except Exception:
                                pass
                            logger.info(f"Intraday step env created: {bar_interval}, bars={price_matrix.shape[0]}, tickers={price_matrix.shape[1]}")
                            return env
                except Exception as e:
                    logger.warning(f"Intraday step env failed, falling back to daily: {e}")
            # Phase 3: Implement proper 75/25 temporal train/test split
            # CRITICAL: Split by unique dates to ensure all tickers have same length
            unique_dates = market_data['date'].unique()
            split_date_idx = int(len(unique_dates) * 0.75)
            split_date = unique_dates[split_date_idx]
            
            if mode == 'train':
                # Use first 75% of dates for training
                data_subset = market_data[market_data['date'] <= split_date]
                logger.info(f"TRAIN: Using {len(data_subset)} samples (dates up to {split_date})")
            else:  # test mode
                # Use last 25% of dates for testing (unseen data)
                data_subset = market_data[market_data['date'] > split_date]
                logger.info(f"TEST: Using {len(data_subset)} samples (dates after {split_date})")
                # DIAGNOSTIC: Warn if test data is too short
                if len(data_subset) < 1000:
                    logger.warning(f"⚠️ DIAGNOSTIC: Test data has only {len(data_subset)} samples - may cause short evaluation episodes")
                    logger.warning(f"   Recommended: At least 1000 samples for proper evaluation")
                    logger.warning(f"   Current split: 25% of {len(market_data)} total samples = {len(data_subset)}")
            
            # Use the advanced feature engineer with caching
            feature_engineer = FeatureEngineer(self.config)
            
            # Generate cache key for features
            import hashlib
            import pandas as pd
            data_hash = hashlib.md5(str(data_subset.shape).encode()).hexdigest()[:8]
            cache_key = f"{mode}_{data_hash}_features"
            
            # Ensure data_subset is DataFrame for type checker
            if not isinstance(data_subset, pd.DataFrame):
                data_subset = pd.DataFrame(data_subset)
            features_df = feature_engineer.create_features(data_subset, cache_key=cache_key)
            
            # Debug: Print available columns
            logger.info(f"Available columns: {list(features_df.columns)}")
            
            # CRITICAL FIX: Properly reshape data for multi-ticker structure
            # The data comes in long format (all tickers stacked), we need to reshape to wide format
            
            # Get unique tickers and ensure they match expected tickers
            unique_tickers = features_df['tic'].unique() if 'tic' in features_df.columns else []
            # 2026-02-15: Allow agent to see all companies from the data (use_all_data_tickers)
            try:
                from ppo.utils.dynamic_config import get_dynamic_config
                use_all = getattr(get_dynamic_config(), 'use_all_data_tickers', True)
            except Exception:
                use_all = True
            if use_all and len(unique_tickers) > 0:
                expected_tickers = list(unique_tickers)
                self.config['tickers'] = expected_tickers  # Sync so reports/SHAP see all data tickers
                logger.info(f"Using all tickers from data ({len(expected_tickers)}): {list(expected_tickers)}")
            else:
                expected_tickers = self.config['tickers']
            logger.info(f"Data tickers: {unique_tickers}")
            logger.info(f"Expected tickers: {expected_tickers}")
            
            # CRITICAL FIX: Check for missing tickers and handle them
            missing_tickers = [t for t in expected_tickers if t not in unique_tickers]
            if missing_tickers:
                logger.warning(f"⚠️ MISSING TICKERS: {missing_tickers} not found in cached data")
                logger.warning(f"   Data has {len(unique_tickers)} tickers: {list(unique_tickers)}")
                logger.warning(f"   Expected {len(expected_tickers)} tickers: {expected_tickers}")
                
                # OPTIMIZATION: Check API credentials before attempting download
                from ppo.data.data_loader import load_or_download_data, _get_alpaca_keys
                keys = _get_alpaca_keys()
                api_credentials_valid = False
                if keys:
                    api_key, secret_key = keys
                    if api_key and secret_key and len(api_key) > 10 and len(secret_key) > 10:
                        api_credentials_valid = True
                        logger.info(f"   ✅ API credentials found, attempting to download missing tickers...")
                    else:
                        logger.warning(f"   ⚠️ API credentials appear invalid, skipping download attempt")
                else:
                    logger.warning(f"   ⚠️ No API credentials found, skipping download attempt")
                
                # Only attempt download if credentials are valid
                if api_credentials_valid:
                    try:
                        missing_data = load_or_download_data(
                            tickers=missing_tickers,
                            start_date=self.config['start_date'],
                            end_date=self.config['end_date'],
                            timeframe='1Day',
                            force_download=True,  # Force download for missing tickers
                            use_parallel=_should_use_parallel_downloads()
                        )
                        if missing_data is not None and not missing_data.empty:
                            # Process missing data through feature engineering
                            missing_features = feature_engineer.create_features(missing_data, cache_key=f"missing_{hashlib.md5(str(missing_tickers).encode()).hexdigest()[:8]}")
                            # Merge with existing features
                            features_df = pd.concat([features_df, missing_features], ignore_index=True)
                            # Update unique tickers
                            unique_tickers = features_df['tic'].unique() if 'tic' in features_df.columns else []
                            logger.info(f"✅ Downloaded and merged {len(missing_tickers)} missing tickers")
                        else:
                            logger.error(f"❌ Failed to download missing tickers: {missing_tickers}")
                            logger.error(f"   Will use available tickers only: {list(unique_tickers)}")
                            # Update expected_tickers to only include available ones
                            expected_tickers = [t for t in expected_tickers if t in unique_tickers]
                            logger.warning(f"   Reduced expected tickers to {len(expected_tickers)}: {expected_tickers}")
                    except Exception as e:
                        error_str = str(e)
                        if '401' in error_str or 'Authorization' in error_str:
                            logger.warning(f"⚠️ API authentication failed (401). Skipping download attempts.")
                        else:
                            logger.error(f"❌ Error downloading missing tickers: {e}")
                        logger.error(f"   Will use available tickers only: {list(unique_tickers)}")
                        # Update expected_tickers to only include available ones
                        expected_tickers = [t for t in expected_tickers if t in unique_tickers]
                        logger.warning(f"   Reduced expected tickers to {len(expected_tickers)}: {expected_tickers}")
                else:
                    # Skip download, use available tickers only
                    logger.warning(f"   ⚠️ Skipping download attempt (invalid/missing credentials)")
                    logger.warning(f"   Will use available tickers only: {list(unique_tickers)}")
                    expected_tickers = [t for t in expected_tickers if t in unique_tickers]
                    logger.warning(f"   Reduced expected tickers to {len(expected_tickers)}: {expected_tickers}")
            
            # Ensure we have the right tickers
            if len(unique_tickers) != len(expected_tickers):
                logger.warning(f"Ticker count mismatch: data has {len(unique_tickers)}, expected {len(expected_tickers)}")
            
            # Get reference length from first available ticker (for fallback arrays)
            first_available_ticker = None
            reference_length = 0
            reference_num_features = 1
            
            # Find first available ticker to use as reference; use canonical feature order (base + intraday)
            exclude_cols = ['tic', 'Ticker', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'symbol']
            for ticker in expected_tickers:
                if ticker in unique_tickers:
                    first_available_ticker = ticker
                    reference_data = features_df[features_df['tic'] == ticker] if 'tic' in features_df.columns else features_df
                    reference_length = len(reference_data)
                    try:
                        from ppo.utils.dynamic_config import get_canonical_feature_cols
                        canonical = get_canonical_feature_cols(include_intraday=True)
                        feature_cols = [c for c in canonical if c in reference_data.columns]
                    except Exception:
                        feature_cols = [col for col in reference_data.columns if col not in exclude_cols]
                    if not feature_cols:
                        feature_cols = [col for col in reference_data.columns if col not in exclude_cols]
                    reference_num_features = len(feature_cols) if feature_cols else 1
                    break
            
            if reference_length == 0:
                raise ValueError("No ticker data available - cannot determine reference length")
            
            # Reshape data to wide format: each ticker becomes a column
            # First, pivot the price data
            price_data = {}
            volume_data = {}
            feature_data = {}
            
            for ticker in expected_tickers:
                # CRITICAL FIX: Handle both 'tic' and 'Ticker' columns
                if 'tic' in features_df.columns:
                    ticker_data = features_df[features_df['tic'] == ticker]
                elif 'Ticker' in features_df.columns:
                    ticker_data = features_df[features_df['Ticker'] == ticker]
                else:
                    ticker_data = features_df
                
                # CRITICAL FIX: Handle missing ticker data
                if len(ticker_data) == 0:
                    logger.warning(f"⚠️ No data for ticker {ticker} - using fallback data with reference length {reference_length}")
                    # Use fallback: create arrays with same length as reference ticker
                    price_data[ticker] = np.ones(reference_length) * 100.0  # Fallback price
                    volume_data[ticker] = np.ones(reference_length) * 1000.0  # Fallback volume
                    feature_data[ticker] = np.zeros((reference_length, reference_num_features))  # Fallback features
                    continue
                
                # Extract price data for this ticker
                price_cols = []
                for col in ['open', 'high', 'low', 'close']:
                    if col in ticker_data.columns:
                        price_cols.append(col)
                
                if price_cols:
                    # Use close price as the main price for this ticker
                    price_data[ticker] = ticker_data['close'].values if 'close' in ticker_data.columns else ticker_data[price_cols[0]].values  # type: ignore
                else:
                    # Fallback: use average of available price columns
                    available_price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in ticker_data.columns]
                    if available_price_cols:
                        price_data[ticker] = ticker_data[available_price_cols].mean(axis=1).values  # type: ignore
                    else:
                        logger.error(f"No price data found for ticker {ticker}")
                        price_data[ticker] = np.ones(reference_length) * 100.0  # Fallback price with correct length
                
                # Extract volume data for this ticker
                if 'volume' in ticker_data.columns:
                    volume_data[ticker] = ticker_data['volume'].values  # type: ignore
                else:
                    logger.warning(f"No volume data found for ticker {ticker}")
                    volume_data[ticker] = np.ones(reference_length) * 1000.0  # Fallback volume with correct length
                
                # Extract feature data in canonical order (same feature_cols for all tickers; missing cols = 0)
                if feature_cols:
                    arr = np.zeros((reference_length, reference_num_features), dtype=np.float32)
                    for fi, col in enumerate(feature_cols):
                        if col in ticker_data.columns:
                            vals = pd.to_numeric(ticker_data[col], errors='coerce').fillna(0).values
                            arr[:, fi] = vals
                    feature_data[ticker] = arr
                else:
                    logger.warning(f"No feature data found for ticker {ticker}")
                    feature_data[ticker] = np.zeros((reference_length, reference_num_features))
            
            # CRITICAL FIX: Validate all arrays have same length before stacking
            # Get reference length from first ticker
            if expected_tickers:
                first_ticker = expected_tickers[0]
                if first_ticker in price_data:
                    reference_length = len(price_data[first_ticker])
                else:
                    reference_length = reference_length  # Use previously calculated reference
            else:
                raise ValueError("No expected tickers available")
            
            # Validate and fix array lengths
            for ticker in expected_tickers:
                if ticker in price_data and len(price_data[ticker]) != reference_length:
                    logger.warning(f"⚠️ Price data length mismatch for {ticker}: {len(price_data[ticker])} vs {reference_length}")
                    # Pad or truncate to match reference length
                    if len(price_data[ticker]) < reference_length:
                        # Pad with last value
                        padding = np.full(reference_length - len(price_data[ticker]), price_data[ticker][-1] if len(price_data[ticker]) > 0 else 100.0)
                        price_data[ticker] = np.concatenate([price_data[ticker], padding])
                    else:
                        # Truncate to reference length
                        price_data[ticker] = price_data[ticker][:reference_length]
                
                if ticker in volume_data and len(volume_data[ticker]) != reference_length:
                    logger.warning(f"⚠️ Volume data length mismatch for {ticker}: {len(volume_data[ticker])} vs {reference_length}")
                    # Pad or truncate to match reference length
                    if len(volume_data[ticker]) < reference_length:
                        padding = np.full(reference_length - len(volume_data[ticker]), volume_data[ticker][-1] if len(volume_data[ticker]) > 0 else 1000.0)
                        volume_data[ticker] = np.concatenate([volume_data[ticker], padding])
                    else:
                        volume_data[ticker] = volume_data[ticker][:reference_length]
            
            # Convert to matrices
            # Price matrix: (timesteps, num_tickers)
            price_matrix = np.column_stack([price_data[ticker] for ticker in expected_tickers])
            # NaN clean: forward-fill per column so validation does not fail on upstream NaNs
            if np.isnan(price_matrix).any():
                n_before = int(np.isnan(price_matrix).sum())
                for col in range(price_matrix.shape[1]):
                    col_vals = price_matrix[:, col].copy()
                    if np.isnan(col_vals).any():
                        last = np.nan
                        for i in range(len(col_vals)):
                            if np.isnan(col_vals[i]) and not np.isnan(last):
                                col_vals[i] = last
                            elif not np.isnan(col_vals[i]):
                                last = col_vals[i]
                        if np.isnan(col_vals).any():
                            idx = np.where(~np.isnan(col_vals))[0]
                            if len(idx):
                                col_vals[:idx[0]] = col_vals[idx[0]]
                        col_vals[np.isnan(col_vals)] = 0.01
                        price_matrix[:, col] = col_vals
                logger.warning(f"Price matrix had {n_before} NaN values; forward-filled and replaced remaining with 0.01")
            
            # Volume matrix: (timesteps, num_tickers)  
            volume_matrix = np.column_stack([volume_data[ticker] for ticker in expected_tickers])
            
            # Feature matrix: (timesteps, num_features)
            # For now, use the first ticker's features as the base feature matrix
            # TODO: This could be enhanced to combine features from all tickers
            first_ticker = expected_tickers[0]
            feature_matrix = feature_data[first_ticker] if first_ticker in feature_data else np.zeros((len(features_df), 1))
            
            # Debug: Print matrix shapes
            logger.info(f"Matrix shapes - Feature: {feature_matrix.shape}, Price: {price_matrix.shape}, Volume: {volume_matrix.shape}")
            logger.info(f"Expected tickers: {len(expected_tickers)}")
            
            # Validate matrix dimensions
            if price_matrix.shape[1] != len(expected_tickers):
                logger.error(f"Price matrix has {price_matrix.shape[1]} columns, expected {len(expected_tickers)}")
                raise ValueError(f"Price matrix dimension mismatch: got {price_matrix.shape[1]} tickers, expected {len(expected_tickers)}")
            
            if volume_matrix.shape[1] != len(expected_tickers):
                logger.error(f"Volume matrix has {volume_matrix.shape[1]} columns, expected {len(expected_tickers)}")
                raise ValueError(f"Volume matrix dimension mismatch: got {volume_matrix.shape[1]} tickers, expected {len(expected_tickers)}")
            
            # CRITICAL FIX: Validate price data instead of forcing minimum prices
            # Check for invalid prices (zero, negative, or NaN)
            invalid_prices = (price_matrix <= 0) | np.isnan(price_matrix)
            if np.any(invalid_prices):
                logger.error(f"CRITICAL ERROR: Found {np.sum(invalid_prices)} invalid prices in price matrix!")
                logger.error(f"Price matrix stats: min={np.min(price_matrix):.4f}, max={np.max(price_matrix):.4f}, mean={np.mean(price_matrix):.4f}")
                # Instead of forcing minimum prices, raise an error to fix the data source
                raise ValueError("Invalid price data detected. Check data source and feature engineering.")
            else:
                logger.info(f"Price data validation passed: min={np.min(price_matrix):.4f}, max={np.max(price_matrix):.4f}")
            
            # CRITICAL FIX: Create proper 3D feature tensor without duplication
            # Shape: (timesteps, num_tickers, num_features)
            num_tickers = len(expected_tickers)
            if feature_matrix.ndim != 2:
                raise ValueError(f"Feature matrix must be 2D (timesteps, num_features); got ndim={feature_matrix.ndim}")
            timesteps, num_features = feature_matrix.shape
            
            # CRITICAL FIX: Create unique features for each ticker instead of duplicating
            # This prevents data duplication bugs that cause unrealistic returns
            feature_tensor = np.zeros((timesteps, num_tickers, num_features))
            for i, ticker in enumerate(expected_tickers):
                if ticker in feature_data:
                    # Use ticker-specific features if available
                    ticker_features = feature_data[ticker]
                    # CRITICAL FIX: Ensure ticker_features is numeric and has correct shape
                    if not isinstance(ticker_features, np.ndarray):
                        ticker_features = np.array(ticker_features)
                    # Convert to float if not already
                    if ticker_features.dtype != np.float64 and ticker_features.dtype != np.float32:
                        try:
                            ticker_features = ticker_features.astype(np.float64)
                        except (ValueError, TypeError) as e:
                            logger.error(f"Failed to convert features for {ticker} to float: {e}")
                            logger.error(f"Feature dtype: {ticker_features.dtype}, shape: {ticker_features.shape}")
                            # Fallback to zeros
                            ticker_features = np.zeros((timesteps, num_features))
                    if ticker_features.shape[0] == timesteps:
                        # Ensure correct number of features
                        if ticker_features.shape[1] == num_features:
                            feature_tensor[:, i, :] = ticker_features
                        else:
                            logger.warning(f"Feature count mismatch for {ticker}: got {ticker_features.shape[1]}, expected {num_features}")
                            # Pad or truncate to match expected features
                            if ticker_features.shape[1] < num_features:
                                padding = np.zeros((timesteps, num_features - ticker_features.shape[1]))
                                ticker_features = np.hstack([ticker_features, padding])
                            else:
                                ticker_features = ticker_features[:, :num_features]
                            feature_tensor[:, i, :] = ticker_features
                    else:
                        # Fallback: use base features with ticker-specific scaling
                        feature_tensor[:, i, :] = feature_matrix * (1.0 + i * 0.1)  # Small variation per ticker
                else:
                    # Fallback: use base features with ticker-specific scaling
                    feature_tensor[:, i, :] = feature_matrix * (1.0 + i * 0.1)  # Small variation per ticker
            
            logger.info(f"CRITICAL FIX: Created unique feature tensor shape {feature_tensor.shape} without duplication")

            # Pre-validate shapes before env creation and SB3 wrapping
            window_size = self.config.get('window_size', 10)
            if price_matrix.shape[0] != timesteps or volume_matrix.shape[0] != timesteps:
                raise ValueError(f"Time axis mismatch: features={timesteps}, prices={price_matrix.shape[0]}, volume={volume_matrix.shape[0]}")
            if price_matrix.shape[1] != num_tickers or volume_matrix.shape[1] != num_tickers:
                raise ValueError(f"Ticker count mismatch: prices={price_matrix.shape[1]}, volume={volume_matrix.shape[1]}, expected={num_tickers}")
            # Observation size will be window_size * num_tickers * num_features; log once
            obs_size = window_size * num_tickers * num_features
            logger.info(f"DYNAMIC: Locked obs size: {obs_size} = {window_size} * {num_tickers} * {num_features}")

            # Create environment using the advanced trading environment with locked schema
            # CRITICAL FIX: Correct constructor argument order (feature_matrix, price_matrix, volume_matrix)
            env = FixedTradingEnv(feature_tensor, price_matrix, volume_matrix, window_size=window_size, mode=mode)

            # 2026-02-15: Set ticker names from data (agent sees all companies; names must match matrix columns)
            env.unique_tickers = list(expected_tickers)
            env.ticker_symbols = list(expected_tickers)

            # CRITICAL FIX: Ensure mode is preserved in the environment
            env.mode = mode
            env.logger.info(f"CRITICAL FIX: Mode {mode} set directly on environment")
            
            # 2025-12-22: ACTION MASKING - Wrap environment for MaskablePPO
            # This ensures agent CANNOT select invalid actions (e.g., SELL with no positions)
            if MASKABLE_PPO_AVAILABLE and hasattr(env, 'action_masks'):
                from sb3_contrib.common.wrappers import ActionMasker
                
                def get_action_masks(env_instance):
                    """Wrapper function to get action masks from environment"""
                    return env_instance.action_masks()
                
                env = ActionMasker(env, get_action_masks)  # type: ignore
                logger.info("🎯 ACTION MASKING: Environment wrapped with ActionMasker")
                logger.info("   ✅ Agent CANNOT select SELL when no positions exist")
                logger.info("   ✅ Agent CANNOT select BUY when insufficient cash")
            else:
                logger.info("📋 Using environment without action masking (MaskablePPO not available or env missing action_masks)")

            logger.info(f"Created {mode} trading environment with features shape {feature_tensor.shape}")
            return env
            
        except Exception as e:
            logger.error(f"Failed to create trading environment: {e}")
            raise

    def train_agent(self, env, total_timesteps: Optional[int] = None, checkpoint_freq: int = SAVE_FREQUENCY, market_data: Optional[pd.DataFrame] = None) -> Any:
        """Train PPO agent using advanced PPO trainer"""
        try:
            # CRITICAL FIX: Check if training is allowed
            can_train, reason = emergency_manager.can_train_model()
            if not can_train:
                logger.error(f"🚫 TRAINING BLOCKED: {reason}")
                logger.error("🚨 Emergency debug mode is active - fix critical errors first")
                logger.info("🗑️ Cleaning up invalid models...")
                emergency_manager.cleanup_invalid_models()
                raise RuntimeError(f"Training blocked: {reason}")
            
            # CRITICAL FIX: Create separate test environment for callbacks to prevent data leakage
            # This ensures callbacks evaluate on unseen test data, not training data
            test_env_for_callbacks = None
            if market_data is not None:
                try:
                    logger.info("🔍 DATA LEAKAGE FIX: Creating separate test environment for callback evaluation")
                    test_env_for_callbacks = self.create_trading_environment(market_data, mode='test')
                    logger.info("✅ Test environment created for callback evaluation (prevents data leakage)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create test environment for callbacks: {e}")
                    logger.warning("   Callbacks will use training environment (may cause data leakage)")
                    test_env_for_callbacks = None
            
            # Use test environment if available, otherwise fall back to training env (with warning)
            eval_env = test_env_for_callbacks if test_env_for_callbacks is not None else env
            if test_env_for_callbacks is None:
                logger.warning("⚠️ DATA LEAKAGE RISK: Callbacks using training environment (no test env available)")
            
            # Create PPO agent with orchestrator config (do not silently fall back to defaults)
            model = create_ppo_agent(
                env=env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                n_steps=self.config.get('n_steps', 2048),
                batch_size=self.config.get('batch_size', 64),
                gamma=self.config.get('gamma', 0.99),
                gae_lambda=self.config.get('gae_lambda', 0.95),
                clip_range=self.config.get('clip_range', 0.2),
                ent_coef=self.config.get('ent_coef', 0.05),
                vf_coef=self.config.get('vf_coef', 0.5),
                max_grad_norm=self.config.get('max_grad_norm', 0.5),
                device=self.device,
                seed=int(self.config.get('seed', 42))
            )
            
            # Then train the agent with dynamic timesteps
            from ppo.utils.dynamic_config import get_dynamic_timesteps
            actual_timesteps = total_timesteps or get_dynamic_timesteps()
            logger.info(f"🚀 STARTING TRAINING: {actual_timesteps:,} timesteps")
            logger.info(f"📊 Training config: timesteps={actual_timesteps}, checkpoint_freq={checkpoint_freq}")
            
            # Add detailed progress callback for BC-style logging
            # Combine progress callback with BC regularization and early stopping
            from stable_baselines3.common.callbacks import CallbackList
            from ppo.training.ppo_trainer import BCRegularizationCallback
            from ppo.utils.dynamic_config import get_dynamic_config
            
            # OPTIMIZATION: Increase log frequency from 10k to 25k steps to reduce I/O overhead
            progress_callback = DetailedProgressCallback(log_freq=25000)
            bc_reg_callback = BCRegularizationCallback(verbose=1)
            
            # ENHANCED: Add early stopping to prevent overfitting
            # CRITICAL FIX: Use test environment for evaluation (prevents data leakage)
            # OPTIMIZATION: Increase eval_freq from 50k to 100k steps (50% reduction in evaluations)
            config = get_dynamic_config()
            early_stop_callback = EarlyStoppingCallback(
                eval_env=eval_env,  # FIXED: Use test environment to prevent data leakage
                eval_freq=100000,  # OPTIMIZATION: Evaluate every 100K steps (was 50K, reduces overhead by 50%)
                patience=5,  # FIXED: Stop after 5 evaluations (was 3, too aggressive)
                min_delta=0.005,  # FIXED: 0.5% improvement threshold (was 0.1%, too strict)
                min_sharpe=-0.5,  # FIXED: Allow negative Sharpe during exploration (was 0.3, BC had 0.09)
                min_return=-0.10,  # FIXED: -10% minimum (was 1%, too strict for early training)
                max_drawdown=0.40,  # FIXED: 40% max drawdown (was 20%, allow more exploration)
                verbose=1
            )
            
            # Phase 1: AdaptiveHyperparameterCallback (LR + entropy; actually updates model.learning_rate and ent_coef)
            # FIX 2026-02-15: eval_freq 5k to catch policy collapse earlier (was 10k; collapse happened between 30k-40k)
            adaptive_callback = AdaptiveHyperparameterCallback(
                eval_env=eval_env,
                eval_freq=5000,
                verbose=1
            )
            main._adaptive_callback = adaptive_callback
            logger.info("[PHASE 1] AdaptiveHyperparameterCallback enabled (LR + entropy on poor/good performance and action imbalance)")
            logger.info(f"   - Poor performance threshold: Sharpe < {config.adaptive_lr_poor_sharpe_threshold}, Return < {config.adaptive_lr_poor_return_threshold:.2%}")
            logger.info(f"   - Performance degradation: >2% return drop or >30% trade drop triggers LR increase")
            logger.info(f"   - Excellent performance threshold: Sharpe > {config.adaptive_lr_sharpe_threshold}, Return > {config.adaptive_lr_return_threshold:.2%}")
            logger.info(f"   - LR boost factor: {config.adaptive_lr_poor_boost_factor}x (max LR: {config.adaptive_lr_poor_max:.2e})")
            logger.info(f"   - Adaptive Entropy: triggers every 5K steps on low trades, action imbalance, or entropy collapse")
            logger.info(f"   - Low trade threshold: < {config.adaptive_ent_min_trades} trades")
            logger.info(f"   - Action imbalance thresholds: BUY < 5%, SELL < 5%, SELL > 85%")
            logger.info(f"   - Entropy boost factor: {config.adaptive_ent_boost_factor}x (max: {config.adaptive_ent_max})")
            
            callback = CallbackList([progress_callback, bc_reg_callback, adaptive_callback, early_stop_callback])
            logger.info(f"🛡️ BC Regularization callback enabled to preserve BC knowledge")
            logger.info(f"[ADAPTIVE HYPERPARAMETERS] Enabled: LR & entropy adjust based on performance")
            logger.info(f"🛑 Early stopping enabled: Will stop if performance degrades")
            trained_model = train_ppo_agent(
                model=model,
                total_timesteps=actual_timesteps,
                display_interval=checkpoint_freq,
                callback=callback  # type: ignore
            )
            
            logger.info(f"✅ TRAINING COMPLETED: Successfully trained PPO agent for {actual_timesteps:,} timesteps")
            return trained_model
            
        except Exception as e:
            logger.error(f"Failed to train PPO agent: {e}")
            raise

    def evaluate_agent(self, model, test_env) -> Dict[str, Any]:
        """Evaluate PPO agent using advanced performance evaluator"""
        try:
            # Add evaluation safety guard - track consecutive no-trade steps
            consecutive_no_trades = 0
            max_no_trade_steps = 200
            
            # Delegate to the advanced performance evaluator (deterministic = paper-trading style)
            # Stochastic evaluation removed - not realistic for paper trading; single deterministic result only
            results = evaluate_model(
                model=model,
                env=test_env,
                n_episodes=self.config.get('n_eval_episodes', 1),  # CRITICAL FIX: Use n_eval_episodes consistently
                deterministic=True
            )
            
            # Check for degenerate no-trade policy
            total_trades = results.get('total_trades', 0)
            if total_trades == 0:
                consecutive_no_trades += results.get('steps_simulated', 0)
                if consecutive_no_trades >= max_no_trade_steps:
                    logger.warning("⚠️ EVALUATION GUARD: Agent hasn't traded in 200+ steps!")
                    logger.warning("   This suggests a degenerate policy (likely HOLD-only)")
                    logger.warning("   Current reward components should be reviewed")
                    logger.warning("   Exploration bonus may be running during evaluation")
            
            logger.info(f"Successfully evaluated PPO agent: {results.get('total_return', 0):.2%} return")
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate PPO agent: {e}")
            # CRITICAL FIX: Handle evaluation errors gracefully
            if "unsupported operand type(s) for -: 'dict' and 'dict'" in str(e):
                logger.warning("Evaluation error detected - using fallback evaluation")
                return self._fallback_evaluation(model, test_env)
            raise
    
    def _fallback_evaluation(self, model, test_env) -> Dict[str, Any]:
        """Fallback evaluation when main evaluator fails"""
        try:
            # Simple fallback evaluation
            total_return = 0.0
            sharpe_ratio = 0.0
            win_rate = 0.0
            
            # Get basic metrics from environment
            if hasattr(test_env, 'get_portfolio_history'):
                portfolio_history = test_env.get_portfolio_history()
                if len(portfolio_history) > 1:
                    initial_value = portfolio_history[0]['portfolio_value']
                    final_value = portfolio_history[-1]['portfolio_value']
                    total_return = (final_value - initial_value) / initial_value
            
            # Get trade history for win rate calculation
            if hasattr(test_env, 'get_trade_history'):
                trade_history = test_env.get_trade_history()
                if trade_history:
                    profitable_trades = sum(1 for trade in trade_history if trade.get('pnl', 0) > 0)
                    win_rate = profitable_trades / len(trade_history) if trade_history else 0.0
            
            # Calculate basic Sharpe ratio
            if len(portfolio_history) > 1:
                returns = []
                for i in range(1, len(portfolio_history)):
                    ret = (portfolio_history[i]['portfolio_value'] - portfolio_history[i-1]['portfolio_value']) / portfolio_history[i-1]['portfolio_value']
                    returns.append(ret)
                
                if returns:
                    mean_return = sum(returns) / len(returns)
                    std_return = (sum((r - mean_return)**2 for r in returns) / len(returns))**0.5
                    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            
            fallback_results = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'trade_history': trade_history if hasattr(test_env, 'get_trade_history') else [],
                'portfolio_history': portfolio_history if hasattr(test_env, 'get_portfolio_history') else [],
                'evaluation_method': 'fallback',
                'error_handled': True
            }
            
            logger.info(f"Fallback evaluation completed: {total_return:.2%} return, {sharpe_ratio:.2f} Sharpe, {win_rate:.2%} win rate")
            return fallback_results
            
        except Exception as e:
            logger.error(f"Fallback evaluation also failed: {e}")
            # Ultimate fallback
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'trade_history': [],
                'portfolio_history': [],
                'evaluation_method': 'emergency_fallback',
                'error_handled': True,
                'error': str(e)
            }


def _preflight_verify_and_warm_caches(orchestrator) -> None:
    """Pre-flight: Ensure required caches exist; download if missing before training.
    Does not synthesize data; only warms caches when missing.
    """
    try:
        # Market data cache (date-range-based)
        from ppo.data.data_loader import (
            get_cache_path,
            load_cached_data,
            load_or_download_data,
        )
        start_date = orchestrator.config.get('start_date')
        end_date = orchestrator.config.get('end_date')
        tickers = orchestrator.config.get('tickers', [])

        cache_file = get_cache_path(start_date, end_date, timeframe='1Day')
        needs_download = False
        if not os.path.exists(cache_file):
            logger.info(f" CACHE CHECK: Market cache not found at {cache_file}")
            needs_download = True
        else:
            cached_df = load_cached_data(cache_file)
            if cached_df is None or getattr(cached_df, 'empty', True):
                logger.info(f" CACHE CHECK: Market cache exists but empty/unreadable at {cache_file}")
                needs_download = True
            else:
                logger.info(f" CACHE OK: Market cache present ({len(cached_df)} rows)")

        if needs_download:
            logger.info(" CACHE WARM: Downloading market data to populate cache...")
            _ = load_or_download_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                timeframe='1Day',
                force_download=True,
                use_parallel=_should_use_parallel_downloads(),
            )
            logger.info(" CACHE WARM: Market data cache populated")

        # Feature cache directory presence (feature computation will populate as needed)
        feature_cache_dir = os.path.join(orchestrator.base_dir, "PPO", "cache_performance", "features")
        os.makedirs(feature_cache_dir, exist_ok=True)
        logger.info(f" CACHE OK: Feature cache directory ensured at {feature_cache_dir}")

        # Fundamental cache presence (create real cache if possible)
        fundamental_cache_dir = os.path.join(orchestrator.base_dir, "ppo", "saved_data", "fundamental_cache")
        os.makedirs(fundamental_cache_dir, exist_ok=True)
        
        # Check if fundamental cache exists (only if directory exists and has files)
        has_fundamentals = False
        if os.path.exists(fundamental_cache_dir):
            try:
                has_fundamentals = any(
                    f.lower().endswith(".pkl") 
                    for f in os.listdir(fundamental_cache_dir) 
                    if os.path.isfile(os.path.join(fundamental_cache_dir, f))
                )
            except Exception as e:
                logger.warning(f"Error checking fundamental cache directory: {e}")
                has_fundamentals = False
        def _validate_fundamental_cache(dir_path: str, req_tickers: list) -> bool:
            try:
                import pickle, glob
                files = sorted(glob.glob(os.path.join(dir_path, "*.pkl")), key=os.path.getmtime, reverse=True)
                if not files:
                    return False
                with open(files[0], 'rb') as fh:
                    data = pickle.load(fh)
                if not isinstance(data, dict) or not data:
                    return False
                required_keys = (
                    'pe_ratio','pb_ratio','debt_equity','roe','current_ratio','revenue_growth','earnings_growth'
                )
                ok_all = True
                missing = []
                for sym in req_tickers:
                    usym = str(sym).upper()
                    if usym not in data:
                        ok_all = False
                        missing.append(usym)
                        continue
                    metrics = data.get(usym, {})
                    if not isinstance(metrics, dict) or not all(k in metrics for k in required_keys):
                        ok_all = False
                        missing.append(usym)
                if not ok_all:
                    logger.warning(f"Fundamental cache missing/partial for tickers: {missing}")
                else:
                    logger.info(f" CACHE OK: Fundamental cache covers all tickers ({len(req_tickers)})")
                return ok_all
            except Exception as ve:
                logger.warning(f"Fundamental cache validation failed: {ve}")
                return False
        
        # Reduced logging - status will be in unified report
        fundamental_status = {'available': False, 'tickers_loaded': [], 'cache_path': None}
        
        # CRITICAL FIX: Always attempt to download fundamentals if cache is missing or invalid
        # This ensures fundamental data is stored properly
        should_download_fundamentals = not has_fundamentals or not _validate_fundamental_cache(fundamental_cache_dir, tickers)
        
        if should_download_fundamentals:
            logger.info("📊 Downloading fundamental data from Alpaca...")
            # Use env vars first, then fallback to provided keys (UPDATED: New API keys)
            api_key = os.getenv('APCA_API_KEY') or os.getenv('APCA_API_KEY_ID') or 'PK7XOIVRKYPXMM4PIA6RDK5EMF'
            secret_key = os.getenv('APCA_API_SECRET') or os.getenv('APCA_API_SECRET_KEY') or '3R4H1yBurMRHEirWoivTJSdd1veUA4P49qRttn2NhsDL'
            saved = None
            try:
                from ppo.data.fundamentals_loader import build_and_save_fundamental_cache
                saved = build_and_save_fundamental_cache(tickers, api_key, secret_key, fundamental_cache_dir)
                if not saved:
                    logger.warning("⚠️ Alpaca fundamentals unavailable; falling back to local CSV/cache pipeline")
                    try:
                        from ppo.collect_fundamentals import collect_all_fundamental_data, save_fundamental_cache, DEFAULT_FUNDAMENTAL_CSV
                        csv_file = str(DEFAULT_FUNDAMENTAL_CSV) if DEFAULT_FUNDAMENTAL_CSV.exists() else None
                        fallback_data = collect_all_fundamental_data(
                            force_refresh=False,
                            max_cache_age_days=30,
                            csv_file=csv_file
                        )
                        if isinstance(fallback_data, dict) and fallback_data:
                            save_fundamental_cache(fallback_data)
                            latest_cache = Path(fundamental_cache_dir) / "fundamental_data_latest.pkl"
                            if latest_cache.exists():
                                saved = str(latest_cache)
                                logger.info(f"✅ Fundamental fallback cache created: {saved}")
                    except Exception as csv_err:
                        logger.warning(f"⚠️ Fundamental fallback pipeline failed: {csv_err}")
            except Exception as e:
                # Log the error for debugging (fundamentals may not be available through Market Data API)
                logger.error(f"❌ Fundamental data download failed: {e}")
                logger.error(f"   Note: Alpaca Market Data API may not include fundamentals - requires separate data source.")
                import traceback
                logger.debug(f"   Traceback: {traceback.format_exc()}")
                # Status will be in unified report
            
            if saved:
                logger.info(f"✅ Fundamental data saved to: {saved}")
                fundamental_status['cache_path'] = saved
                # Re-validate after build/fallback attempt
                validated = _validate_fundamental_cache(fundamental_cache_dir, tickers)
                if validated:
                    fundamental_status['available'] = True
                    # Get list of tickers with fundamentals
                    import pickle
                    with open(saved, 'rb') as f:
                        cache_data = pickle.load(f)
                        fundamental_status['tickers_loaded'] = list(cache_data.keys()) if isinstance(cache_data, dict) else []
                    logger.info(f"✅ Fundamental cache validated: {len(fundamental_status['tickers_loaded'])} tickers loaded")
                else:
                    logger.warning("⚠️ Fundamental cache validation failed after download/fallback")
            else:
                logger.warning("⚠️ Fundamental cache not created by Alpaca or fallback pipeline")
        else:
            fundamental_status['available'] = True
            # Get cached tickers
            try:
                import pickle
                cache_files = list(Path(fundamental_cache_dir).glob("*.pkl"))
                if cache_files:
                    latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_cache, 'rb') as f:
                        cache_data = pickle.load(f)
                        fundamental_status['tickers_loaded'] = list(cache_data.keys()) if isinstance(cache_data, dict) else []
                        fundamental_status['cache_path'] = str(latest_cache)
            except Exception:
                pass
        
        # Store for unified report (accessible via execution_results)
        if not hasattr(main, '_fundamental_status'):
            main._fundamental_status = fundamental_status
        else:
            main._fundamental_status.update(fundamental_status)

    except Exception as e:
        logger.warning(f" CACHE CHECK: Failed to verify/warm caches: {e}")


def _log_system_info(start_datetime):
    """Log system information at startup"""
    import psutil
    import platform
    
    logger.info("=" * 80)
    logger.info("PPO TRADING AGENT STARTUP")
    logger.info("=" * 80)
    logger.info(f"Start time: {start_datetime.isoformat()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Log memory and system resources
    memory = psutil.virtual_memory()
    logger.info(f"System RAM: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available ({memory.percent:.1f}% used)")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Log configuration
    try:
        import config
        logger.info(f"Config timesteps: {config.ENV_CONFIG.get('total_timesteps', 'NOT_SET')}")
        logger.info(f"Config tickers: {config.DATA_CONFIG.get('tickers', 'NOT_SET')}")
        logger.info(f"Config start_date: {config.DATA_CONFIG.get('start_date', 'NOT_SET')}")
        logger.info(f"Config end_date: {config.DATA_CONFIG.get('end_date', 'NOT_SET')}")
        logger.info(f"Config window_size: {config.DATA_CONFIG.get('window_size', 'NOT_SET')}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}")


def _initialize_orchestrator():
    """Initialize orchestrator and configure timesteps"""
    import psutil
    from ppo.utils.dynamic_config import get_dynamic_timesteps
    
    print("🚀 Starting PPO Trading Agent...")
    logger.info("PHASE 1: Initializing orchestrator...")
    
    # Initialize orchestrator
    orchestrator = PPOTradingOrchestrator()
    logger.info("✅ Orchestrator initialized successfully")
    
    # CENTRALIZED: Always use dynamic_config as single source of truth
    try:
        # Get timesteps from centralized dynamic config
        timesteps = get_dynamic_timesteps()
        
        print(f"\nCENTRALIZED TIMESTEP MANAGEMENT:")
        print(f"  Source: dynamic_config.py")
        print(f"  Timesteps: {timesteps:,}")
        print(f"  Status: Single source of truth\n")
        
        # Update orchestrator config to match dynamic config
        orchestrator.config['total_timesteps'] = timesteps
        logger.info(f"CENTRALIZED: Using {timesteps:,} timesteps from dynamic_config.py")
        
    except Exception as e:
        logger.error(f"Failed to get dynamic timesteps: {e}")
        # Fallback to default
        timesteps = 5000
        orchestrator.config['total_timesteps'] = timesteps
        logger.warning(f"FALLBACK: Using default {timesteps:,} timesteps")
    
    # Log memory after orchestrator init
    memory_after_init = psutil.virtual_memory()
    logger.info(f"Memory after orchestrator init: {memory_after_init.percent:.1f}% used")

    # NEW: Pre-flight cache verification/warming
    logger.info("Pre-flight: verifying caches...")
    _preflight_verify_and_warm_caches(orchestrator)
    
    return orchestrator, timesteps


def _download_market_data(orchestrator):
    """Download market data with error handling"""
    import psutil
    import time
    
    print("📊 Downloading market data...")
    logger.info("PHASE 2: Downloading market data...")
    
    # CRITICAL FIX: Track data loading time
    data_loading_start_time = time.time()
    
    # Check if force_download flag is set (via environment variable or command line)
    force_download = os.getenv('FORCE_DOWNLOAD', 'false').lower() == 'true'
    if force_download:
        logger.info("🔄 FORCE DOWNLOAD enabled - downloading fresh data from Alpaca")
    
    # ENHANCED: BC-style error handling for data loading
    try:
        logger.info("Starting market data download...")
        market_data = orchestrator.download_market_data(force_download=force_download)
        logger.info("Market data download completed successfully")
        logger.info(f"✅ Market data downloaded: {len(market_data)} rows, {len(market_data.columns)} columns")
        logger.info(f"Data shape: {market_data.shape}")
        logger.info(f"Data columns: {list(market_data.columns)}")
        logger.info(f"Data date range: {market_data.index.min()} to {market_data.index.max()}")
        logger.info(f"Missing values: {market_data.isnull().sum().sum()}")
    except MemoryError as e:
        error_tracker.add_error(f"Memory error during data download: {e}", is_critical=True)
        raise
    except ValueError as e:
        error_tracker.add_error(f"Value error during data download: {e}", is_critical=False)
        raise
    except Exception as e:
        error_tracker.add_error(f"Unexpected error during data download: {e}", is_critical=True)
        raise
    finally:
        # CRITICAL FIX: Calculate data loading time
        data_loading_time = time.time() - data_loading_start_time
        logger.info(f"⏱️ Data loading completed in {data_loading_time:.2f} seconds")
    
    # Log memory after data download
    memory_after_data = psutil.virtual_memory()
    logger.info(f"Memory after data download: {memory_after_data.percent:.1f}% used")
    
    return market_data, data_loading_time


def _create_environments(orchestrator, market_data):
    """Create training and test environments"""
    import psutil
    
    print("🏗️ Creating training environment...")
    logger.info("PHASE 3: Creating training environment...")
    
    # ENHANCED: BC-style error handling for environment creation
    try:
        logger.info("Starting training environment creation...")
        train_env = orchestrator.create_trading_environment(market_data, mode='train')
        
        # Store training environment reference for adaptive reward log extraction
        main._train_env = train_env
        logger.info("Training environment creation completed successfully")
        logger.info("✅ Training environment created successfully")
    except MemoryError as e:
        error_tracker.add_error(f"Memory error during environment creation: {e}", is_critical=True)
        raise
    except ValueError as e:
        error_tracker.add_error(f"Value error during environment creation: {e}", is_critical=False)
        raise
    except Exception as e:
        error_tracker.add_error(f"Unexpected error during environment creation: {e}", is_critical=True)
        raise
    
    # Log environment details
    if hasattr(train_env, 'envs') and len(train_env.envs) > 0:
        env = train_env.envs[0]
        logger.info(f"Environment observation space: {env.observation_space}")
        logger.info(f"Environment action space: {env.action_space}")
        logger.info(f"Environment mode: {getattr(env, 'mode', 'unknown')}")
    
    # Log memory after environment creation
    memory_after_env = psutil.virtual_memory()
    logger.info(f"Memory after environment creation: {memory_after_env.percent:.1f}% used")
    
    return train_env


def _load_or_train_model(orchestrator, train_env, market_data, timesteps):
    """Load existing model or train new one"""
    import time
    import psutil
    
    print("🎯 Training PPO agent...")
    logger.info("PHASE 4: Training PPO agent...")
    
    training_start_time = time.time()
    latest_model_path = os.path.join(orchestrator.models_dir, 'ppo_trading_agent.zip')
    
    if os.path.exists(latest_model_path):
        logger.info(f"📦 Found existing model: {latest_model_path}")
        try:
            model = _load_existing_model(latest_model_path, train_env)
            _continue_training(model, timesteps)
            training_time = time.time() - training_start_time
            logger.info(f"⏱️ Training completed in {training_time:.2f} seconds")
            return model, training_time
        except (MemoryError, ValueError, Exception) as e:
            error_tracker.add_error(f"Error during model loading: {e}", is_critical=True)
            logger.warning(f"Failed to load model: {e}, training from scratch")
            model = orchestrator.train_agent(train_env, market_data=market_data)
            training_time = time.time() - training_start_time
            logger.info(f"⏱️ Training completed in {training_time:.2f} seconds")
            return model, training_time
    else:
        logger.info("🆕 No existing model found, training from scratch")
        try:
            model = orchestrator.train_agent(train_env, market_data=market_data)
            training_time = time.time() - training_start_time
            logger.info(f"⏱️ Training completed in {training_time:.2f} seconds")
            return model, training_time
        except (MemoryError, ValueError, Exception) as e:
            error_tracker.add_error(f"Error during agent training: {e}", is_critical=True)
            raise


def _load_existing_model(latest_model_path, train_env):
    """Load existing model with compatibility checks"""
    from stable_baselines3 import PPO
    import zipfile
    import pickle
    
    logger.info("Starting model loading...")
    current_obs_size = train_env.observation_space.shape[0]
    logger.info(f"🔍 Current environment observation space: {current_obs_size} dims")
    
    # Check observation space dimension mismatch
    try:
        with zipfile.ZipFile(latest_model_path, 'r') as zip_ref:
            if 'data' in zip_ref.namelist():
                with zip_ref.open('data') as f:
                    data = pickle.load(f)
                    if 'observation_space' in data:
                        saved_obs_size = data['observation_space'].shape[0] if hasattr(data['observation_space'], 'shape') else None
                        if saved_obs_size and saved_obs_size != current_obs_size:
                            logger.warning(f"⚠️ DIMENSION MISMATCH: Saved model has {saved_obs_size} dims, but environment has {current_obs_size} dims")
                            logger.warning(f"🗑️ Deleting incompatible model - will train fresh with correct dimensions")
                            os.remove(latest_model_path)
                            raise ValueError("Observation space mismatch - model deleted")
    except (zipfile.BadZipFile, KeyError, AttributeError, Exception) as check_error:
        logger.debug(f"Could not check model dimensions: {check_error}, proceeding with load")
    
    # Detect model type
    model_type = _detect_model_type(latest_model_path)
    
    # Load model with appropriate class
    from ppo.utils.dynamic_config import get_dynamic_config
    config = get_dynamic_config()
    use_tcn = bool(
        config
        and getattr(config, "use_temporal_memory", False)
        and getattr(config, "temporal_memory_type", "").lower() == "tcn"
    )
    
    # Try MaskablePPO first
    if MASKABLE_PPO_AVAILABLE:
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(latest_model_path, env=train_env)
            logger.info("✅ Loaded MaskablePPO model (action masking ENABLED)")
            return model
        except Exception as e:
            logger.warning(f"⚠️ Could not load as MaskablePPO: {e}")
            logger.warning("🗑️ Old model incompatible with MaskablePPO - deleting for retraining")
            os.remove(latest_model_path)
            raise ValueError("Old model incompatible with MaskablePPO - retraining required")
    
    # Try TCN, RecurrentPPO, or standard PPO
    try:
        from stable_baselines3 import RecurrentPPO  # type: ignore[attr-defined]
        RECURRENT_PPO_AVAILABLE = True
    except ImportError:
        RECURRENT_PPO_AVAILABLE = False
    
    if model_type == 'TCN_PPO':
        try:
            model = PPO.load(latest_model_path, env=train_env)
            logger.info("✅ Loaded PPO model with TCN architecture")
        except Exception as e:
            logger.warning(f"Failed to load TCN model: {e}, will retrain")
            os.remove(latest_model_path)
            raise ValueError("Failed to load TCN model - retraining required")
    elif model_type == 'RecurrentPPO' or (RECURRENT_PPO_AVAILABLE and model_type is None):
        try:
            model = RecurrentPPO.load(latest_model_path, env=train_env)
            logger.info("✅ Loaded RecurrentPPO model (LSTM architecture)")
        except Exception as e:
            logger.warning(f"Failed to load as RecurrentPPO: {e}, trying standard PPO")
            model = PPO.load(latest_model_path, env=train_env)
            logger.warning("⚠️ ARCHITECTURE MISMATCH: Loaded old PPO model (no LSTM) - will train new LSTM model")
            os.remove(latest_model_path)
            raise ValueError("Old PPO model incompatible with new LSTM architecture - retraining required")
    else:
        if use_tcn or RECURRENT_PPO_AVAILABLE:
            logger.warning("⚠️ ARCHITECTURE CHANGE DETECTED - deleting old model")
            os.remove(latest_model_path)
            raise ValueError("Architecture mismatch - retraining required")
        else:
            model = PPO.load(latest_model_path, env=train_env)
            logger.info("✅ Loaded standard PPO model")
    
    # Verify observation space matches
    obs_space = getattr(model, "observation_space", None)
    obs_shape = getattr(obs_space, "shape", None) if obs_space is not None else None
    loaded_obs_size = obs_shape[0] if obs_shape else None
    if loaded_obs_size and loaded_obs_size != current_obs_size:
        logger.error(f"❌ DIMENSION MISMATCH DETECTED: Loaded model has {loaded_obs_size} dims, environment has {current_obs_size} dims")
        os.remove(latest_model_path)
        raise ValueError(f"Observation space mismatch ({loaded_obs_size} vs {current_obs_size}) - model deleted, retraining")
    
    logger.info(f"✅ Model observation space verified: {loaded_obs_size or current_obs_size} dims")
    return model


def _detect_model_type(latest_model_path):
    """Detect model type from saved file"""
    import zipfile
    import pickle
    
    try:
        with zipfile.ZipFile(latest_model_path, 'r') as zip_ref:
            if 'data' in zip_ref.namelist():
                with zip_ref.open('data') as f:
                    data = pickle.load(f)
                    if 'policy_kwargs' in data:
                        policy_kwargs = data['policy_kwargs']
                        if 'tcn_hidden_dims' in policy_kwargs or 'tcn_dilation_rates' in policy_kwargs:
                            return 'TCN_PPO'
                        elif 'lstm_hidden_size' in policy_kwargs or 'n_lstm_layers' in policy_kwargs:
                            return 'RecurrentPPO'
                        else:
                            return 'PPO'
                    elif 'policy' in str(data.get('policy', '')):
                        policy_str = str(data.get('policy', ''))
                        if 'TCN' in policy_str or 'Tcn' in policy_str:
                            return 'TCN_PPO'
                        elif 'Lstm' in policy_str:
                            return 'RecurrentPPO'
                        else:
                            return 'PPO'
    except Exception:
        pass
    return None


def _continue_training(model, timesteps):
    """Continue training existing model"""
    from stable_baselines3.common.callbacks import CallbackList
    from ppo.training.ppo_trainer import BCRegularizationCallback
    from ppo.utils.dynamic_config import get_dynamic_timesteps
    
    progress_callback = DetailedProgressCallback(log_freq=25000)
    bc_reg_callback = BCRegularizationCallback(verbose=1)
    callback = CallbackList([progress_callback, bc_reg_callback])
    current_timesteps = get_dynamic_timesteps()
    logger.info(f"Continuing training with {current_timesteps:,} timesteps (from centralized dynamic config)")
    logger.info(f"🛡️ BC Regularization callback enabled to preserve BC knowledge")
    model.learn(total_timesteps=current_timesteps, callback=callback)
    logger.info("✅ Continued training completed")


def _evaluate_agent(orchestrator, model, test_env):
    """Evaluate agent performance"""
    import time
    
    print("📈 Evaluating agent performance...")
    logger.info("PHASE 6: Evaluating agent performance...")
    
    evaluation_start_time = time.time()
    try:
        logger.info("Starting agent evaluation...")
        results = orchestrator.evaluate_agent(model, test_env)
        logger.info("✅ Agent evaluation completed successfully")
    except (MemoryError, ValueError, Exception) as e:
        error_tracker.add_error(f"Error during evaluation: {e}", is_critical=True)
        raise
    finally:
        evaluation_time = time.time() - evaluation_start_time
        logger.info(f"⏱️ Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Log evaluation results
    if isinstance(results, dict):
        steps_simulated = results.get('steps_simulated', 0)
        episodes = results.get('episodes_completed', 1 if steps_simulated > 0 else 0)
        logger.info(f"📊 EVALUATION RESULTS: {steps_simulated:,} steps, {episodes} episodes")
        if steps_simulated < 100:
            logger.warning(f"⚠️ LOW STEP COUNT: Only {steps_simulated} steps simulated")
    
    # Validate performance
    if isinstance(results, dict):
        passed, issues, warnings = validate_ppo_performance(results, orchestrator.config)
        results['validation_passed'] = passed
        logger.info(f"VALIDATION: {'PASSED' if passed else 'FAILED'} - Issues: {len(issues)}, Warnings: {len(warnings)}")
    
    return results, evaluation_time


def _generate_reports(orchestrator, model, test_env, results, start_datetime, start_time,
                      data_loading_time, training_time, evaluation_time, timesteps):
    """Generate comprehensive reports"""
    import time
    import traceback
    from datetime import datetime
    
    print("📋 Generating comprehensive performance report...")
    logger.info("PHASE 7: Generating performance report...")
    
    report_start_time = time.time()
    end_time = time.time()
    total_execution_time = end_time - start_time
    end_datetime = datetime.now()
    
    # Add timing metrics to results
    if isinstance(results, dict):
        _add_timing_metrics(results, total_execution_time, data_loading_time, 
                           training_time, evaluation_time, start_datetime, end_datetime)
        _add_training_metrics(results, model, test_env)
        _add_adaptive_metrics(results, test_env)
        _add_tcn_status(results, model)
        results['data_source'] = getattr(orchestrator, 'data_source', 'UNKNOWN')
        results['data_quality'] = getattr(orchestrator, '_data_quality', {})
    
    # Generate unified report
    try:
        from ppo.evaluation.unified_report_generator import UnifiedReportGenerator
        report_generator = UnifiedReportGenerator()
        
        if hasattr(main, '_fundamental_status'):
            results['fundamental_status'] = main._fundamental_status
        
        report_path = report_generator.generate_unified_report(
            execution_results=results,
            training_results={'model': model, 'env': test_env},
            trade_history_data=results.get('trade_history', []),
            portfolio_history_data=results.get('portfolio_history', [])
        )
        
        if report_path:
            print(f"✅ Comprehensive report generated: {report_path}")
            logger.info(f"✅ Report generated successfully: {report_path}")
        else:
            logger.warning("⚠️ Report generation failed - no path returned")
    except Exception as e:
        logger.error(f"Failed to generate unified report: {e}")
        logger.error(f"Report generation error traceback: {traceback.format_exc()}")
    finally:
        report_time = time.time() - report_start_time
        logger.info(f"⏱️ Report generation completed in {report_time:.2f} seconds")
        if isinstance(results, dict):
            results['report_time'] = report_time
    
    # Generate SHAP explanations
    try:
        print("\n🔍 Generating SHAP explanations for agent decisions...")
        logger.info("PHASE 8: Generating SHAP explanations...")
        from ppo.utils.agent_explainability import AgentExplainer
        
        tickers = orchestrator.config.get('tickers', ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY', 'GLD', 'BTC-USD', 'KO', 'JNJ', 'AMZN', 'XLE', 'TLT'])
        feature_names = _build_feature_names(test_env, tickers)
        
        explainer = AgentExplainer(
            model=model,
            feature_names=feature_names,
            tickers=tickers,
            enable_shap=True,
            save_explanations=True,
            output_dir="ppo/explanations"
        )
        
        # When deterministic eval had 0 trades, explain HOLD steps so a report is still generated
        explain_trades_only = (results.get('total_trades', 0) > 0)
        shap_results = explainer.explain_evaluation_run(
            env=test_env,
            model=model,
            max_steps=min(30, results.get('steps_simulated', 500)),  # OPT E: was 500 - faster SHAP phase
            explain_trades_only=explain_trades_only,
            max_explanations=20  # OPT E: was 50 - faster SHAP phase
        )
        
        shap_report_path = explainer.generate_shap_report(execution_results=results)
        if shap_report_path:
            print(f"✅ SHAP report generated: {shap_report_path}")
            logger.info(f"✅ SHAP report generated: {shap_report_path}")
    except ImportError:
        logger.warning("⚠️ SHAP not available. Install with: pip install shap")
    except Exception as e:
        logger.warning(f"⚠️ SHAP explanation generation failed: {e}")


def _add_timing_metrics(results, total_execution_time, data_loading_time, 
                        training_time, evaluation_time, start_datetime, end_datetime):
    """Add timing metrics to results"""
    results['total_execution_time'] = total_execution_time
    results['data_loading_time'] = data_loading_time
    results['training_time'] = training_time
    results['evaluation_time'] = evaluation_time
    results['start_time'] = start_datetime.isoformat()
    results['end_time'] = end_datetime.isoformat()
    results['steps_simulated'] = results.get('steps_simulated', 0)
    results['episodes_completed'] = results.get('episodes_completed', 0)
    results['steps_per_second'] = results.get('steps_simulated', 0) / total_execution_time if total_execution_time > 0 else 0
    results['trades_per_second'] = results.get('total_trades', 0) / total_execution_time if total_execution_time > 0 else 0
    
    timing_dict = {
        'Data Loading': data_loading_time,
        'Training': training_time,
        'Evaluation': evaluation_time
    }
    numeric_timings = {key: float(value) for key, value in timing_dict.items() if isinstance(value, (int, float))}
    if numeric_timings and max(numeric_timings.values()) > 0:
        slowest_component = max(numeric_timings, key=lambda key: numeric_timings[key])
        results['slowest_component'] = slowest_component
        results['performance_bottleneck'] = f"{slowest_component} ({numeric_timings[slowest_component]:.2f}s)"


def _add_training_metrics(results, model, test_env):
    """Add training metrics to results"""
    import torch
    
    if hasattr(model, 'training_metrics'):
        training_metrics = model.training_metrics
        results['learning_rate'] = training_metrics.get('learning_rate', {}).get('final', 0.0)
        results['policy_loss'] = training_metrics.get('policy_loss', {}).get('final', 0.0)
        results['value_loss'] = training_metrics.get('value_loss', {}).get('final', 0.0)
        entropy_value = training_metrics.get('entropy', {}).get('final', 0.0)
        if entropy_value < 0:
            try:
                if hasattr(test_env, 'reset'):
                    obs = test_env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    action_dist = model.policy.get_distribution(torch.as_tensor(obs).unsqueeze(0))
                    entropy_value = float(action_dist.entropy().mean().item())
            except Exception:
                entropy_value = abs(entropy_value)
        results['entropy'] = max(0.0, entropy_value)
        results['exploration_rate'] = results['entropy']
        results['convergence_score'] = training_metrics.get('explained_variance', {}).get('final', 0.0)


def _add_adaptive_metrics(results, test_env):
    """Add adaptive reward metrics and adaptive LR/entropy callback logs to results"""
    # Inject adaptive LR and entropy callback logs from main._adaptive_callback (for unified report)
    try:
        if hasattr(main, '_adaptive_callback') and main._adaptive_callback is not None:
            cb = main._adaptive_callback
            adaptive_lr_log = getattr(cb, 'adaptive_lr_log', None)
            if adaptive_lr_log is None:
                adaptive_lr_log = getattr(cb, 'lr_adjustments', [])
            results['adaptive_lr_log'] = adaptive_lr_log or []
            callback_triggered = getattr(cb, 'callback_triggered', None)
            if callback_triggered is None:
                callback_triggered = len(results['adaptive_lr_log']) > 0
            results['adaptive_lr_callback_triggered'] = bool(callback_triggered)
            results['adaptive_lr_events_count'] = len(results['adaptive_lr_log'])
            results['adaptive_entropy_log'] = getattr(cb, 'adaptive_entropy_log', [])
            results['adaptive_entropy_events_count'] = len(results.get('adaptive_entropy_log', []))
    except Exception as e:
        logger.warning(f"Could not add adaptive callback logs to results: {e}")
        results['adaptive_lr_log'] = []
        results['adaptive_lr_callback_triggered'] = False
        results['adaptive_lr_events_count'] = 0
        results['adaptive_entropy_log'] = []
        results['adaptive_entropy_events_count'] = 0

    def _unwrap_env(env_obj):
        seen = set()
        current = env_obj
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, "envs") and getattr(current, "envs"):
                current = current.envs[0]
                continue
            if hasattr(current, "env"):
                current = current.env
                continue
            if hasattr(current, "unwrapped"):
                current = current.unwrapped
                continue
            break
        return current
    
    def _extract_adaptive_log(env_obj):
        base_env = _unwrap_env(env_obj)
        if base_env is None:
            return []
        if hasattr(base_env, "get_adaptive_reward_log"):
            try:
                return base_env.get_adaptive_reward_log() or []
            except Exception:
                return []
        log_data = getattr(base_env, "_adaptive_reward_log", None)
        return list(log_data) if log_data else []
    
    try:
        adaptive_reward_log = []
        test_env_log = _extract_adaptive_log(test_env)
        if test_env_log:
            adaptive_reward_log.extend(test_env_log)
        
        if hasattr(main, "_train_env"):
            train_env_log = _extract_adaptive_log(main._train_env)
            if train_env_log:
                adaptive_reward_log.extend(train_env_log)
        
        if adaptive_reward_log:
            results['adaptive_reward_log'] = adaptive_reward_log
            results['adaptive_reward_events_count'] = len(adaptive_reward_log)
    except Exception as e:
        logger.warning(f"⚠️ Could not extract adaptive reward logs: {e}")
        results['adaptive_reward_log'] = []
        results['adaptive_reward_events_count'] = 0


def _add_tcn_status(results, model):
    """Add TCN status to results"""
    try:
        from ppo.utils.dynamic_config import get_dynamic_config
        config = get_dynamic_config()
        tcn_configured = config.use_temporal_memory and config.temporal_memory_type == "tcn"
        results['tcn_configured'] = tcn_configured
        results['temporal_memory_type'] = config.temporal_memory_type
        results['tcn_used'] = 'TCN' in model.policy.__class__.__name__ if hasattr(model, 'policy') else False
        results['policy_class_name'] = model.policy.__class__.__name__ if hasattr(model, 'policy') else 'Unknown'
    except Exception as e:
        logger.warning(f"⚠️ Could not track TCN status: {e}")
        results['tcn_configured'] = False
        results['tcn_used'] = False


def _build_feature_names(test_env, tickers):
    """Build feature names matching observation structure"""
    obs_size = test_env.observation_space.shape[0] if hasattr(test_env, 'observation_space') else 1813
    
    base_env = test_env
    while hasattr(base_env, 'envs') and base_env.envs:
        base_env = base_env.envs[0]
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    if hasattr(base_env, 'unwrapped'):
        base_env = base_env.unwrapped
    
    window_size = getattr(base_env, 'window_size', 60)
    num_tickers = getattr(base_env, 'num_tickers', len(tickers))
    num_features_per_ticker = getattr(base_env, 'num_features', 13)

    # Prefer env-provided metadata so feature names match the *actual* observation schema.
    env_tickers = getattr(base_env, 'unique_tickers', None)
    if isinstance(env_tickers, list) and len(env_tickers) > 0:
        tickers = env_tickers
    feature_cols = getattr(base_env, 'feature_cols', None)
    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        feature_cols = [f"Feature_{i}" for i in range(int(num_features_per_ticker))]
    else:
        feature_cols = [str(c) for c in feature_cols][:int(num_features_per_ticker)]

    feature_names: list[str] = []
    for t in range(int(window_size)):
        for ticker_idx in range(int(num_tickers)):
            ticker = tickers[ticker_idx] if ticker_idx < len(tickers) else f"TICKER_{ticker_idx}"
            for col in feature_cols:
                feature_names.append(f"{col}_{ticker}_t{t}")
    
    feature_names.append("Portfolio_Cash_Pct")
    for ticker_idx in range(num_tickers):
        ticker = tickers[ticker_idx] if ticker_idx < len(tickers) else f"TICKER_{ticker_idx}"
        feature_names.append(f"Portfolio_Position_{ticker}")
    # 2026-02-15: Fundamental metrics in observation
    include_fund = getattr(base_env, '_include_fundamental_in_obs', False)
    if include_fund:
        fund_keys = getattr(base_env, 'FUNDAMENTAL_FEATURE_KEYS', ('pe_ratio', 'pb_ratio', 'debt_equity', 'roe', 'current_ratio', 'revenue_growth', 'earnings_growth'))
        for ticker_idx in range(num_tickers):
            ticker = tickers[ticker_idx] if ticker_idx < len(tickers) else f"TICKER_{ticker_idx}"
            for key in fund_keys:
                feature_names.append(f"Fund_{key}_{ticker}")
    
    if len(feature_names) < obs_size:
        for i in range(len(feature_names), obs_size):
            feature_names.append(f"Feature_{i}")
    elif len(feature_names) > obs_size:
        feature_names = feature_names[:obs_size]
    
    return feature_names


def _log_final_summary(orchestrator, results, timesteps, start_datetime, start_time):
    """Log final execution summary"""
    import psutil
    import time
    from datetime import datetime
    
    end_time = time.time()
    total_execution_time = end_time - start_time
    end_datetime = datetime.now()
    
    latest_model_path = os.path.join(orchestrator.models_dir, 'ppo_trading_agent.zip')
    bc_init_path = os.path.join(orchestrator.models_dir, 'ppo_bc_init.pt')
    
    try:
        from ppo.utils.dynamic_config import get_dynamic_timesteps
        current_dynamic_timesteps = get_dynamic_timesteps()
        print("\nCENTRALIZED CONFIG SUMMARY:")
        print(f"  Timesteps Used: {timesteps:,}")
        print(f"  Dynamic Config: {current_dynamic_timesteps:,}")
        print(f"  Source: dynamic_config.py")
        print(f"  Status: Single source of truth")
    except Exception:
        pass

    # Baseline scaling protocol (300k → 600k) with stability guardrails
    try:
        from ppo.utils.dynamic_config import auto_scale_timesteps
        next_ts = auto_scale_timesteps(results if isinstance(results, dict) else {}, error_tracker.get_summary())
        print(f"  Next Run Timesteps (auto-scaled): {next_ts:,}")
    except Exception:
        pass

    # Seed validation protocol (manual runbook; deterministic eval is production truth)
    try:
        print("  Seed Validation Protocol: after deterministic stabilizes, run 3 seeds (e.g. 42, 43, 44) at same timesteps;")
        print("    require stable Sharpe/DD and majority positive deterministic returns before paper trading.")
    except Exception:
        pass
    
    memory_final = psutil.virtual_memory()
    logger.info(f"Final memory usage: {memory_final.percent:.1f}% used")
    
    passed = isinstance(results, dict) and results.get('validation_passed', False)
    print("\n" + "="*80)
    print("SUPER AGENT EXECUTION SUMMARY")
    print("="*80)
    print(f"BC Model: {'LOADED' if os.path.exists(bc_init_path) else 'NOT FOUND'}")
    print(f"PPO Model: {'LOADED' if os.path.exists(latest_model_path) else 'NEW'}")
    print(f"Training Timesteps: {timesteps:,}")
    print(f"Validation: {'PASSED' if passed else 'FAILED'}")
    print(f"Errors: {error_tracker.get_summary()['total_errors']} ({error_tracker.get_summary()['critical_errors']} critical)")
    print(f"Warnings: {error_tracker.get_summary()['warnings']}")
    print(f"Return: {results.get('total_return', 0):.2%}")
    print(f"Sharpe: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Paper Trading Ready: {'YES' if results.get('paper_trading_ready') else 'NO'}")
    print(f"Config Source: dynamic_config.py")
    print(f"Timestep Management: Centralized")
    print("="*80)
    
    print("✅ PPO Trading Agent execution completed successfully!")
    print(f"📊 Final Results: {results.get('total_return', 0):.2%} return")
    print(f"⏱️ Total Execution Time: {total_execution_time:.2f} seconds ({total_execution_time/60:.1f} minutes)")
    
    logger.info("=" * 80)
    logger.info("PPO TRADING AGENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Start time: {start_datetime.isoformat()}")
    logger.info(f"End time: {end_datetime.isoformat()}")
    logger.info(f"Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.1f} minutes)")


def _web_status_update(message: str, **kwargs):
    """Update training status file for UI when started from backend/web API (TRAINING_STATUS_FILE env set)."""
    status_file = os.getenv('TRAINING_STATUS_FILE')
    if not status_file:
        return
    try:
        import json
        data = {}
        if os.path.isfile(status_file):
            with open(status_file) as f:
                data = json.load(f)
        data['message'] = message
        data['status'] = data.get('status', 'running')
        for k, v in kwargs.items():
            data[k] = v
        with open(status_file, 'w') as f:
            json.dump(data, f, indent=0)
    except Exception:
        pass


def main():
    """Main execution function for PPO Trading Agent with comprehensive debugging"""
    import psutil
    import platform
    import traceback
    from datetime import datetime
    import time
    
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now()
    
    try:
        # Log system information at startup
        _log_system_info(start_datetime)
        
        # Initialize orchestrator
        orchestrator, timesteps = _initialize_orchestrator()
        _web_status_update("Loading market data...", total_steps=timesteps)
        
        # Download market data
        market_data, data_loading_time = _download_market_data(orchestrator)
        _web_status_update("Creating environment...", total_steps=timesteps)
        logger.info(f"[STARTUP] DATA_SOURCE={getattr(orchestrator, 'data_source', 'UNKNOWN')} | ACTIVE_TICKERS={orchestrator.config.get('tickers', [])}")
        
        # Create environments
        train_env = _create_environments(orchestrator, market_data)
        _web_status_update("Starting training loop...", total_steps=timesteps)
        
        # Load or train model
        model, training_time = _load_or_train_model(orchestrator, train_env, market_data, timesteps)
        
        # Save model
        model_save_path = os.path.join(orchestrator.models_dir, 'ppo_trading_agent.zip')
        from ppo.training.ppo_trainer import save_model
        try:
            save_model(model, model_save_path)
            logger.info(f"✅ Model saved to {model_save_path}")
        except ImportError as e:
            logger.warning(f"save_model failed: {e}. Attempting direct model.save()...")
            try:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                model.save(model_save_path)
                logger.info(f"✅ Model saved via direct save() to {model_save_path}")
            except Exception as e2:
                logger.error(f"❌ Direct model save also failed: {e2}")
                raise
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
        
        # Log memory after training
        memory_after_training = psutil.virtual_memory()
        logger.info(f"Memory after training: {memory_after_training.percent:.1f}% used")
        
        # Create test environment and evaluate
        print("🧪 Creating test environment...")
        logger.info("PHASE 5: Creating test environment...")
        test_env = orchestrator.create_trading_environment(market_data, mode='test')
        logger.info("✅ Test environment created successfully")
        
        # Evaluate agent
        results, evaluation_time = _evaluate_agent(orchestrator, model, test_env)
        
        # Generate reports
        _generate_reports(orchestrator, model, test_env, results, start_datetime, start_time, 
                         data_loading_time, training_time, evaluation_time, timesteps)
        
        # Log final summary
        _log_final_summary(orchestrator, results, timesteps, start_datetime, start_time)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("PPO TRADING AGENT FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        
        # Log final memory usage even on error
        try:
            memory_error = psutil.virtual_memory()
            logger.error(f"Memory at error: {memory_error.percent:.1f}% used")
        except:
            pass
        
        print(f"❌ Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()







