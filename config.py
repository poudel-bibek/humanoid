"""
Configuration file containing all hyperparameters for training and evaluation
"""

config = {
    # Training hyperparameters
    'num_iterations': 1000,
    'horizon': 2048,
    'gamma': 0.99,
    'clip_range': 0.2,
    
    # Network architectures
    'policy_hidden_size': 256,
    'value_hidden_size': 256,
    'transformer_sequence_length': 32,
    'transformer_nhead': 8,
    'transformer_num_layers': 3,
    'transformer_dim_feedforward': 512,
    
    # Optimization
    'policy_lr': 1e-4,
    'value_lr': 5e-4,
    'ppo_epochs': 4,
    'batch_size': 64,
    
    # Environment parameters
    'forward_reward_weight': 2.0,
    'ctrl_cost_weight': 0.005,
    'healthy_reward': 1.5,
    
    # Evaluation parameters
    'eval_every': 10,  # Evaluate every N iterations
    'eval_episodes': 3,  # Number of episodes for evaluation
    'eval_horizon': 500,  # Steps per evaluation episode
    
    # Saving and logging
    'models_dir': "./models",
    'plots_dir': "./plots",
    'save_every': 50,  # Save checkpoints every N iterations
} 