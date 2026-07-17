"""
Centralized configuration for wafer map defect classification model.

This module organizes all parameters by topic for easy configuration:
- Data: Dataset paths and loading parameters
- Model Architecture: CNN layer configurations
- Training: Optimization and training loop parameters
- Logging: Model checkpointing and logging parameters
"""

params_cnn = {
    # ============================================================================
    # DATA CONFIGURATION
    # ============================================================================
    'data': {
        'data_path': 'data/LSWMD.pkl',
        'train_split': 'train',
        'test_split': 'test',
        'labeled': True,
        'data_balancing': False,  # Whether to use data balancing for training
    },
    
    # ============================================================================
    # MODEL ARCHITECTURE - NN Configuration
    # ============================================================================
    'architecture': {
        # Convolutional layers configuration
        'conv1': {
            'out_channels': 16,
            'kernel_size': 3,
            'padding': 1,
        },
        'conv2': {
            'out_channels': 32,
            'kernel_size': 3,
            'padding': 1,
        },
        'conv3': {
            'out_channels': 64,
            'kernel_size': 3,
            'padding': 1,
        },
        
        # Fully connected layers configuration
        'fc': {
            'fc1_hidden': 128,
            'fc2_hidden': 64,
            'num_classes': 9,
        },
        
        # Common layer parameters
        'dropout_rate': 0.5,
        'pool_kernel_size': 2,
        'pool_stride': 2,
    },
    
    # ============================================================================
    # MODEL SELECTION
    # ============================================================================
    'model': {
        'type': 'baseline',  # 'baseline' or 'e2cnn'
        'e2cnn': {
            'rotations': 4,  # C4 group (4-fold rotations)
        },
    },
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    'training': {
        'num_steps': 8e3,  # Total number of training steps
        'batch_size': 512,
        'learning_rate': 0.0012,
        'min_lr': 5e-6,  # Minimum learning rate for scheduler
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'l2_lambda': 4e-4,  # L2 regularization strength (optional)
    },
    
    # ============================================================================
    # LOGGING AND CHECKPOINTING
    # ============================================================================
    'logging': {
        'save_checkpoints': True,
        'save_period': 1000,  # Save model every N steps
        'log_period': 1,  # Log metrics every N steps
        'test_period': 10,  # Test metrics every N steps
        'log_dir': 'results',
        'tensorboard_enabled': True,
    },
}

params_e2cnn = {
    # ============================================================================
    # DATA CONFIGURATION
    # ============================================================================
    'data': {
        'data_path': 'data/LSWMD.pkl',
        'train_split': 'train',
        'test_split': 'test',
        'labeled': True,
        'data_balancing': False,  # Whether to use data balancing for training
    },
    
    # ============================================================================
    # MODEL ARCHITECTURE - NN Configuration
    # ============================================================================
    'architecture': {
        # Convolutional layers configuration
        'conv1': {
            'out_channels': 16,
            'kernel_size': 3,
            'padding': 1,
        },
        'conv2': {
            'out_channels': 32,
            'kernel_size': 3,
            'padding': 1,
        },
        'conv3': {
            'out_channels': 64,
            'kernel_size': 3,
            'padding': 1,
        },
        
        # Fully connected layers configuration
        'fc': {
            'fc1_hidden': 128,
            'fc2_hidden': 64,
            'num_classes': 9,
        },
        
        # Common layer parameters
        'dropout_rate': 0.5,
        'pool_kernel_size': 2,
        'pool_stride': 2,
    },
    
    # ============================================================================
    # MODEL SELECTION
    # ============================================================================
    'model': {
        'type': 'e2cnn',  # 'baseline' or 'e2cnn'
        'e2cnn': {
            'rotations': 4,  # C4 group (4-fold rotations)
        },
    },
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    'training': {
        'num_steps': 8e3,  # Total number of training steps
        'batch_size': 512,
        'learning_rate': 0.0012,
        'min_lr': 5e-6,  # Minimum learning rate for scheduler
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'l2_lambda': 8e-4,  # L2 regularization strength (optional)
    },
    
    # ============================================================================
    # LOGGING AND CHECKPOINTING
    # ============================================================================
    'logging': {
        'save_checkpoints': True,
        'save_period': 1000,  # Save model every N steps
        'log_period': 1,  # Log metrics every N steps
        'test_period': 10,  # Test metrics every N steps
        'log_dir': 'results',
        'tensorboard_enabled': True,
    },
}
