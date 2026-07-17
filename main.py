import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import shutil
from datetime import datetime
from core import LSWMDDataset, CNNBaseline, E2CNN, collate_fn, init_weighted_random_sampler
from core import CostSensitiveLoss, get_cost_matrix
from params import params_e2cnn, params_cnn
import argparse
from torchmetrics.classification import MulticlassRecall, MulticlassConfusionMatrix


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in ('true', '1', 'yes', 'y'):
        return True
    if normalized in ('false', '0', 'no', 'n'):
        return False

    raise argparse.ArgumentTypeError('Expected a boolean value: true or false')


def train_and_test_model(params, save=True, model='cnn', data_balancing=False,
                         use_cost_sensitive_loss=True):
    """
    Train and test a CNN model on the LSWMD dataset using cross entropy loss.
    Logs metrics to TensorboardX and saves model periodically.
    
    Args:
        params (dict): Dictionary of model and training parameters
        save (bool): Whether to save the model checkpoints
        model (str): Type of model to use ('cnn' or 'e2cnn')
        data_balancing (bool): Whether to use data balancing for training
        use_cost_sensitive_loss (bool): Whether to switch to the cost-sensitive
            loss halfway through training
    """
    # Extract parameters from nested dict
    params['data']['data_balancing'] = data_balancing  # Update data balancing flag
    data_cfg = params['data']
    training_cfg = params['training']
    logging_cfg = params['logging']
    model_cfg = params['model']
    
    batch_size = training_cfg['batch_size']
    learning_rate = training_cfg['learning_rate']
    save_period = logging_cfg['save_period']
    log_period = logging_cfg['log_period']
    test_period = logging_cfg['test_period']
    
    # Create a unique log directory that records the training settings.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_base_dir = logging_cfg['log_dir']
    run_name = (
        f"{timestamp}_model-{model_cfg['type']}_balancing-{data_balancing}"
        f"_cost-sensitive-loss-{use_cost_sensitive_loss}"
    )
    log_dir = os.path.join(log_base_dir, run_name)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    code_dir = os.path.join(log_dir, 'code')
    
    if save and logging_cfg['save_checkpoints']:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if save:
        os.makedirs(code_dir, exist_ok=True)
        source_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in ('main.py', 'core.py', 'params.py'):
            shutil.copy2(os.path.join(source_dir, filename), code_dir)
    
    # Initialize TensorboardX writer
    if logging_cfg['tensorboard_enabled'] and save:
        writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
    else:
        writer = None
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create model based on type
    if model_cfg['type'] == 'e2cnn':
        params = params_e2cnn
        model = E2CNN(params).to(device)
        print("Using Rotation Invariant CNN (E2CNN)")
    else:
        params = params_cnn
        model = CNNBaseline(params).to(device)
        print("Using Baseline CNN")
    
    # Define loss function and optimizer
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = None
    if use_cost_sensitive_loss:
        cost_matrix = get_cost_matrix()
        criterion2 = CostSensitiveLoss(cost_matrix, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=training_cfg.get('l2_lambda', 0.0))

    def create_scheduler():
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=100, min_lr=training_cfg['min_lr']
        )

    scheduler = create_scheduler()
    metric = MulticlassRecall(num_classes=params['architecture']['fc']['num_classes'], average=None)

    # Load train and test datasets
    train_dataset = LSWMDDataset(
        data_path=data_cfg['data_path'],
        labeled=data_cfg['labeled'],
        split=data_cfg['train_split']
    )

    test_dataset = LSWMDDataset(
        data_path=data_cfg['data_path'],
        labeled=data_cfg['labeled'],
        split=data_cfg['test_split'],
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Checkpoints will be saved every {save_period} epoch(s)")
    print(f"Logs and checkpoints will be saved to: {log_dir}\n")
    
    # Create data loaders
    if data_cfg.get('data_balancing', False):
        train_sampler = init_weighted_random_sampler(train_dataset, 
                                                     num_classes=params['architecture']['fc']['num_classes'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    def evaluate_model(dataloader, criterion):
        was_training = model.training
        model.eval()
        metric.reset()

        total_loss = 0
        total_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                metric.update(predicted, labels)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * total_correct / total
        recall = metric.compute()

        if was_training:
            model.train()

        return avg_loss, accuracy, recall
    
    # Training loop
    print("Starting training and testing...")
    global_step = 0
    epoch = 0
    num_epochs = int(training_cfg['num_steps'] / len(train_loader)) + 1
    loss_transition_step = training_cfg['num_steps'] // 2
    scheduler_reset_for_cost_loss = False
    
    while global_step < training_cfg['num_steps']:
        # Training phase
        model.train()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            # Switch loss function (just once) based on global step
            if global_step >= loss_transition_step and not scheduler_reset_for_cost_loss:
                scheduler = create_scheduler()
                scheduler_reset_for_cost_loss = True

            if global_step < loss_transition_step or not use_cost_sensitive_loss:
                loss = criterion1(outputs, labels)
            else:
                # Use cost-sensitive loss only when the flag is set and after the transition step
                loss = criterion2(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == labels).sum().item() / labels.size(0) * 100
            
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                f"Train Loss: {loss.item():.4f}"
            )
        
            if logging_cfg['tensorboard_enabled'] and writer and global_step % log_period == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Accuracy/train', train_accuracy, global_step)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)
            
            do_test = global_step % test_period == 0
            if do_test:
                if global_step < loss_transition_step or not use_cost_sensitive_loss:
                    _, _, train_recall = evaluate_model(train_loader, criterion1)
                    test_batch_loss, test_accuracy, test_recall = evaluate_model(test_loader, criterion1)
                else:
                    _, _, train_recall = evaluate_model(train_loader, criterion2)
                    test_batch_loss, test_accuracy, test_recall = evaluate_model(test_loader, criterion2)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Test Loss: {test_batch_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
                    f"Train Recall: {[round(recall.item(), 4) for recall in train_recall]}, "
                    f"Test Recall: {[round(recall.item(), 4) for recall in test_recall]}"
                )
                scheduler.step(test_batch_loss)
                if logging_cfg['tensorboard_enabled'] and save:
                    writer.add_scalar('Loss/test', test_batch_loss, global_step)
                    writer.add_scalar('Accuracy/test', test_accuracy, global_step)
                    for class_idx, class_recall in enumerate(train_recall):
                        writer.add_scalar(f'Recall/train/class_{class_idx}', class_recall, global_step)
                    for class_idx, class_recall in enumerate(test_recall):
                        writer.add_scalar(f'Recall/test/class_{class_idx}', class_recall, global_step)

            # Save model periodically
            if save and logging_cfg['save_checkpoints'] and global_step % save_period == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{global_step+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Model saved to {checkpoint_path}")

            global_step += 1

        epoch += 1

    # End of training
    model.eval()
    cm_metric = MulticlassConfusionMatrix(num_classes=params['architecture']['fc']['num_classes']).to(device)
    with torch.no_grad():
        for input, labels in test_loader:
            input, labels = input.to(device), labels.to(device)
            logits = model(input)
            preds = logits.argmax(dim=1)
            cm_metric.update(preds, labels)
    confusion_matrix = cm_metric.compute().cpu().numpy()
    print("Confusion Matrix:\n", confusion_matrix)
    np.save(os.path.join(log_dir, 'confusion_matrix.npy'), confusion_matrix)
        
    if logging_cfg['tensorboard_enabled'] and writer:
        writer.close()
    print("Training and testing complete!")
    if logging_cfg['tensorboard_enabled']:
        print(f"TensorboardX logs saved to {os.path.join(log_dir, 'tensorboard')}")
    if logging_cfg['save_checkpoints']:
        print(f"Checkpoints saved to {checkpoint_dir}")
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', help='Whether to save model checkpoints and logs', type=str_to_bool, default=True)
    parser.add_argument('--model', help='Model type: cnn or e2cnn')
    parser.add_argument('--data-balancing', help='Whether to use data balancing for training', type=str_to_bool, default=False)
    parser.add_argument('--use-cost-sensitive-loss', help='Whether to use the customized loss function with the cost matrix', type=str_to_bool, default=True)
    args = parser.parse_args()
    if args.model == 'e2cnn':
        params = params_e2cnn
    else:
        params = params_cnn
    model = train_and_test_model(params=params, save=args.save, 
                                 model=args.model, data_balancing=args.data_balancing,
                                 use_cost_sensitive_loss=args.use_cost_sensitive_loss)
