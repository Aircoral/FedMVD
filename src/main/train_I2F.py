import argparse
import logging

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data.bearing_signal_dataset import PU, BJTU
from models.Model import Model
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def parse_args():
    
    parser = argparse.ArgumentParser(description='Training arguments for baseline models')
    
    # Data
    parser.add_argument('--data_path', type=str, default='/home/Lxr/Fed/dataset', help='Path to dataset folder')
    parser.add_argument('--data_name', type=str, choices=['PU', 'BJTU'] , default='PU', help='Name of dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    # Model
    parser.add_argument('--model_name', type=str, choices=['WDCNN', 'TCNN', 'QCNN', 'MA1DCNN'], default='MA1DCNN', help='Name of model')
    parser.add_argument('--in_channel', type=int, default=1, help='Number of input channels')
    parser.add_argument('--out_channel', type=int, default=5, help='Number of output channels')
    parser.add_argument('--num_class', type=int, default=5, help='Number of classes')
    parser.add_argument('--lstm_out_dim', type=int, default=128, help='LSTM output dimension')
    parser.add_argument('--resnet_out_dim', type=int, default=128, help='ResNet output dimension')

    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    
    args = parser.parse_args()
    return args

def get_dataloader(args):
    """加载并划分数据集"""
    if args.data_name == 'PU':
        dataset = torch.load('/home/Lxr/Fed/dataset/PU_dataset.pth')
    if args.data_name == 'BJTU':
        dataset = torch.load('/home/Lxr/Fed/dataset/BJTU_dataset.pth')
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    return (
        DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=args.batch_size),
        DataLoader(test_set, batch_size=args.batch_size)
    )
    
def metrics(model, loader, args):
    """计算模型评估指标"""
    
    logger.info('Evaluating model...')
    
    model.eval()
    model.to(args.device)
    all_labels = []
    all_preds = []
    for batch_idx, data in enumerate(loader):
        time, freq, cwt, labels = data['time'].to(args.device), data['freq'].to(args.device), data['cwt'].to(args.device), data['label']
        labels = labels.to(args.device).long()
            
        outputs = model(time, freq, cwt)
        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = 2 * pre * rec / (pre + rec)
    return {
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1
    }
        
def train(model, loader, args):
    """训练模型"""
    logger.info('Training model...')
    model.to(args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        avg_loss = 0
        num_batches = 0
        for batch_idx, data in enumerate(loader):
            time, freq, cwt, labels = data['time'].to(args.device), data['freq'].to(args.device), data['cwt'].to(args.device), data['label']
            labels = labels.to(args.device).long()
            
            outputs = model(time, freq, cwt)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            num_batches += 1

        avg_loss /= num_batches
        logger.info(f'Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}')
    

def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    
    train_loader, val_loader, test_loader = get_dataloader(args)
    model = Model(args)
    train(model, train_loader, args)
    metrics_test = metrics(model, test_loader, args)
    
    logger.info(f"Dataset {args.data_name}: Model {Model} Test | Acc: {metrics_test['accuracy']:.4f} | Precision: {metrics_test['precision']:.4f} | Recall: {metrics_test['recall']:.4f} | F1: {metrics_test['f1']:.4f}")
    print(f"Dataset {args.data_name}: Model {Model} Test | Acc: {metrics_test['accuracy']:.4f} | Precision: {metrics_test['precision']:.4f} | Recall: {metrics_test['recall']:.4f} | F1: {metrics_test['f1']:.4f}")
if __name__ == '__main__':
    main()