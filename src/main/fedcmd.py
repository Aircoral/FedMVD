import argparse
import logging
import torch
import numpy as np
import json
from models.Model import Model
from data.bearing_signal_dataset import PU, BJTU, load_pu_data, load_bjtu_data
from data.distribution import iid, non_iid_class_skew, non_iid_quantity_skew, non_iid_noisy_skew
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from torch.utils.data import DataLoader, random_split

# 初始化配置
# 在算法文件开头的日志设置部分
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 如果没有处理程序，添加一个默认的控制台处理程序
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Federated Learning with FedAvg')
    
    # 参数分组
    data_group = parser.add_argument_group('Data')
    model_group = parser.add_argument_group('Model')
    train_group = parser.add_argument_group('Training')
    fl_group = parser.add_argument_group('Federated Learning')

    # 数据参数
    data_group.add_argument('--data_name', type=str, choices=['PU', 'BJTU'] ,default='BJTU', help='Dataset name')
    data_group.add_argument('--data_dir', type=str, default='/home/Lxr/Fed/dataset/BJTU', help='Dataset directory')
    data_group.add_argument('--batch_size', type=int, default=16, help='Batch size')
    data_group.add_argument('--distribution', type=str, choices=['iid', 'class', 'quantity', 'noisy'], default='noisy', help='Data distribution')
    # 模型参数
    model_group.add_argument('--num_class', type=int, default=5, help='Number of classes')
    model_group.add_argument('--lstm_out_dim', type=int, default=128, help='LSTM output dimension')
    model_group.add_argument('--resnet_out_dim', type=int, default=128, help='ResNet output dimension')
    
    # 训练参数
    train_group.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    train_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    train_group.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam', help='Optimizer')
    
    # 联邦学习参数
    fl_group.add_argument('--num_clients', type=int, default=4, help='Number of clients')
    fl_group.add_argument('--fed_rounds', type=int, default=5, help='Federated rounds')
    fl_group.add_argument('--local_epochs', type=int, default=10, help='Local epochs per client')
    fl_group.add_argument('--diff_ratio', type=float, default=0.25, help='Ratio of clients to optimize')
    fl_group.add_argument('--fnr_epochs', type=int, default=5, help='FNR epochs')
    return parser.parse_args()

# 特征钩子
features = {}

class FeatureHook:
    def __init__(self):
        self.features = {}
    
    def __call__(self, module, input, output):
        self.features.update({'input': input, 'output': output})


# 修改：函数重命名并修改参数和内部变量名
# def calculate_feature_norm_diff(feature_norm_all_clients, optimal_client_model_ids): 
def calculate_feature_variance_diff(feature_variance_all_clients, optimize_ids): # <--- 修改：重命名函数和参数
	"""计算客户端间的特征方差差异""" # Docstring 更新
	# 修改：变量重命名
	# class_norm_diffs = {client_id: {} for client_id in optimal_client_model_ids}
	class_variance_diffs = {client_id: {} for client_id in optimize_ids} # <--- 修改：重命名输出字典

	# for client_id in optimal_client_model_ids: # 原循环变量名
	for client_id in optimize_ids: # 使用传入的参数名
		# for other_id, norms in feature_norm_all_clients.items(): # 原变量名
		for other_id, variances in feature_variance_all_clients.items(): # <--- 修改：内部变量名 (variances)
			if other_id == client_id:
				continue
				
			# for class_label, norm in norms.items(): # 原变量名
			for class_label, variance in variances.items(): # <--- 修改：内部变量名 (variance)
				# if class_label in feature_norm_all_clients[client_id]: # 原变量名
				if class_label in feature_variance_all_clients[client_id]: # <--- 修改：内部变量名
					# diff = norm - feature_norm_all_clients[client_id][class_label] # 原计算
					# 计算方差差异
					diff = variance - feature_variance_all_clients[client_id][class_label] # <--- 修改：计算逻辑不变，变量更新
					
					# class_norm_diffs[client_id][class_label] = class_norm_diffs[client_id].get(class_label, 0) + diff # 原代码
					class_variance_diffs[client_id][class_label] = class_variance_diffs[client_id].get(class_label, 0) + diff # <--- 修改：字典名

		# 日志信息也应更新
		logger.info(f"Client {client_id} feature variance difference: {class_variance_diffs[client_id]}") # <--- 修改：日志内容

	# return class_norm_diffs # 原返回
	return class_variance_diffs # <--- 修改：返回修改后的字典

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

args = parse_args()
train_loader, val_loader, test_loader = get_dataloader(args)

if args.distribution == 'iid':
    split_train_loader_into_clients = iid
elif args.distribution == 'class':
    split_train_loader_into_clients = non_iid_class_skew
elif args.distribution == 'quantity':
    split_train_loader_into_clients = non_iid_quantity_skew
elif args.distribution == 'noisy':
    split_train_loader_into_clients = non_iid_noisy_skew
else:
    raise ValueError("Invalid data distribution")
client_data_loaders, ratio = split_train_loader_into_clients(train_loader, args.num_clients)

def train_fnr(id, model, args):
    """训练客户端模型并记录特征"""

    train_model(model, id, args)
# 

# 注意：这个函数现在计算和返回方差，而不是范数
def evaluate_model(model, data_loader, return_features=False):
	"""评估模型性能，并可选择返回特征方差""" # Docstring 更新
	model.to(device).eval()
	all_labels, all_preds = [], []
	# 修改：变量重命名，反映存储的是方差
	# class_norms = [[] for _ in range(args.num_class)] 
	class_variances = [[] for _ in range(args.num_class)] # <--- 修改：重命名列表

	feature_hook = FeatureHook() if return_features else None
	hook_handle = None # 用于之后移除钩子，防止内存泄漏
	if return_features:
		# 确保在正确的层注册钩子 (假设 fc1 仍然是目标层)
		# 注意：如果模型结构变化，可能需要修改这里的层名 'fc1'
		try:
			hook_handle = model.fc1.register_forward_hook(feature_hook) 
		except AttributeError:
			logger.error("无法在 model.fc1 注册钩子，请检查模型结构和层名。")
			return_features = False # 无法获取特征，则禁用后续计算

	with torch.no_grad():
		for data in data_loader:
			time_data, freq_data, ft_data = data['time'].to(device), data['freq'].to(device), data['cwt'].to(device)
			labels = data['label'].to(device).long()

			logits = model(time_data, freq_data, ft_data)
			preds = torch.argmax(logits, dim=1)

			all_labels.append(labels.cpu().numpy())
			all_preds.append(preds.cpu().numpy())

			# 修改：计算方差而不是范数
			if return_features and hook_handle: # 确保钩子已成功注册
				# feature_norms = torch.norm(feature_hook.features['output'], p=2, dim=-1).cpu().numpy() # 原代码
				# 计算方差 (二阶中心距)
				# dim=-1 表示沿着最后一个维度（特征维度）计算
				# unbiased=True 使用样本方差 (N-1 分母)
				feature_variances = torch.var(feature_hook.features['output'], dim=-1, unbiased=True).cpu().numpy() # <--- 修改：核心计算
				for i, label in enumerate(labels):
					# class_norms[label.item()].append(feature_norms[i]) # 原代码
					class_variances[label.item()].append(feature_variances[i]) # <--- 修改：使用新变量名存储

	# 及时移除钩子
	if hook_handle:
		hook_handle.remove()

	metrics = {
		'accuracy': accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
	}

	# 修改：返回方差信息
	if return_features:
		# metrics['feature_norms'] = { # 原代码
		# 	 i: np.mean(norms) for i, norms in enumerate(class_norms) if norms 
		# }
		metrics['feature_variances'] = { # <--- 修改：字典键名
			# 使用修改后的列表名和变量名计算平均方差
			i: np.mean(variances) for i, variances in enumerate(class_variances) if variances # <--- 修改：计算平均方差
		}
	
	return metrics

def global_metric(model, args):
    """评估全局模型性能"""
    model.to(device).eval()
    all_labels, all_preds = [], []
    
    for data in test_loader:
        time_data, freq_data, ft_data = data['time'].to(device), data['freq'].to(device), data['cwt'].to(device)
        labels = data['label'].to(device).long()

        
        logits = model(time_data, freq_data, ft_data)
        preds = torch.argmax(logits, dim=1)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    
    acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
    pre = precision_score(np.concatenate(all_labels), np.concatenate(all_preds), average='macro')
    rec = recall_score(np.concatenate(all_labels), np.concatenate(all_preds), average='macro')
    f1 = 2 * pre * rec / (pre + rec)
    
    distribution = args.distribution
    
    # 保存预测结果
    with open(f'/home/Lxr/Fed/src/main/{distribution}_preds_label.json', 'w') as f:
        json.dump({
            'preds': np.concatenate(all_preds).tolist(),
            'labels': np.concatenate(all_labels).tolist()
        }, f)
    
    return {
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1
    }

def train_model(model, client_id, args):
    """训练客户端模型"""
    model.to(device).train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    loader = client_data_loaders[client_id]
    logger.info(f"Training client {client_id}")
    
    for epoch in range(args.local_epochs):
        total_loss = 0.0
        for data in loader:
            time_data, freq_data, ft_data = data['time'].to(device), data['freq'].to(device), data['cwt'].to(device)
            labels = data['label'].to(device).long()
            
            optimizer.zero_grad()
            logits = model(time_data, freq_data, ft_data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        logger.info(f"Client {client_id} Epoch {epoch}: Loss {total_loss/len(loader):.4f}")

# 修改：修改传入的参数名 diffs -> variance_diffs
# def update_worst_model(client_models, client_ids, diffs, args): 
def update_worst_model(client_models, client_ids, variance_diffs, args): # <--- 修改：重命名参数 variance_diffs
	"""更新表现最差的客户端模型 (使用方差差异正则化)""" # Docstring 更新
	criterion = torch.nn.CrossEntropyLoss().to(device)
	
	for client_id in client_ids:
		model = client_models[client_id]
		model.to(device).train()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
		
		logger.info(f"Updating client {client_id} using variance regularization") # <--- 修改：日志信息

		for epoch in range(args.fnr_epochs):
			total_loss, total_acc = 0.0, 0.0
			
			# 注意：这里仍然使用 val_loader 进行更新，这是一个设计选择
			# 也可以考虑使用该客户端自己的数据加载器 client_data_loaders[client_id]
			for data in val_loader: 
				time_data, freq_data, ft_data = data['time'].to(device), data['freq'].to(device), data['cwt'].to(device)
				labels = data['label'].to(device).long()
				
				# 计算基础损失
				logits = model(time_data, freq_data, ft_data)
				loss = criterion(logits, labels)
				
				# 添加特征方差差异正则化
				for class_label in labels.unique(): # 遍历批次中唯一的标签
					class_label_item = class_label.item() # 从Tensor获取Python int作为字典键
					# 修改：使用 variance_diffs 变量
					# if class_label in diffs[client_id]: # 原代码
					if class_label_item in variance_diffs[client_id]: # <--- 修改：使用新参数名和 .item()
						class_weight = (labels == class_label).sum().item() / len(labels) # 计算该类在批次中的权重
						# loss += 0.1 * class_weight * diffs[client_id][class_label] # 原代码
						# 使用方差差异进行正则化，注意正则化系数 0.1 可调
						loss += 0.1 * class_weight * variance_diffs[client_id][class_label_item] # <--- 修改：正则化项计算

				# 更新模型
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				# 计算准确率 (可选，用于监控更新过程)
				preds = torch.argmax(logits.detach(), dim=1) # 使用 detach() 以免影响梯度
				acc = (preds == labels).sum().item() / len(labels)
				
				total_loss += loss.item() # 记录包含正则化项的总损失
				total_acc += acc
				
			logger.info(f"Client {client_id} Update Epoch {epoch}: Loss {total_loss/len(val_loader):.4f} | Acc {total_acc/len(val_loader):.4f}")
			
		model.to('cpu') # 训练/更新完成后移回 CPU

def aggregate_models(client_models, client_loaders):
    """聚合客户端模型参数"""
    total_samples = sum(len(loader.dataset) for loader in client_loaders)
    global_params = {k: torch.zeros_like(v) for k, v in client_models[0].state_dict().items()}
    
    for model, loader in zip(client_models, client_loaders):
        weight = len(loader.dataset) / total_samples
        for k, v in model.state_dict().items():
            if global_params[k].dtype == torch.long:
                global_params[k] += (v.float() * weight).long()
            else:
                global_params[k] += v.float() * weight
    
    return global_params

def local_train_fnr(client_models, global_model, args):
	"""本地训练和模型更新 (使用方差)""" # Docstring 更新
	# 训练所有客户端模型
	client_metrics = {}
	# 修改：变量重命名
	# feature_norms = {} 
	feature_variances = {} # <--- 修改：重命名

	for client_id, model in enumerate(client_models):
		# train_fnr 内部调用 train_model，train_model 本身不受影响
		train_fnr(client_id, model, args) 
		# 调用修改后的 evaluate_model 获取方差信息
		metrics = evaluate_model(model, val_loader, return_features=True) 
		client_metrics[client_id] = metrics['accuracy']
		# 修改：使用新的键名和变量名存储方差
		# feature_norms[client_id] = metrics['feature_norms'] 
		if 'feature_variances' in metrics: # 检查键是否存在 (如果钩子注册失败则不存在)
			feature_variances[client_id] = metrics['feature_variances'] # <--- 修改：使用新键名
		else:
			feature_variances[client_id] = {} # 如果没有获取到方差，则设为空字典
		
		model.to('cpu')
		# 可选：在日志中添加平均方差信息
		avg_var_str = f"{np.mean(list(feature_variances[client_id].values())) if feature_variances[client_id] else 'N/A':.4f}"
		logger.info(f"Client {client_id} | Val Acc: {metrics['accuracy']:.4f} | Avg Feature Var: {avg_var_str}")


	# 选择需要优化的客户端 (逻辑不变)
	num_optimize = int(args.diff_ratio * args.num_clients)
	# 确保有客户端需要优化且已成功获取方差数据
	if num_optimize > 0 and any(feature_variances.values()):
		optimize_ids = sorted(client_metrics, key=client_metrics.get)[:num_optimize]
		
		# 计算特征方差差异并更新模型
		logger.info(f"Calculating variance differences for clients: {optimize_ids}")
		# 修改：调用重命名后的函数，传入重命名后的变量
		# diffs = calculate_feature_norm_diff(feature_norms, optimize_ids) # 原代码
		variance_diffs = calculate_feature_variance_diff(feature_variances, optimize_ids) # <--- 修改：调用新函数
		
		# 修改：调用修改后的函数，传入重命名后的变量
		# update_worst_model(client_models, optimize_ids, diffs, args) # 原代码
		update_worst_model(client_models, optimize_ids, variance_diffs, args) # <--- 修改：调用新函数
	else:
		logger.info("Skipping worst model update (num_optimize=0 or no variance data).")

	# 聚合和分发模型 (逻辑不变)
	# 聚合所有客户端模型（包括被更新和未被更新的）
	global_params = aggregate_models(client_models, client_data_loaders) 
	# 将聚合后的参数分发给所有客户端模型和全局模型
	for model in client_models: 
		model.load_state_dict(global_params)
	global_model.load_state_dict(global_params) 
	global_model.eval() # 设置全局模型为评估模式

	# 评估全局模型 (逻辑不变)
	metrics = global_metric(global_model, args)
	logger.info(f"Global Test | Acc: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
def global_train_fnr(args):
    """全局联邦训练"""
    client_models = [Model(args) for _ in range(args.num_clients)]
    global_model = Model(args)
    
    for round in range(args.fed_rounds):
        logger.info(f"\n=== Round {round + 1}/{args.fed_rounds} ===")
        local_train_fnr(client_models, global_model, args)

def main():
    """主函数"""
    args = parse_args()
    global_train_fnr(args)

if __name__ == '__main__':
    main()
