import os
import sys
import logging
import importlib
from datetime import datetime
import contextlib
from io import StringIO

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入必需的类
from data.bearing_signal_dataset import PU, BJTU

class LogCapture:
    def __init__(self, target_logger):
        self.target_logger = target_logger
        self.captured_logs = StringIO()
        self.handler = logging.StreamHandler(self.captured_logs)
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    def __enter__(self):
        # 为目标 logger 添加临时处理程序
        self.target_logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 移除临时处理程序
        self.target_logger.removeHandler(self.handler)
        logs = self.captured_logs.getvalue()
        self.captured_logs.close()
        if logs:
            self.target_logger.info("Captured logs:\n" + logs)

def setup_logging(algorithm, dataset, distribution):
    """设置分层日志结构"""
    # 创建基础日志目录
    base_log_dir = 'logs'
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)
    
    # 创建算法特定的目录
    algorithm_dir = os.path.join(base_log_dir, algorithm)
    if not os.path.exists(algorithm_dir):
        os.makedirs(algorithm_dir)
    
    # 创建数据集特定的目录
    dataset_dir = os.path.join(algorithm_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # 创建日志文件（使用分布类型和时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(dataset_dir, f'{distribution}_{timestamp}.log')
    
    # 创建logger
    logger = logging.getLogger(f"{algorithm}_{dataset}_{distribution}")
    logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以捕获所有日志
    
    # 清除现有的处理程序（如果有）
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件处理程序
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 改为DEBUG级别以显示所有日志
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 确保 logger 不会传播到父级
    logger.propagate = False
    
    return logger

def run_single_experiment(algorithm_name, dataset_name, distribution_name, logger):
    """运行单个实验的具体实现"""
    # 保存原始的sys.argv和模块字典
    original_argv = sys.argv.copy()
    original_modules = dict(sys.modules)
    
    # 设置新的命令行参数
    sys.argv = [
        f'{algorithm_name}.py',
        f'--data_name={dataset_name}',
        f'--distribution={distribution_name}',
        '--num_clients=4',
        '--fed_rounds=5',
        '--local_epochs=10'
    ]
    
    try:
        # 确保数据集类在全局命名空间中可用
        globals()['PU'] = PU
        globals()['BJTU'] = BJTU
        
        # 记录实验开始
        logger.debug("Debug test message")
        logger.info(f"Starting experiment with parameters:")
        logger.info(f"Algorithm: {algorithm_name}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Distribution: {distribution_name}")
        
        # 导入并运行算法模块
        algorithm_module = importlib.import_module(algorithm_name)
        
        # 确保算法模块使用传入的 logger
        if hasattr(algorithm_module, 'logger'):
            algorithm_module.logger = logger
        
        # 设置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 使用日志捕获器运行算法
        with LogCapture(logger):
            algorithm_module.main()
            
        logger.info("Test log after experiment")
        return True
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        raise e
    finally:
        # 恢复原始状态
        sys.argv = original_argv
        sys.modules = original_modules

def run_experiment(algorithm, dataset, distribution):
    """运行单个实验"""
    # 设置该实验的logger
    logger = setup_logging(algorithm, dataset, distribution)
    logger.debug("Debug test message")
    logger.info("Info test message")
    
    try:
        # 确保在正确的目录下运行
        original_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        success = run_single_experiment(algorithm, dataset, distribution, logger)
        if success:
            logger.info(f"Experiment completed successfully")
        
        # 检查日志文件
        log_file = logger.handlers[0].baseFilename
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                if not content:
                    logger.warning("Log file is empty!")
                else:
                    logger.info(f"Log file size: {os.path.getsize(log_file)} bytes")
                    
    except Exception as e:
        logger.error(f"Error in {algorithm}: {str(e)}")
    finally:
        # 恢复原始工作目录
        os.chdir(original_dir)

def main():
    """主函数"""
    # 实验参数
    algorithms = [
        'fedavg', 
        # 'fedfnr', 
        # 'fedmoon', 
        'fednova', 
        'fedprox', 
        'fedscaffold'
    ]
    datasets = ['PU', 'BJTU']
    distributions = ['iid', 'class', 'quantity', 'noisy']
    
    # 创建主日志目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 计算总实验数
    total_experiments = len(algorithms) * len(datasets) * len(distributions)
    current_experiment = 0
    
    # 设置主日志
    main_logger = setup_logging('main', 'all', 'summary')
    main_logger.info("Starting all experiments")
    
    for algorithm in algorithms:
        for dataset in datasets:
            for distribution in distributions:
                current_experiment += 1
                main_logger.info(f"Progress: {current_experiment}/{total_experiments}")
                main_logger.info(f"Running experiment: {algorithm} with {dataset} dataset and {distribution} distribution")
                
                run_experiment(algorithm, dataset, distribution)
                
                main_logger.info(f"Completed experiment: {algorithm} - {dataset} - {distribution}")
                main_logger.info("-" * 80)
    
    main_logger.info("All experiments completed")

if __name__ == '__main__':
    main()
