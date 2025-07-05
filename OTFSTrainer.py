class OTFSTrainer:
    """Train"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 多任务损失函数
        self.gain_criterion = nn.MSELoss()
        self.delay_criterion = nn.MSELoss()
        self.aoa_criterion = nn.MSELoss()
        self.path_criterion = nn.BCELoss()
        self.sync_criterion = nn.MSELoss()
           
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        
        # 损失权重
        self.loss_weights = {
            'gains': 1.0,
            'delays': 1.0,
            'aoas': 0.5,
            'paths': 0.8,
            'sync': 1.5
        }
        
        # 训练历史
        self.train_history = []
        self.val_history = []
    
    def compute_loss(self, predictions, targets):
        """计算多任务损失"""
        gains_loss = self.gain_criterion(predictions['gains'], targets['gains'])
        delays_loss = self.delay_criterion(predictions['delays'], targets['delays'])
        aoas_loss = self.aoa_criterion(predictions['aoas'], targets['aoas'])
        paths_loss = self.path_criterion(predictions['paths'], targets['paths'])
        sync_loss = self.sync_criterion(predictions['sync'], targets['sync'])
        
        # 加权总损失
        total_loss = (self.loss_weights['gains'] * gains_loss +
                     self.loss_weights['delays'] * delays_loss +
                     self.loss_weights['aoas'] * aoas_loss +
                     self.loss_weights['paths'] * paths_loss +
                     self.loss_weights['sync'] * sync_loss)
        
        return total_loss, {
            'gains': gains_loss.item(),
            'delays': delays_loss.item(),
            'aoas': aoas_loss.item(),
            'paths': paths_loss.item(),
            'sync': sync_loss.item(),
            'total': total_loss.item()
        }
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {'gains': 0, 'delays': 0, 'aoas': 0, 'paths': 0, 'sync': 0}
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # 移到设备
            dd_features = features[0].to(self.device)
            dt_features = features[1].to(self.device)
            
            batch_targets = {}
            for key, value in targets.items():
                batch_targets[key] = value.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(dd_features, dt_features)
            
            # 计算损失
            loss, loss_dict = self.compute_loss(predictions, batch_targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累积损失
            total_loss += loss_dict['total']
            for key in loss_components:
                loss_components[key] += loss_dict[key]
        
        # 平均损失
        avg_loss = total_loss / len(train_loader)
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        return avg_loss, loss_components
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        loss_components = {'gains': 0, 'delays': 0, 'aoas': 0, 'paths': 0, 'sync': 0}
        
        with torch.no_grad():
            for features, targets in val_loader:
                dd_features = features[0].to(self.device)
                dt_features = features[1].to(self.device)
                
                batch_targets = {}
                for key, value in targets.items():
                    batch_targets[key] = value.to(self.device)
                
                predictions = self.model(dd_features, dt_features)
                loss, loss_dict = self.compute_loss(predictions, batch_targets)
                
                total_loss += loss_dict['total']
                for key in loss_components:
                    loss_components[key] += loss_dict[key]
        
        avg_loss = total_loss / len(val_loader)
        for key in loss_components:
            loss_components[key] /= len(val_loader)
        
        return avg_loss, loss_components
    
    def train(self, train_loader, val_loader, num_epochs=50):
        """训练模型"""
        print("开始训练...")
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_components = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_components = self.validate(val_loader)
            
            # 记录历史
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            
            # 学习率调度
            self.scheduler.step()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_optimized_otfs_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'  Train Loss: {train_loss:.6f}')
                print(f'    Gains: {train_components["gains"]:.6f}, Delays: {train_components["delays"]:.6f}')
                print(f'    AoAs: {train_components["aoas"]:.6f}, Paths: {train_components["paths"]:.6f}')
                print(f'    Sync: {train_components["sync"]:.6f}')
                print(f'  Val Loss: {val_loss:.6f}')
                print(f'  LR: {self.scheduler.get_last_lr()[0]:.8f}')
                print(f'  Patience: {patience_counter}/{patience}')
                print('-' * 60)
            
            if patience_counter >= patience:
                print("早停触发！")
                break
        
        print("训练完成！")
    
    def evaluate_model(self, test_loader):
        """评估模型性能"""
        print("开始评估模型...")
        
        # 加载最佳模型
        try:
            self.model.load_state_dict(torch.load('best_optimized_otfs_model.pth', map_location=self.device))
            print("已加载最佳模型")
        except FileNotFoundError:
            print("未找到保存的模型，使用当前模型进行评估")
        
        self.model.eval()
        
        # 收集预测结果和真实标签
        all_predictions = {
            'gains': [],
            'delays': [],
            'aoas': [],
            'paths': [],
            'sync': []
        }
        
        all_targets = {
            'gains': [],
            'delays': [],
            'aoas': [],
            'paths': [],
            'sync': []
        }
        
        with torch.no_grad():
            for features, targets in test_loader:
                dd_features = features[0].to(self.device)
                dt_features = features[1].to(self.device)
                
                # 预测
                predictions = self.model(dd_features, dt_features)
                
                # 收集结果
                for key in all_predictions.keys():
                    all_predictions[key].append(predictions[key].cpu().numpy())
                    all_targets[key].append(targets[key].numpy())
        
        # 合并所有批次的结果
        for key in all_predictions.keys():
            all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
            all_targets[key] = np.concatenate(all_targets[key], axis=0)
        
        # 计算性能指标
        results = self._compute_metrics(all_predictions, all_targets)
        
        # 打印评估结果
        self._print_evaluation_results(results)
        
        return results, all_predictions, all_targets
    
    def _compute_metrics(self, predictions, targets):
        """计算各种性能指标"""
        results = {}
        
        # 信道参数MSE (增益 + 延迟 + 到达角)
        channel_pred = np.concatenate([
            predictions['gains'],
            predictions['delays'],
            predictions['aoas']
        ], axis=1)
        
        channel_target = np.concatenate([
            targets['gains'],
            targets['delays'],
            targets['aoas']
        ], axis=1)
        
        results['channel_mse'] = np.mean((channel_pred - channel_target) ** 2)
        
        # 同步参数MSE
        results['sync_mse'] = np.mean((predictions['sync'] - targets['sync']) ** 2)
        
        # 定时偏移MAE (反归一化)
        timing_pred = predictions['sync'][:, 0] * 10.0  # 反归一化
        timing_target = targets['sync'][:, 0] * 10.0
        results['timing_mae'] = np.mean(np.abs(timing_pred - timing_target))
        
        # 频偏MAE (反归一化)
        freq_pred = predictions['sync'][:, 1] * 0.1  # 反归一化
        freq_target = targets['sync'][:, 1] * 0.1
        results['freq_mae'] = np.mean(np.abs(freq_pred - freq_target))
        
        # 各个分量的详细指标
        results['gains_mse'] = np.mean((predictions['gains'] - targets['gains']) ** 2)
        results['delays_mse'] = np.mean((predictions['delays'] - targets['delays']) ** 2)
        results['aoas_mse'] = np.mean((predictions['aoas'] - targets['aoas']) ** 2)
        results['paths_mse'] = np.mean((predictions['paths'] - targets['paths']) ** 2)
        
        # 路径检测准确率
        path_pred_binary = (predictions['paths'] > 0.5).astype(int)
        path_target_binary = targets['paths'].astype(int)
        results['path_accuracy'] = np.mean(path_pred_binary == path_target_binary)
        
        # 增益估计的归一化均方根误差 (NRMSE)
        gains_rmse = np.sqrt(results['gains_mse'])
        gains_range = np.max(targets['gains']) - np.min(targets['gains'])
        results['gains_nrmse'] = gains_rmse / gains_range if gains_range > 0 else 0
        
        # 延迟估计的归一化均方根误差
        delays_rmse = np.sqrt(results['delays_mse'])
        delays_range = np.max(targets['delays']) - np.min(targets['delays'])
        results['delays_nrmse'] = delays_rmse / delays_range if delays_range > 0 else 0
        
        return results
    
    def _print_evaluation_results(self, results):
        """打印评估结果"""
        print("\n" + "=" * 50)
        print("=== 模型评估结果 ===")
        print("=" * 50)
        
        # 主要指标
        print(f"信道参数 MSE: {results['channel_mse']:.6f}")
        print(f"同步参数 MSE: {results['sync_mse']:.6f}")
        print(f"定时偏移 MAE: {results['timing_mae']:.6f}")
        print(f"频偏 MAE: {results['freq_mae']:.6f}")
        
        print("\n" + "-" * 30)
        print("=== 详细指标 ===")
        print("-" * 30)
        
        # 详细指标
        print(f"路径增益 MSE: {results['gains_mse']:.6f}")
        print(f"延迟抽头 MSE: {results['delays_mse']:.6f}")
        print(f"到达角 MSE: {results['aoas_mse']:.6f}")
        print(f"路径存在性 MSE: {results['paths_mse']:.6f}")
        
        print(f"\n路径检测准确率: {results['path_accuracy']:.4f}")
        print(f"增益估计 NRMSE: {results['gains_nrmse']:.6f}")
        print(f"延迟估计 NRMSE: {results['delays_nrmse']:.6f}")
        
        print("=" * 50)
    
    def plot_evaluation_results(self, results, predictions, targets):
        """可视化"""
        # 创建单个图表
        plt.figure(figsize=(10, 6))
        
        # 训练历史
        plt.plot(self.train_history, label='Train Loss', alpha=0.8, linewidth=2, color='#1f77b4')
        plt.plot(self.val_history, label='Validation Loss', alpha=0.8, linewidth=2, color='#ff7f0e')
        
        # 设置图表标题和标签
        plt.title('训练和验证损失历史', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        
        # 设置图例
        plt.legend(fontsize=12, loc='best')
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 设置图表样式
        plt.tight_layout()
        plt.show()
