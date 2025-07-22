import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.special import jv
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

class OptimizedOTFSChannelModel:
    """
    信道模型
    h[ℓ,k] = Σ(i=0 to Γ-1) α_i × e^(j2πκ_max/MN × (k-ℓ)cosψ_i) × δ[ℓ-ℓ_i]
    """
    def __init__(self, M=128, N=32, max_delay_taps=8, kappa_max=0.3):
        self.M = M  # 延迟bins
        self.N = N  # 多普勒bins
        self.L = max_delay_taps  # 最大延迟抽头
        self.kappa_max = kappa_max  # 最大归一化多普勒频移
        
    def generate_channel_parameters(self, num_paths=None):
        """信道参数"""
        if num_paths is None:
            num_paths = np.random.randint(1, min(6, self.L + 1))  # 最多6个
        
        # 复高斯路径增益 (瑞利衰落)
        alpha = (np.random.normal(0, 1, num_paths) + 
                1j * np.random.normal(0, 1, num_paths)) / np.sqrt(2)
        
        # 延迟抽头位置 (不重复)
        delay_taps = np.sort(np.random.choice(self.L, num_paths, replace=False))
        
        # 到达角 (均匀分布)
        psi = np.random.uniform(-np.pi, np.pi, num_paths)
        
        # 功率归一化
        alpha = alpha / np.sqrt(np.sum(np.abs(alpha)**2))
        
        return {
            'alpha': alpha,
            'delay_taps': delay_taps,
            'psi': psi,
            'num_paths': num_paths,
            'Gamma': num_paths
        }
    
    def compute_channel_response(self, channel_params, time_samples):
        """
        信道响应矩阵 h[ℓ,k]
         h[ℓ,k] = Σ α_i × e^(j2πκ_max/MN × (k-ℓ)cosψ_i) × δ[ℓ-ℓ_i]
        """
        alpha = channel_params['alpha']
        delay_taps = channel_params['delay_taps']
        psi = channel_params['psi']
        Gamma = channel_params['Gamma']
        
        # 初始化信道响应矩阵 [L × time_samples]
        h = np.zeros((self.L, len(time_samples)), dtype=complex)
        
        # 对每条路径计算贡献
        for i in range(Gamma):
            ell_i = delay_taps[i]  # 第i条路径的延迟抽头
            alpha_i = alpha[i]     # 第i条路径的增益
            psi_i = psi[i]         # 第i条路径的到达角
            
            # 计算时变相位项
            for k_idx, k in enumerate(time_samples):
                # 相位项: e^(j2πκ_max/MN × (k-ℓ)cosψ_i)
                phase_arg = 2 * np.pi * self.kappa_max / (self.M * self.N) * (k - ell_i) * np.cos(psi_i)
                phase_term = np.exp(1j * phase_arg)
                
                # 仅在对应的延迟抽头位置有非零响应 (δ[ℓ-ℓ_i])，
                h[ell_i, k_idx] += alpha_i * phase_term
        
        return h

class OptimizedOTFSSignalProcessor:
    """
    接收信号模型
    接收信号: r[k] = e^(j2πεk/MN) × Σ(ℓ=0 to L-1) h[ℓ,k] × s[k-ℓ-θ] + η[k]
    """
    def __init__(self, M=128, N=32):
        self.M = M
        self.N = N
        self.NT = M * N
        
    def generate_pilot_signal(self, pilot_type='impulse', power_scale=1.0):
        """生成导频信号"""
        if pilot_type == 'impulse':
            # 在delay-Doppler域放置冲激导频
            pilot_dd = np.zeros((self.M, self.N), dtype=complex)
            
            # 多个导频位置以提高估计精度
            pilot_positions = [
                (self.M // 4, self.N // 4),
                (3 * self.M // 4, self.N // 4),
                (self.M // 4, 3 * self.N // 4),
                (3 * self.M // 4, 3 * self.N // 4)
            ]
            
            for mp, np_pilot in pilot_positions:
                pilot_dd[mp, np_pilot] = np.sqrt(self.M * self.N) * power_scale / len(pilot_positions)
            
            # 变换到delay-time域
            pilot_dt = np.fft.ifft(pilot_dd, axis=1) * np.sqrt(self.N)
            
            # 串行化
            s = pilot_dt.T.flatten()
            
            pilot_info = {
                'positions': pilot_positions,
                'type': 'impulse',
                'dd_grid': pilot_dd
            }
            
        elif pilot_type == 'scattered':
            # 散布导频
            pilot_dd = np.zeros((self.M, self.N), dtype=complex)
            
            # 规则间隔的导频
            pilot_positions = []
            for m in range(0, self.M, self.M // 8):
                for n in range(0, self.N, self.N // 4):
                    pilot_dd[m, n] = np.sqrt(self.M * self.N) * power_scale / 32
                    pilot_positions.append((m, n))
            
            # 变换到delay-time域
            pilot_dt = np.fft.ifft(pilot_dd, axis=1) * np.sqrt(self.N)
            s = pilot_dt.T.flatten()
            
            pilot_info = {
                'positions': pilot_positions,
                'type': 'scattered',
                'dd_grid': pilot_dd
            }
        
        return s, pilot_info
    
    def channel_transmission(self, s, channel_model, channel_params, 
                           timing_offset=0, freq_offset=0, noise_var=0):
        """
        信道传输：r[k] = e^(j2πεk/MN) × Σ(ℓ=0 to L-1) h[ℓ,k] × s[k-ℓ-θ] + η[k]
        """
        # 扩展信号以处理延迟
        s_ext = np.concatenate([s, np.zeros(channel_model.L)])
        time_samples = np.arange(len(s_ext))
        
        # 计算信道响应
        h = channel_model.compute_channel_response(channel_params, time_samples)
        
        # 初始化接收信号
        r = np.zeros(len(s), dtype=complex)
        
        # 按公式计算接收信号
        for k in range(len(s)):
            # 载波频偏项: e^(j2πεk/MN)
            cfo_term = np.exp(1j * 2 * np.pi * freq_offset * k / (self.M * self.N))
            
            # 信道卷积项: Σ(ℓ=0 to L-1) h[ℓ,k] × s[k-ℓ-θ]
            conv_sum = 0
            for ell in range(channel_model.L):
                signal_idx = k - ell - timing_offset
                if 0 <= signal_idx < len(s_ext):
                    conv_sum += h[ell, k] * s_ext[signal_idx]
            
            # 组合信号
            r[k] = cfo_term * conv_sum
        
        # 添加噪声
        if noise_var > 0:
            noise = np.sqrt(noise_var/2) * (np.random.normal(0, 1, len(r)) + 
                                           1j * np.random.normal(0, 1, len(r)))
            r += noise
        
        return r
    
    def extract_delay_doppler_features(self, received_signal):
        """提取delay-Doppler域特征"""
        # 重塑为delay-time域
        r_dt = received_signal.reshape(self.N, self.M).T
        
        # 变换到delay-Doppler域
        r_dd = np.fft.fft(r_dt, axis=1) / np.sqrt(self.N)
        
        return r_dt, r_dd

class OTFSNeuralEstimator(nn.Module):
    """
    对于DD域，采用两种卷积核分别在延迟和多普勒卷积
    对于延迟时间域，1dcnn
    """
    def __init__(self, M=128, N=32, L=8):
        super(OTFSNeuralEstimator, self).__init__()
        self.M = M
        self.N = N
        self.L = L
        
        # 2D卷积特征提取器 (处理delay-Doppler域)
        self.dd_feature_extractor = nn.Sequential(
            # 第一层：捕获局部特征
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 实部虚部分离输入
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 第二层：延迟维特征
            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 第三层：多普勒维特征
            nn.Conv2d(64, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 第四层：联合特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # 1D卷积特征提取器 (处理delay-time域)
        self.dt_feature_extractor = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        
        # 特征融合
        fusion_dim = 128 * 8 * 8 + 128 * 64  #2d和1d
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 多任务输出头
        # 路径增益估计 (实部+虚部)
        self.gain_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2 * L)  # 实部和虚部
        )
        
        # 延迟抽头估计
        self.delay_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, L),
            nn.Sigmoid()  # 归一化到[0,1]
        )
        
        # 到达角估计
        self.aoa_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, L),
            nn.Tanh()  # 归一化到[-1,1]
        )
        
        # 路径存在性估计
        self.path_detector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, L),
            nn.Sigmoid()
        )
        
        # 同步参数估计
        self.sync_estimator = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),  # 定时偏移 + 频偏
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, dd_features, dt_features):
        """
        前向传播
        Args:
            dd_features: delay-Doppler域特征 [batch, 2, M, N]
            dt_features: delay-time域特征 [batch, 2, M*N]
        """
        batch_size = dd_features.size(0)
        
        # 2D特征提取
        dd_feat = self.dd_feature_extractor(dd_features)
        dd_feat = dd_feat.view(batch_size, -1)
        
        # 1D特征提取
        dt_feat = self.dt_feature_extractor(dt_features)
        dt_feat = dt_feat.view(batch_size, -1)
        
        # 特征融合
        fused_feat = torch.cat([dd_feat, dt_feat], dim=1)
        fused_feat = self.fusion_layer(fused_feat)
        
        # 多任务输出
        gains = self.gain_estimator(fused_feat)
        delays = self.delay_estimator(fused_feat)
        aoas = self.aoa_estimator(fused_feat)
        paths = self.path_detector(fused_feat)
        sync = self.sync_estimator(fused_feat)
        
        return {
            'gains': gains,      # [batch, 2*L]
            'delays': delays,    # [batch, L]
            'aoas': aoas,        # [batch, L]
            'paths': paths,      # [batch, L]
            'sync': sync         # [batch, 2]
        }

class OptimizedOTFSDataset(Dataset):
    """OTFS数据"""
    def __init__(self, num_samples=1000, M=64, N=32, L=6, snr_range=(10, 30)):
        self.num_samples = num_samples
        self.M = M
        self.N = N
        self.L = L
        self.snr_range = snr_range
        
        # 初始化模型
        self.channel_model = OptimizedOTFSChannelModel(M, N, L)
        self.signal_processor = OptimizedOTFSSignalProcessor(M, N)
        
        # 生成数据
        self.data = self._generate_samples()
    
    def _generate_samples(self):
        """生成数据样本"""
        samples = []
        
        print(f"生成 {self.num_samples} 个优化的OTFS样本...")
        
        for i in range(self.num_samples):
            if (i + 1) % 100 == 0:
                print(f"进度: {i + 1}/{self.num_samples}")
            
            # 信道参数
            channel_params = self.channel_model.generate_channel_parameters()
            
            # 导频信号
            pilot_signal, pilot_info = self.signal_processor.generate_pilot_signal('impulse')
            
            # 随机同步参数
            timing_offset = np.random.randint(-5, 6)
            freq_offset = np.random.uniform(-0.05, 0.05)
            
            # 信道传输
            snr_db = np.random.uniform(*self.snr_range)
            noise_var = np.mean(np.abs(pilot_signal)**2) / (10**(snr_db/10))
            
            received_signal = self.signal_processor.channel_transmission(
                pilot_signal, self.channel_model, channel_params,
                timing_offset, freq_offset, noise_var
            )
            
            # 特征提取
            features = self._extract_features(received_signal)
            labels = self._encode_labels(channel_params, timing_offset, freq_offset)
            
            samples.append((features, labels))
        
        return samples
    
    def _extract_features(self, received_signal):
        """提取优化特征"""
        # 提取delay-Doppler和delay-time特征
        r_dt, r_dd = self.signal_processor.extract_delay_doppler_features(received_signal)
        
        # delay-Doppler域特征 (实部虚部分离)
        dd_real = r_dd.real
        dd_imag = r_dd.imag
        dd_features = np.stack([dd_real, dd_imag], axis=0)  # [2, M, N]
        
        # delay-time域特征 (实部虚部分离)
        dt_real = r_dt.real.flatten()
        dt_imag = r_dt.imag.flatten()
        dt_features = np.stack([dt_real, dt_imag], axis=0)  # [2, M*N]
        
        return {
            'dd_features': dd_features.astype(np.float32),
            'dt_features': dt_features.astype(np.float32)
        }
    
    def _encode_labels(self, channel_params, timing_offset, freq_offset):
        """编码标签"""
        labels = {}
        
        # 路径增益 (实部虚部)
        gains = np.zeros(2 * self.L)
        alpha = channel_params['alpha']
        for i in range(len(alpha)):
            gains[2*i] = alpha[i].real
            gains[2*i + 1] = alpha[i].imag
        
        # 延迟抽头 (归一化)
        delays = np.zeros(self.L)
        delay_taps = channel_params['delay_taps']
        for i in range(len(delay_taps)):
            delays[i] = delay_taps[i] / self.L
        
        # 到达角 (归一化)
        aoas = np.zeros(self.L)
        psi = channel_params['psi']
        for i in range(len(psi)):
            aoas[i] = psi[i] / np.pi
        
        # 路径存在性
        paths = np.zeros(self.L)
        for i in range(channel_params['num_paths']):
            paths[i] = 1.0
        
        # 同步参数
        sync = np.array([timing_offset / 10.0, freq_offset / 0.1])
        
        labels = {
            'gains': gains.astype(np.float32),
            'delays': delays.astype(np.float32),
            'aoas': aoas.astype(np.float32),
            'paths': paths.astype(np.float32),
            'sync': sync.astype(np.float32)
        }
        
        return labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, labels = self.data[idx]
        
        # 转换为tensor
        dd_features = torch.FloatTensor(features['dd_features'])
        dt_features = torch.FloatTensor(features['dt_features'])
        
        tensor_labels = {}
        for key, value in labels.items():
            tensor_labels[key] = torch.FloatTensor(value)
        
        return (dd_features, dt_features), tensor_labels

class OptimizedOTFSTrainer:
    """trainer"""
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
        """train"""
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
        """val"""
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
        """train"""
        print("开始训练OTFS信道估计...")
        
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
                torch.save(self.model.state_dict(), 'otfs_model.pth')
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
                print("early stop")
                break
        
        print("complete,let me see the result hh")
    
    def evaluate_model(self, test_loader):
        """评估模型性能"""
        print("val...")
        
        # 加载最佳模型
        try:
            self.model.load_state_dict(torch.load('otfs_model.pth', map_location=self.device))
            print("load model")
        except FileNotFoundError:
            print("sorry, no model")
        
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
        """指标"""
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
        """可视化评估结果 - 只显示训练和验证损失历史"""
        # 创建单个图表
        plt.figure(figsize=(10, 6))
        
        # 训练历史
        plt.plot(self.train_history, label='Train Loss', alpha=0.8, linewidth=2, color='#1f77b4')
        plt.plot(self.val_history, label='Validation Loss', alpha=0.8, linewidth=2, color='#ff7f0e')
        
        # 设置图表标题和标签
        plt.title('', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        
        # 设置图例
        plt.legend(fontsize=12, loc='best')
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 设置图表样式
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    print("OTFS神经网络信道估计")
    print("=" * 60)
    
    # 参数设置
    M, N, L = 64, 32, 6
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    
    print("\n1. 创建数据集...")
    train_dataset = OptimizedOTFSDataset(num_samples=20000, M=M, N=N, L=L, snr_range=(15, 25))
    val_dataset = OptimizedOTFSDataset(num_samples=4000, M=M, N=N, L=L, snr_range=(15, 25))
    test_dataset = OptimizedOTFSDataset(num_samples=4000, M=M, N=N, L=L, snr_range=(10, 20))
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 测试样本: {len(test_dataset)}")
    
    print("\n2. 创建模型...")
    model = OTFSNeuralEstimator(M=M, N=N, L=L)
    trainer = OptimizedOTFSTrainer(model, device=device)
    
    print("\n3. 训练模型...")
    trainer.train(train_loader, val_loader, num_epochs=50)
    
    print("\n4. 评估模型...")
    results, predictions, targets = trainer.evaluate_model(test_loader)
    
    print("\n5. 可视化结果...")
    trainer.plot_evaluation_results(results, predictions, targets)
    
    print("\n训练和评估完成！")
    
    return trainer, model, results

if __name__ == "__main__":
    trainer, model, results = main()
