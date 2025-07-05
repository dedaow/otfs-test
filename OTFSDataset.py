class OTFSDataset(Dataset):
    """数据集"""
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
            if (i + 1) % 500 == 0:
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
