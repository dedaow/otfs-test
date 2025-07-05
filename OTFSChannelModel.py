class OTFSChannelModel:
    """
    h[ℓ,k] = Σ(i=0 to Γ-1) α_i × e^(j2πκ_max/MN × (k-ℓ)cosψ_i) × δ[ℓ-ℓ_i]
    """
    def __init__(self, M=128, N=32, max_delay_taps=8, kappa_max=0.3):
        self.M = M  # 延迟bins
        self.N = N  # 多普勒bins
        self.L = max_delay_taps  # 最大延迟抽头
        self.kappa_max = kappa_max  # 最大归一化多普勒频移
        
    def generate_channel_parameters(self, num_paths=None):
        """生成信道参数"""
        if num_paths is None:
            num_paths = np.random.randint(1, min(6, self.L + 1))  # 限制路径数
        
        # 复高斯路径增益 (瑞利衰落)
        alpha = (np.random.normal(0, 1, num_paths) + 
                1j * np.random.normal(0, 1, num_paths)) / np.sqrt(2)
        
        # 延迟抽头位置 (不重复)
        delay_taps = np.sort(np.random.choice(self.L, num_paths, replace=False))
        
        # 到达角 (均匀分布)ψ
        psi = np.random.uniform(-np.pi, np.pi, num_paths)#-pi到pi
        
        # 功率归一化
        alpha = alpha / np.sqrt(np.sum(np.abs(alpha)**2))
        
        return {
            'alpha': alpha,
            'delay_taps': delay_taps,
            'psi': psi,
            'num_paths': num_paths,
            'Gamma': num_paths  # 路径总数
        }
    
    def compute_channel_response(self, channel_params, time_samples):
        """
        计算信道响应矩阵 h[ℓ,k]
        信道公式: h[ℓ,k] = Σ α_i × e^(j2πκ_max/MN × (k-ℓ)cosψ_i) × δ[ℓ-ℓ_i]
        """
        alpha = channel_params['alpha']
        delay_taps = channel_params['delay_taps']
        psi = channel_params['psi']
        Gamma = channel_params['Gamma']
        
        # 初始化信道响应矩阵 [L × time_samples]
        h = np.zeros((self.L, len(time_samples)), dtype=complex)#先全部归零
        
        # 对每条路径计算贡献
        for i in range(Gamma):#外层遍历所有的路径后，内层遍历下所有的k
            ell_i = delay_taps[i]  # 第i条路径的延迟抽头
            alpha_i = alpha[i]     # 第i条路径的增益
            psi_i = psi[i]         # 第i条路径的到达角
            
            # 计算时变相位项
            for k_idx, k in enumerate(time_samples):
                # 相位项: e^(j2πκ_max/MN × (k-ℓ)cosψ_i)
                phase_arg = 2 * np.pi * self.kappa_max / (self.M * self.N) * (k - ell_i) * np.cos(psi_i)
                phase_term = np.exp(1j * phase_arg)
                
                # 仅在对应的延迟抽头位置有非零响应 (δ[ℓ-ℓ_i])
                h[ell_i, k_idx] += alpha_i * phase_term
        
        return h

外循环：
1延迟位置 ℓ：决定这条路径的能量落在哪一行；
  
2增益 α：路径的复数衰落（包括幅度与相位）；
  
 3 角度 ψ：路径的入射角，对应多普勒频移。
内循环：  
  所有时间采样 k	计算当前路径 i 在每个时间点的相位影响
