class OTFSSignalProcessor:
    """
    接收信号: r[k] = e^(j2πεk/MN) × Σ(ℓ=0 to L-1) h[ℓ,k] × s[k-ℓ-θ] + η[k]
    """
    def __init__(self, M=128, N=32):
        self.M = M
        self.N = N
        self.NT = M * N
        
    def generate_pilot_signal(self, pilot_type='impulse', power_scale=1.0):
        """生成导频信号"""
        if pilot_type == 'impulse':
            # 在delay-Doppler域放置冲激导频,没有使用qpsk调制的，直接impulse
            pilot_dd = np.zeros((self.M, self.N), dtype=complex)
            
            # 导频位置还没有细看，整个网格很大 所以我先放4个
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
            
            # 串行化，ps，sp那个
            s = pilot_dt.T.flatten()
            
            pilot_info = {
                'positions': pilot_positions,
                'type': 'impulse',
                'dd_grid': pilot_dd
            }
            
        elif pilot_type == 'scattered':
            # 如果4个不够，多放点
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
