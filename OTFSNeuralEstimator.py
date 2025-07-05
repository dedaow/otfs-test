class OTFSNeuralEstimator(nn.Module):
    """
    优化的OTFS神经网络估计器
    专门设计用于OTFS信号的2D结构
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
        fusion_dim = 128 * 8 * 8 + 128 * 64  # 2D延迟多普勒 特征 + 1D延迟时间 特征
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
