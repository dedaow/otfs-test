# otfs
按照信道与接收的两个模型，一次性估计所有参数，

损失函数是加权后的所有loss   loss=weight1*parameter1+weight2*parameter2+weight*parameter

特征提取：不同于原始论文里按照时频分别估计，补偿等，这里直接设计两个域（延迟多普勒 延迟时间）
    
延迟多普勒     延迟时间    分别提取后融合，输出多个参数
