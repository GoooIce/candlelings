use candle_core::{Result, Tensor, Module};

/// 练习1：基本MSE损失
/// 实现简单的均方误差损失函数
#[derive(Debug)]
struct MSELoss {
    reduction: Reduction,
}

/// 损失函数的规约方式
#[derive(Debug, Clone, Copy)]
enum Reduction {
    None,      // 不规约，返回所有元素的损失
    Mean,      // 返回平均损失
    Sum,       // 返回总损失
}

impl MSELoss {
    /// 创建新的MSE损失函数
    fn new(reduction: Reduction) -> Self {
        // TODO: 实现MSE损失创建
        todo!("实现MSE损失创建")
    }

    /// 计算MSE损失
    fn forward(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // TODO: 实现MSE前向计算
        // squared_error = (pred - target)^2
        // 根据reduction规约方式返回结果
        todo!("实现MSE前向计算")
    }
}

/// 练习2：带权重的MSE损失
/// 实现支持样本权重的MSE损失
struct WeightedMSELoss {
    reduction: Reduction,
}

impl WeightedMSELoss {
    /// 计算带权重的MSE损失
    fn forward(&self, pred: &Tensor, target: &Tensor, weight: &Tensor) -> Result<Tensor> {
        // TODO: 实现带权重的MSE计算
        // weighted_squared_error = weight * (pred - target)^2
        todo!("实现带权重的MSE计算")
    }
}

/// 练习3：掩码MSE损失
/// 实现支持掩码的MSE损失（用于处理缺失值）
struct MaskedMSELoss {
    reduction: Reduction,
}

impl MaskedMSELoss {
    /// 计算带掩码的MSE损失
    fn forward(&self, pred: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // TODO: 实现带掩码的MSE计算
        // 只计算mask为1的位置的损失
        todo!("实现带掩码的MSE计算")
    }
}

/// 练习4：相对MSE损失
/// 实现相对均方误差损失
struct RelativeMSELoss {
    reduction: Reduction,
    epsilon: f32,  // 避免除零
}

impl RelativeMSELoss {
    /// 计算相对MSE损失
    fn forward(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // TODO: 实现相对MSE计算
        // relative_squared_error = ((pred - target)/(target + epsilon))^2
        todo!("实现相对MSE计算")
    }
}

/// 练习5：平滑MSE损失
/// 实现带平滑项的MSE损失
struct SmoothedMSELoss {
    reduction: Reduction,
    beta: f32,    // 平滑参数
}

impl SmoothedMSELoss {
    /// 计算平滑MSE损失
    fn forward(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        // TODO: 实现平滑MSE计算
        // 当差异小于beta时使用平方损失，否则使用线性损失
        todo!("实现平滑MSE计算")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_mse_basic() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
        let target = Tensor::new(&[2.0f32, 2.0, 4.0], &device)?;
        
        let mse = MSELoss::new(Reduction::Mean);
        let loss = mse.forward(&pred, &target)?;
        
        // ((1-2)^2 + (2-2)^2 + (3-4)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
        assert!((loss.to_vec0::<f32>()? - 2.0/3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_weighted_mse() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::new(&[1.0f32, 2.0], &device)?;
        let target = Tensor::new(&[2.0f32, 2.0], &device)?;
        let weight = Tensor::new(&[0.5f32, 1.5], &device)?;
        
        let mse = WeightedMSELoss { reduction: Reduction::Sum };
        let loss = mse.forward(&pred, &target, &weight)?;
        
        // 0.5*(1-2)^2 + 1.5*(2-2)^2 = 0.5
        assert!((loss.to_vec0::<f32>()? - 0.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_masked_mse() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
        let target = Tensor::new(&[2.0f32, 2.0, 4.0], &device)?;
        let mask = Tensor::new(&[1.0f32, 0.0, 1.0], &device)?;
        
        let mse = MaskedMSELoss { reduction: Reduction::Mean };
        let loss = mse.forward(&pred, &target, &mask)?;
        
        // ((1-2)^2 + (3-4)^2) / 2 = 1
        assert!((loss.to_vec0::<f32>()? - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_relative_mse() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::new(&[1.0f32, 2.0], &device)?;
        let target = Tensor::new(&[2.0f32, 4.0], &device)?;
        
        let mse = RelativeMSELoss { 
            reduction: Reduction::Mean,
            epsilon: 1e-6,
        };
        let loss = mse.forward(&pred, &target)?;
        
        // Mean of ((1-2)/2)^2 and ((2-4)/4)^2 = 0.25
        assert!((loss.to_vec0::<f32>()? - 0.25).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_smoothed_mse() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::new(&[1.0f32, 2.0], &device)?;
        let target = Tensor::new(&[2.0f32, 2.1], &device)?;
        
        let mse = SmoothedMSELoss { 
            reduction: Reduction::Mean,
            beta: 0.5,
        };
        let loss = mse.forward(&pred, &target)?;
        
        // First difference (1.0) uses linear loss, second difference (0.1) uses squared loss
        assert!(loss.to_vec0::<f32>()? > 0.0);
        Ok(())
    }
}

// 提示：
// - 使用Tensor的算术运算方法
// - 注意数值稳定性
// - 正确处理批量维度
// - 实现不同的规约方式
// - 确保梯度计算正确
