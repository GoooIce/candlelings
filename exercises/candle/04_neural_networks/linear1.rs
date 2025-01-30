use candle_core::{Result, Tensor, Device, Module};

/// 练习1：线性层结构定义
/// 实现线性层的基本结构
#[derive(Debug)]
struct Linear {
    weight: Tensor,         // 权重矩阵
    bias: Option<Tensor>,   // 可选的偏置项
    in_features: usize,     // 输入特征维度
    out_features: usize,    // 输出特征维度
}

impl Linear {
    /// 创建新的线性层
    fn new(in_features: usize, out_features: usize, bias: bool, device: &Device) -> Result<Self> {
        // TODO: 实现线性层的创建
        // - 初始化权重（使用Xavier初始化）
        // - 初始化偏置（如果需要）
        // - 设置requires_grad
        todo!("实现线性层创建")
    }

    /// 实现Xavier权重初始化
    fn xavier_init(in_features: usize, out_features: usize, device: &Device) -> Result<Tensor> {
        // TODO: 实现Xavier初始化
        // bound = sqrt(6.0 / (in_features + out_features))
        // uniform(-bound, bound)
        todo!("实现Xavier初始化")
    }
}

/// 练习2：前向传播实现
impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: 实现前向传播
        // output = input @ weight.t() + bias
        todo!("实现前向传播")
    }
}

/// 练习3：参数管理
impl Linear {
    /// 获取所有可训练参数
    fn parameters(&self) -> Vec<&Tensor> {
        // TODO: 返回所有需要训练的参数
        todo!("实现参数获取")
    }

    /// 获取参数数量
    fn num_parameters(&self) -> usize {
        // TODO: 计算参数总数
        todo!("实现参数计数")
    }
}

/// 练习4：设备管理
impl Linear {
    /// 将层移动到指定设备
    fn to_device(&self, device: &Device) -> Result<Self> {
        // TODO: 实现设备转移
        todo!("实现设备转移")
    }
}

/// 练习5：自定义初始化
impl Linear {
    /// 使用自定义方法初始化权重
    fn init_weight(&mut self, init_fn: impl Fn(&Tensor) -> Result<Tensor>) -> Result<()> {
        // TODO: 实现自定义初始化
        todo!("实现自定义初始化")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() -> Result<()> {
        let device = Device::Cpu;
        let linear = Linear::new(10, 5, true, &device)?;
        assert_eq!(linear.in_features, 10);
        assert_eq!(linear.out_features, 5);
        assert!(linear.bias.is_some());
        Ok(())
    }

    #[test]
    fn test_forward() -> Result<()> {
        let device = Device::Cpu;
        let linear = Linear::new(2, 3, true, &device)?;
        let input = Tensor::new(&[[1.0f32, 2.0]], &device)?;
        let output = linear.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 3]);
        Ok(())
    }

    #[test]
    fn test_parameters() -> Result<()> {
        let device = Device::Cpu;
        let linear = Linear::new(2, 3, true, &device)?;
        let params = linear.parameters();
        assert_eq!(params.len(), 2); // weight和bias
        Ok(())
    }

    #[test]
    fn test_device_transfer() -> Result<()> {
        let cpu = Device::Cpu;
        let linear = Linear::new(2, 3, true, &cpu)?;
        let _linear_cpu = linear.to_device(&cpu)?;
        Ok(())
    }

    #[test]
    fn test_custom_init() -> Result<()> {
        let device = Device::Cpu;
        let mut linear = Linear::new(2, 3, true, &device)?;
        linear.init_weight(|w| w.mul_scalar(2.0))?;
        Ok(())
    }
}

// 提示：
// - 使用张量的矩阵运算方法
// - 注意形状兼容性检查
// - 合理处理偏置项
// - 正确设置requires_grad
// - 实现高效的设备转移
