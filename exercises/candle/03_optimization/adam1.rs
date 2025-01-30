use candle_core::{Result, Tensor, Device};

/// Adam优化器配置
struct AdamConfig {
    lr: f32,              // 学习率
    betas: (f32, f32),    // 动量系数 (β1, β2)
    eps: f32,             // 数值稳定项
    weight_decay: f32,    // 权重衰减
    amsgrad: bool,        // 是否使用AMSGrad变体
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        }
    }
}

/// 练习1：初始化Adam状态
/// 为参数创建必要的状态变量
fn initialize_adam_state(param: &Tensor) -> Result<(Tensor, Tensor)> {
    // TODO: 初始化一阶矩和二阶矩
    // m = zeros_like(param)
    // v = zeros_like(param)
    todo!("初始化Adam状态")
}

/// 练习2：实现基本的Adam更新
/// 不考虑AMSGrad，实现标准Adam算法
fn basic_adam_update(
    param: &mut Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    config: &AdamConfig,
    step: usize,
) -> Result<()> {
    // TODO: 实现Adam更新规则
    // m = β1*m + (1-β1)*grad
    // v = β2*v + (1-β2)*grad^2
    // m_hat = m/(1-β1^t)
    // v_hat = v/(1-β2^t)
    // param = param - lr*m_hat/(sqrt(v_hat) + ε)
    todo!("实现Adam更新")
}

/// 练习3：实现AMSGrad变体
/// 使用最大二阶矩估计
fn amsgrad_update(
    param: &mut Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    v_max: &mut Tensor,
    config: &AdamConfig,
    step: usize,
) -> Result<()> {
    // TODO: 实现AMSGrad更新规则
    // m = β1*m + (1-β1)*grad
    // v = β2*v + (1-β2)*grad^2
    // v_max = max(v_max, v)
    // m_hat = m/(1-β1^t)
    // param = param - lr*m_hat/(sqrt(v_max) + ε)
    todo!("实现AMSGrad")
}

/// 练习4：带权重衰减的Adam
/// 实现AdamW变体
fn adamw_update(
    param: &mut Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    config: &AdamConfig,
    step: usize,
) -> Result<()> {
    // TODO: 实现AdamW更新规则
    // 在更新前应用权重衰减
    todo!("实现AdamW")
}

/// 练习5：实现学习率调度
/// 实现Adam的warm-up学习率调度
fn adam_lr_schedule(base_lr: f32, step: usize, warmup_steps: usize) -> f32 {
    // TODO: 实现warm-up学习率调度
    // lr = base_lr * min(step^(-0.5), step * warmup_steps^(-1.5))
    todo!("实现学习率调度")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_state() -> Result<()> {
        let device = Device::Cpu;
        let param = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let (m, v) = initialize_adam_state(&param)?;
        
        // 检查形状
        assert_eq!(m.shape(), param.shape());
        assert_eq!(v.shape(), param.shape());
        
        // 检查初始值
        assert!(m.sum()?.to_vec0::<f32>()? < 1e-6);
        assert!(v.sum()?.to_vec0::<f32>()? < 1e-6);
        Ok(())
    }

    #[test]
    fn test_basic_adam() -> Result<()> {
        let device = Device::Cpu;
        let mut param = Tensor::new(1.0f32, &device)?;
        let grad = Tensor::new(0.1f32, &device)?;
        let mut m = Tensor::new(0.0f32, &device)?;
        let mut v = Tensor::new(0.0f32, &device)?;
        let config = AdamConfig::default();

        basic_adam_update(&mut param, &grad, &mut m, &mut v, &config, 1)?;
        
        // 验证更新后的值
        let param_val = param.to_vec0::<f32>()?;
        assert!(param_val < 1.0 && param_val > 0.99);
        Ok(())
    }

    #[test]
    fn test_amsgrad() -> Result<()> {
        let device = Device::Cpu;
        let mut param = Tensor::new(1.0f32, &device)?;
        let grad = Tensor::new(0.1f32, &device)?;
        let mut m = Tensor::new(0.0f32, &device)?;
        let mut v = Tensor::new(0.0f32, &device)?;
        let mut v_max = Tensor::new(0.0f32, &device)?;
        let config = AdamConfig {
            amsgrad: true,
            ..Default::default()
        };

        amsgrad_update(&mut param, &grad, &mut m, &mut v, &mut v_max, &config, 1)?;
        
        // 验证更新后的值
        assert!(param.to_vec0::<f32>()? < 1.0);
        Ok(())
    }

    #[test]
    fn test_adamw() -> Result<()> {
        let device = Device::Cpu;
        let mut param = Tensor::new(1.0f32, &device)?;
        let grad = Tensor::new(0.1f32, &device)?;
        let mut m = Tensor::new(0.0f32, &device)?;
        let mut v = Tensor::new(0.0f32, &device)?;
        let config = AdamConfig {
            weight_decay: 0.01,
            ..Default::default()
        };

        adamw_update(&mut param, &grad, &mut m, &mut v, &config, 1)?;
        
        // 带权重衰减的更新应该比基本的Adam减少得更多
        assert!(param.to_vec0::<f32>()? < 0.99);
        Ok(())
    }

    #[test]
    fn test_warmup_schedule() {
        let base_lr = 0.001;
        let warmup_steps = 1000;
        
        // 热身阶段
        let lr_warmup = adam_lr_schedule(base_lr, 100, warmup_steps);
        assert!(lr_warmup < base_lr);
        
        // 热身后
        let lr_after = adam_lr_schedule(base_lr, 2000, warmup_steps);
        assert!(lr_after < lr_warmup);
    }
}

// 提示：
// - 使用Tensor的算术操作方法
// - 注意数值稳定性，避免除零
// - 正确处理偏差修正
// - 高效实现原位更新
// - 适当使用设备无关的操作
