use candle_core::{Result, Tensor, Module};

/// SGD优化器的配置参数
struct SGDConfig {
    lr: f32,                 // 学习率
    momentum: Option<f32>,   // 动量因子
    weight_decay: f32,       // 权重衰减
    nesterov: bool,          // 是否使用Nesterov动量
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: None,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

/// 练习1：实现基本的SGD更新
/// 不考虑动量，仅实现最基本的梯度下降
fn basic_sgd_update(param: &mut Tensor, grad: &Tensor, lr: f32) -> Result<()> {
    // TODO: 实现基本的SGD更新规则
    // param = param - lr * grad
    todo!("实现基本SGD更新")
}

/// 练习2：带动量的SGD更新
/// 实现带动量的参数更新规则
fn momentum_sgd_update(
    param: &mut Tensor,
    grad: &Tensor,
    velocity: &mut Tensor,
    config: &SGDConfig,
) -> Result<()> {
    // TODO: 实现带动量的SGD更新
    // v = momentum * v + grad
    // param = param - lr * v
    todo!("实现动量SGD更新")
}

/// 练习3：实现Nesterov动量
/// 使用Nesterov加速梯度的更新规则
fn nesterov_sgd_update(
    param: &mut Tensor,
    grad: &Tensor,
    velocity: &mut Tensor,
    config: &SGDConfig,
) -> Result<()> {
    // TODO: 实现Nesterov动量更新
    // v_next = momentum * v + grad
    // param = param - lr * (momentum * v_next + grad)
    todo!("实现Nesterov动量更新")
}

/// 练习4：权重衰减
/// 在更新中加入L2正则化
fn sgd_with_weight_decay(
    param: &mut Tensor,
    grad: &Tensor,
    weight_decay: f32,
    lr: f32,
) -> Result<()> {
    // TODO: 实现带权重衰减的更新
    // grad = grad + weight_decay * param
    // param = param - lr * grad
    todo!("实现带权重衰减的更新")
}

/// 练习5：学习率调度
/// 实现简单的学习率衰减
fn adjust_learning_rate(initial_lr: f32, epoch: usize, decay_factor: f32) -> f32 {
    // TODO: 实现学习率衰减
    // new_lr = initial_lr * (decay_factor ^ epoch)
    todo!("实现学习率调度")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_basic_sgd() -> Result<()> {
        let device = Device::Cpu;
        let mut param = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
        let grad = Tensor::new(&[0.1f32, 0.2, 0.3], &device)?;
        basic_sgd_update(&mut param, &grad, 0.1)?;
        let expected = Tensor::new(&[0.99f32, 1.98, 2.97], &device)?;
        assert!(param.sub(&expected)?.abs()?.max()? < 1e-5);
        Ok(())
    }

    #[test]
    fn test_momentum_sgd() -> Result<()> {
        let device = Device::Cpu;
        let mut param = Tensor::new(&[1.0f32], &device)?;
        let grad = Tensor::new(&[0.1f32], &device)?;
        let mut velocity = Tensor::new(&[0.0f32], &device)?;
        let config = SGDConfig {
            lr: 0.1,
            momentum: Some(0.9),
            weight_decay: 0.0,
            nesterov: false,
        };
        momentum_sgd_update(&mut param, &grad, &mut velocity, &config)?;
        // v = 0.1, param = 1.0 - 0.1 * 0.1 = 0.99
        assert!((param.to_vec0::<f32>()? - 0.99).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_nesterov() -> Result<()> {
        let device = Device::Cpu;
        let mut param = Tensor::new(&[1.0f32], &device)?;
        let grad = Tensor::new(&[0.1f32], &device)?;
        let mut velocity = Tensor::new(&[0.0f32], &device)?;
        let config = SGDConfig {
            lr: 0.1,
            momentum: Some(0.9),
            weight_decay: 0.0,
            nesterov: true,
        };
        nesterov_sgd_update(&mut param, &grad, &mut velocity, &config)?;
        // Final param should differ from standard momentum
        assert!((param.to_vec0::<f32>()? - 0.991).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_weight_decay() -> Result<()> {
        let device = Device::Cpu;
        let mut param = Tensor::new(&[1.0f32], &device)?;
        let grad = Tensor::new(&[0.1f32], &device)?;
        sgd_with_weight_decay(&mut param, &grad, 0.1, 0.1)?;
        // param = 1.0 - 0.1 * (0.1 + 0.1 * 1.0) = 0.98
        assert!((param.to_vec0::<f32>()? - 0.98).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_lr_schedule() {
        let initial_lr = 0.1;
        let decay_factor = 0.9;
        let epoch = 2;
        let new_lr = adjust_learning_rate(initial_lr, epoch, decay_factor);
        // 0.1 * (0.9 ^ 2) = 0.081
        assert!((new_lr - 0.081).abs() < 1e-5);
    }
}

// 提示：
// - 使用Tensor的add()、mul()等方法进行计算
// - 注意原位操作和非原位操作的选择
// - 确保数值稳定性
// - 正确处理动量项的累积
// - 考虑并行计算的影响
