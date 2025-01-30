use candle_core::{Device, Result, Tensor};

/// 练习1：创建需要梯度的张量
/// 创建一个标量张量，并设置requires_grad为true
fn create_tensor_with_grad() -> Result<Tensor> {
    // TODO: 创建一个值为2.0的标量张量，并启用梯度计算
    todo!("创建带梯度的张量")
}

/// 练习2：基本运算和梯度计算
/// 计算y = x^2的导数
fn compute_square_gradient(x: &Tensor) -> Result<Tensor> {
    // TODO: 计算y = x^2，进行反向传播，并返回x的梯度
    todo!("计算平方函数的梯度")
}

/// 练习3：多次梯度累积
/// 进行多次反向传播，观察梯度累积效果
fn accumulate_gradients(x: &Tensor, num_steps: usize) -> Result<Tensor> {
    // TODO: 多次计算y = x^2的梯度，观察梯度累积
    todo!("实现梯度累积")
}

/// 练习4：梯度清零
/// 在计算之间重置梯度
fn reset_gradients(x: &Tensor) -> Result<Tensor> {
    // TODO: 计算梯度，清零，再次计算
    todo!("实现梯度重置")
}

/// 练习5：防止梯度计算
/// 使用no_grad()避免梯度计算
fn prevent_gradient_computation(x: &Tensor) -> Result<Tensor> {
    // TODO: 使用no_grad()执行操作
    todo!("实现无梯度计算")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_with_grad() -> Result<()> {
        let t = create_tensor_with_grad()?;
        assert!(t.requires_grad());
        assert_eq!(t.to_vec0::<f32>()?, 2.0);
        Ok(())
    }

    #[test]
    fn test_square_gradient() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(3.0f32, &device)?.requires_grad(true);
        let grad = compute_square_gradient(&x)?;
        // d/dx(x^2) = 2x, so at x=3, gradient should be 6
        assert_eq!(grad.to_vec0::<f32>()?, 6.0);
        Ok(())
    }

    #[test]
    fn test_gradient_accumulation() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(2.0f32, &device)?.requires_grad(true);
        let grad = accumulate_gradients(&x, 3)?;
        // Gradient should accumulate: 3 * (2 * 2) = 12
        assert_eq!(grad.to_vec0::<f32>()?, 12.0);
        Ok(())
    }

    #[test]
    fn test_gradient_reset() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(2.0f32, &device)?.requires_grad(true);
        let grad = reset_gradients(&x)?;
        // After reset, should be just 2 * 2 = 4
        assert_eq!(grad.to_vec0::<f32>()?, 4.0);
        Ok(())
    }

    #[test]
    fn test_prevent_gradient() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(2.0f32, &device)?.requires_grad(true);
        let result = prevent_gradient_computation(&x)?;
        assert!(!result.requires_grad());
        assert_eq!(result.to_vec0::<f32>()?, 4.0); // x^2 = 4
        Ok(())
    }
}

// 提示：
// - 使用requires_grad(true)启用梯度计算
// - 使用backward()进行反向传播
// - 使用grad()获取梯度
// - 注意梯度累积和清零的时机
// - 使用no_grad()包装不需要梯度的计算
// - 确保在适当的时候释放计算图资源
