use candle_core::{Device, Result, Tensor};

/// 练习1：复合函数求导
/// 计算 y = sin(x^2) 的导数
fn compute_composite_gradient(x: &Tensor) -> Result<Tensor> {
    // TODO: 计算复合函数的梯度
    // 提示：使用链式法则
    todo!("计算复合函数梯度")
}

/// 练习2：多变量函数求导
/// 计算 z = x^2 + y^2 关于x和y的偏导数
fn compute_partial_derivatives(x: &Tensor, y: &Tensor) -> Result<(Tensor, Tensor)> {
    // TODO: 计算多变量函数的偏导数
    todo!("计算偏导数")
}

/// 练习3：梯度累积和链式法则
/// 计算 y = x^3 在多个点上的梯度并累积
fn accumulate_cube_gradients(points: &[f32]) -> Result<Tensor> {
    // TODO: 在多个点上累积立方函数的梯度
    todo!("累积多点梯度")
}

/// 练习4：向量函数求导
/// 计算向量值函数 f(x) = [x^2, x^3] 的雅可比矩阵
fn compute_jacobian(x: &Tensor) -> Result<Tensor> {
    // TODO: 计算雅可比矩阵
    todo!("计算雅可比矩阵")
}

/// 练习5：优化梯度计算
/// 实现带有梯度裁剪的梯度计算
fn compute_clipped_gradient(x: &Tensor, min_grad: f32, max_grad: f32) -> Result<Tensor> {
    // TODO: 实现带裁剪的梯度计算
    todo!("实现梯度裁剪")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_gradient() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(2.0f32, &device)?.requires_grad(true);
        let grad = compute_composite_gradient(&x)?;
        // 验证梯度值（sin(x^2)的导数在x=2处）
        let expected = 2.0 * 2.0 * (-0.7568024953079282f32).sin();
        assert!((grad.to_vec0::<f32>()? - expected).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_partial_derivatives() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(2.0f32, &device)?.requires_grad(true);
        let y = Tensor::new(3.0f32, &device)?.requires_grad(true);
        let (dx, dy) = compute_partial_derivatives(&x, &y)?;
        // ∂z/∂x = 2x, ∂z/∂y = 2y
        assert_eq!(dx.to_vec0::<f32>()?, 4.0); // 2 * 2
        assert_eq!(dy.to_vec0::<f32>()?, 6.0); // 2 * 3
        Ok(())
    }

    #[test]
    fn test_accumulate_cube() -> Result<()> {
        let points = vec![1.0f32, 2.0, 3.0];
        let grad = accumulate_cube_gradients(&points)?;
        // sum of 3x^2 at x = 1,2,3
        assert_eq!(grad.to_vec0::<f32>()?, 3.0 + 12.0 + 27.0);
        Ok(())
    }

    #[test]
    fn test_jacobian() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(2.0f32, &device)?.requires_grad(true);
        let j = compute_jacobian(&x)?;
        // [[2x, 3x^2]] at x=2
        assert_eq!(j.to_vec2::<f32>()?, vec![vec![4.0, 12.0]]);
        Ok(())
    }

    #[test]
    fn test_clipped_gradient() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(3.0f32, &device)?.requires_grad(true);
        let grad = compute_clipped_gradient(&x, -5.0, 5.0)?;
        // 2x at x=3 should be 6, but clipped to 5
        assert_eq!(grad.to_vec0::<f32>()?, 5.0);
        Ok(())
    }
}

// 提示：
// - 使用链式法则处理复合函数
// - 对于多变量函数，分别计算各个变量的偏导数
// - 梯度累积时注意清零时机
// - 向量函数求导需要考虑每个分量
// - 实现梯度裁剪时注意数值范围的处理
// - 验证梯度计算的正确性时考虑数值误差
