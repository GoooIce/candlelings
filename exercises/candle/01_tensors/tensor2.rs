use candle_core::{Result, Tensor, Device};

/// 练习1：张量加法
/// 实现两个张量的加法运算
fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // TODO: 实现张量加法运算
    todo!("实现张量加法")
}

/// 练习2：张量乘法
/// 实现两个矩阵的乘法运算
fn matrix_multiply(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // TODO: 实现矩阵乘法
    todo!("实现矩阵乘法")
}

/// 练习3：标量乘法
/// 实现张量与标量的乘法运算
fn scalar_multiply(t: &Tensor, scalar: f32) -> Result<Tensor> {
    // TODO: 实现标量乘法
    todo!("实现标量乘法")
}

/// 练习4：逐元素运算
/// 实现张量的逐元素乘法（Hadamard乘积）
fn element_wise_multiply(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // TODO: 实现逐元素乘法
    todo!("实现逐元素乘法")
}

/// 练习5：原位操作
/// 实现张量的原位加法操作（就地修改张量的值）
fn add_inplace(t: &mut Tensor, other: &Tensor) -> Result<()> {
    // TODO: 实现原位加法操作
    todo!("实现原位加法")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let b = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &device)?;
        let c = add_tensors(&a, &b)?;
        assert_eq!(
            c.to_vec2::<f32>()?,
            vec![vec![6.0, 8.0], vec![10.0, 12.0]]
        );
        Ok(())
    }

    #[test]
    fn test_matrix_multiply() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let b = Tensor::new(&[[2.0f32, 0.0], [1.0, 3.0]], &device)?;
        let c = matrix_multiply(&a, &b)?;
        assert_eq!(
            c.to_vec2::<f32>()?,
            vec![vec![4.0, 6.0], vec![10.0, 12.0]]
        );
        Ok(())
    }

    #[test]
    fn test_scalar_multiply() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let b = scalar_multiply(&a, 2.0)?;
        assert_eq!(
            b.to_vec2::<f32>()?,
            vec![vec![2.0, 4.0], vec![6.0, 8.0]]
        );
        Ok(())
    }

    #[test]
    fn test_element_wise_multiply() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let b = Tensor::new(&[[2.0f32, 3.0], [4.0, 5.0]], &device)?;
        let c = element_wise_multiply(&a, &b)?;
        assert_eq!(
            c.to_vec2::<f32>()?,
            vec![vec![2.0, 6.0], vec![12.0, 20.0]]
        );
        Ok(())
    }

    #[test]
    fn test_add_inplace() -> Result<()> {
        let device = Device::Cpu;
        let mut a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let b = Tensor::new(&[[2.0f32, 3.0], [4.0, 5.0]], &device)?;
        add_inplace(&mut a, &b)?;
        assert_eq!(
            a.to_vec2::<f32>()?,
            vec![vec![3.0, 5.0], vec![7.0, 9.0]]
        );
        Ok(())
    }
}

// 提示：
// - 使用张量的add()方法进行加法运算
// - 使用matmul()进行矩阵乘法
// - 使用mul()进行标量乘法
// - 使用mul()或broadcast_mul()进行逐元素乘法
// - 注意检查张量的形状是否兼容
// - 原位操作通常需要使用mut关键字标记可变性
