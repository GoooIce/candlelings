use candle_core::{Result, Tensor, Device};

/// 练习1：改变张量形状
/// 将2x3的矩阵重塑为3x2的矩阵
fn reshape_matrix(t: &Tensor) -> Result<Tensor> {
    // TODO: 实现张量重塑
    todo!("实现reshape操作")
}

/// 练习2：增加维度
/// 在指定位置增加一个维度（unsqueeze操作）
fn add_dimension(t: &Tensor, dim: usize) -> Result<Tensor> {
    // TODO: 实现维度增加
    todo!("实现unsqueeze操作")
}

/// 练习3：压缩维度
/// 移除大小为1的维度（squeeze操作）
fn remove_dimension(t: &Tensor, dim: usize) -> Result<Tensor> {
    // TODO: 实现维度压缩
    todo!("实现squeeze操作")
}

/// 练习4：转置操作
/// 交换矩阵的行和列
fn transpose_matrix(t: &Tensor) -> Result<Tensor> {
    // TODO: 实现矩阵转置
    todo!("实现transpose操作")
}

/// 练习5：广播机制
/// 将向量广播成矩阵进行运算
fn broadcast_and_add(vector: &Tensor, matrix: &Tensor) -> Result<Tensor> {
    // TODO: 实现广播加法
    todo!("实现广播运算")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;
        let reshaped = reshape_matrix(&t)?;
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(
            reshaped.to_vec2::<f32>()?,
            vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![5.0, 6.0],
            ]
        );
        Ok(())
    }

    #[test]
    fn test_add_dimension() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
        let expanded = add_dimension(&t, 0)?;
        assert_eq!(expanded.shape().dims(), &[1, 3]);
        assert_eq!(
            expanded.to_vec2::<f32>()?,
            vec![vec![1.0, 2.0, 3.0]]
        );
        Ok(())
    }

    #[test]
    fn test_remove_dimension() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[[1.0f32], [2.0], [3.0]], &device)?;
        let squeezed = remove_dimension(&t, 1)?;
        assert_eq!(squeezed.shape().dims(), &[3]);
        assert_eq!(
            squeezed.to_vec1::<f32>()?,
            vec![1.0, 2.0, 3.0]
        );
        Ok(())
    }

    #[test]
    fn test_transpose() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let transposed = transpose_matrix(&t)?;
        assert_eq!(
            transposed.to_vec2::<f32>()?,
            vec![vec![1.0, 3.0], vec![2.0, 4.0]]
        );
        Ok(())
    }

    #[test]
    fn test_broadcast() -> Result<()> {
        let device = Device::Cpu;
        let vector = Tensor::new(&[1.0f32, 2.0], &device)?;
        let matrix = Tensor::new(&[[3.0f32, 4.0], [5.0, 6.0]], &device)?;
        let result = broadcast_and_add(&vector, &matrix)?;
        assert_eq!(
            result.to_vec2::<f32>()?,
            vec![vec![4.0, 6.0], vec![6.0, 8.0]]
        );
        Ok(())
    }
}

// 提示：
// - reshape操作使用reshape()方法
// - unsqueeze操作可以在指定维度添加大小为1的维度
// - squeeze操作可以移除大小为1的维度
// - transpose操作使用transpose()方法
// - 广播操作会自动处理兼容的维度
// - 确保在操作前检查维度的兼容性
