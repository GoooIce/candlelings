use candle_core::{Device, Result, Tensor, DType};

/// 练习1：创建标量张量
/// 创建一个包含单个浮点数值的标量张量
fn create_scalar() -> Result<Tensor> {
    // TODO: 创建一个包含值42.0的标量张量
    todo!("创建一个标量张量")
}

/// 练习2：创建向量张量
/// 创建一个一维张量（向量）
fn create_vector() -> Result<Tensor> {
    // TODO: 创建一个包含[1.0, 2.0, 3.0, 4.0, 5.0]的向量张量
    todo!("创建一个向量张量")
}

/// 练习3：创建矩阵张量
/// 创建一个二维张量（矩阵）
fn create_matrix() -> Result<Tensor> {
    // TODO: 创建一个2x3的矩阵张量，包含[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    todo!("创建一个矩阵张量")
}

/// 练习4：获取张量属性
/// 检查张量的形状、数据类型等属性
fn check_tensor_properties(t: &Tensor) -> Result<(Vec<usize>, DType)> {
    // TODO: 返回张量的形状和数据类型
    todo!("获取并返回张量属性")
}

/// 练习5：创建指定设备上的张量
/// 在特定设备（CPU/CUDA）上创建张量
fn create_tensor_on_device(device: &Device) -> Result<Tensor> {
    // TODO: 在指定设备上创建一个2x2的张量[[1.0, 2.0], [3.0, 4.0]]
    todo!("在指定设备上创建张量")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar() -> Result<()> {
        let s = create_scalar()?;
        assert_eq!(s.shape().dims().len(), 0); // 标量张量维度为0
        assert_eq!(s.to_vec0::<f32>()?, 42.0);
        Ok(())
    }

    #[test]
    fn test_vector() -> Result<()> {
        let v = create_vector()?;
        assert_eq!(v.shape().dims(), &[5]); // 一维，长度为5
        assert_eq!(v.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_matrix() -> Result<()> {
        let m = create_matrix()?;
        assert_eq!(m.shape().dims(), &[2, 3]); // 2x3矩阵
        assert_eq!(
            m.to_vec2::<f32>()?,
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
        );
        Ok(())
    }

    #[test]
    fn test_properties() -> Result<()> {
        let t = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;
        let (shape, dtype) = check_tensor_properties(&t)?;
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(dtype, DType::F32);
        Ok(())
    }

    #[test]
    fn test_device() -> Result<()> {
        let device = Device::Cpu;
        let t = create_tensor_on_device(&device)?;
        assert_eq!(t.shape().dims(), &[2, 2]);
        assert_eq!(
            t.to_vec2::<f32>()?,
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
        Ok(())
    }
}

// 提示：这些练习旨在帮助你熟悉Candle的张量API
// - 使用Tensor::new()创建新张量
// - 使用适当的数据类型（如f32）
// - 注意处理Result类型
// - 使用to_vec0/to_vec1/to_vec2等方法验证结果
// - 记住检查张量的设备位置
