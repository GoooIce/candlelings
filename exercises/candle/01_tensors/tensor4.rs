use candle_core::{Device, DType, Result, Tensor};

/// 练习1：创建CUDA张量
/// 在GPU上创建张量（如果可用）
fn create_cuda_tensor() -> Result<Tensor> {
    // TODO: 在GPU上创建张量，如果GPU不可用则在CPU上创建
    // 提示：使用Device::cuda_if_available()
    todo!("在GPU上创建张量")
}

/// 练习2：设备间数据传输
/// 将张量在CPU和GPU之间移动
fn move_to_device(t: &Tensor, device: &Device) -> Result<Tensor> {
    // TODO: 实现设备间的张量移动
    todo!("实现设备间数据传输")
}

/// 练习3：混合精度操作
/// 在不同数据类型间转换
fn convert_dtype(t: &Tensor, dtype: DType) -> Result<Tensor> {
    // TODO: 实现数据类型转换
    todo!("实现数据类型转换")
}

/// 练习4：设备感知操作
/// 在正确的设备上执行张量操作
fn device_aware_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // TODO: 确保在相同设备上执行加法操作
    todo!("实现设备感知的加法操作")
}

/// 练习5：内存优化
/// 使用就地操作优化内存使用
fn memory_efficient_op(t: &mut Tensor) -> Result<()> {
    // TODO: 实现内存高效的操作（将张量值翻倍）
    todo!("实现内存优化操作")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_tensor() -> Result<()> {
        let t = create_cuda_tensor()?;
        let data = t.to_vec2::<f32>()?;
        assert_eq!(data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_device_transfer() -> Result<()> {
        let cpu_tensor = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu)?;
        let target_device = Device::Cpu; // 如果有CUDA则可以测试GPU
        let moved = move_to_device(&cpu_tensor, &target_device)?;
        assert_eq!(moved.to_vec2::<f32>()?, vec![vec![1.0, 2.0]]);
        Ok(())
    }

    #[test]
    fn test_dtype_conversion() -> Result<()> {
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        let converted = convert_dtype(&t, DType::F64)?;
        assert_eq!(converted.dtype(), DType::F64);
        assert_eq!(converted.to_vec1::<f64>()?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_device_aware_operation() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::new(&[[1.0f32]], &device)?;
        let b = Tensor::new(&[[2.0f32]], &device)?;
        let c = device_aware_add(&a, &b)?;
        assert_eq!(c.to_vec2::<f32>()?, vec![vec![3.0]]);
        Ok(())
    }

    #[test]
    fn test_memory_efficient() -> Result<()> {
        let device = Device::Cpu;
        let mut t = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        memory_efficient_op(&mut t)?;
        assert_eq!(
            t.to_vec2::<f32>()?,
            vec![vec![2.0, 4.0], vec![6.0, 8.0]]
        );
        Ok(())
    }
}

// 提示：
// - 使用Device::cuda_if_available()检查GPU可用性
// - 使用to_device()方法在设备间移动张量
// - 使用to_dtype()进行数据类型转换
// - 检查张量的device()以确保设备兼容性
// - 使用原位操作（in-place operations）提高内存效率
// - 处理GPU相关操作时要考虑错误处理
