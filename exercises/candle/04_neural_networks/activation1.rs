use candle_core::{Result, Tensor, Module};

/// 练习1：ReLU激活函数
/// 实现ReLU的前向传播
#[derive(Debug)]
struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: 实现ReLU
        // f(x) = max(0, x)
        todo!("实现ReLU激活函数")
    }
}

/// 练习2：Sigmoid激活函数
/// 实现Sigmoid的前向传播
#[derive(Debug)]
struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: 实现Sigmoid
        // f(x) = 1 / (1 + exp(-x))
        todo!("实现Sigmoid激活函数")
    }
}

/// 练习3：Tanh激活函数
/// 实现Tanh的前向传播
#[derive(Debug)]
struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: 实现Tanh
        // f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        todo!("实现Tanh激活函数")
    }
}

/// 练习4：LeakyReLU激活函数
/// 实现带有负斜率的ReLU
#[derive(Debug)]
struct LeakyReLU {
    negative_slope: f32,
}

impl LeakyReLU {
    fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: 实现LeakyReLU
        // f(x) = x if x > 0 else negative_slope * x
        todo!("实现LeakyReLU激活函数")
    }
}

/// 练习5：自定义激活函数
/// 实现一个可以自定义函数的激活层
#[derive(Debug)]
struct CustomActivation<F>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    activation_fn: F,
}

impl<F> CustomActivation<F>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    fn new(activation_fn: F) -> Self {
        Self { activation_fn }
    }
}

impl<F> Module for CustomActivation<F>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: 使用自定义激活函数
        todo!("实现自定义激活函数")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_relu() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &device)?;
        let relu = ReLU;
        let output = relu.forward(&input)?;
        assert_eq!(output.to_vec1::<f32>()?, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_sigmoid() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[0.0f32], &device)?;
        let sigmoid = Sigmoid;
        let output = sigmoid.forward(&input)?;
        assert!((output.to_vec0::<f32>()? - 0.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_tanh() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[0.0f32], &device)?;
        let tanh = Tanh;
        let output = tanh.forward(&input)?;
        assert!(output.to_vec0::<f32>()?.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_leaky_relu() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[-1.0f32, 1.0], &device)?;
        let leaky_relu = LeakyReLU::new(0.1);
        let output = leaky_relu.forward(&input)?;
        assert_eq!(output.to_vec1::<f32>()?, vec![-0.1, 1.0]);
        Ok(())
    }

    #[test]
    fn test_custom_activation() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
        
        // 创建一个平方激活函数
        let square_activation = CustomActivation::new(|x: &Tensor| x.mul(x));
        let output = square_activation.forward(&input)?;
        
        assert_eq!(output.to_vec1::<f32>()?, vec![1.0, 4.0, 9.0]);
        Ok(())
    }
}

// 提示：
// - 使用Tensor的element_wise操作
// - 注意数值稳定性
// - 合理处理边界情况
// - 优化计算性能
// - 确保梯度计算正确
