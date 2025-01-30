use candle_core::{Result, Tensor, Module};
use std::vec::Vec;

/// 练习1：Sequential容器实现
/// 实现一个按顺序执行模块的容器
struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// 创建一个新的Sequential容器
    fn new() -> Self {
        // TODO: 初始化Sequential
        todo!("实现Sequential构造函数")
    }

    /// 添加一个新层
    fn add(&mut self, layer: Box<dyn Module>) {
        // TODO: 添加新层到容器
        todo!("实现层添加功能")
    }

    /// 获取指定索引的层
    fn layer(&self, index: usize) -> Option<&Box<dyn Module>> {
        // TODO: 获取指定层
        todo!("实现层访问功能")
    }
}

/// 练习2：实现Module特征
impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: 实现Sequential的前向传播
        todo!("实现Sequential前向传播")
    }
}

/// 练习3：实现迭代器
impl Sequential {
    /// 返回层的迭代器
    fn iter(&self) -> impl Iterator<Item = &Box<dyn Module>> {
        // TODO: 实现层迭代器
        todo!("实现迭代器")
    }
}

/// 练习4：构建辅助函数
impl Sequential {
    /// 使用builder模式构建网络
    fn builder() -> SequentialBuilder {
        // TODO: 实现构建器
        todo!("实现构建器模式")
    }

    /// 获取网络的总层数
    fn num_layers(&self) -> usize {
        // TODO: 获取层数
        todo!("实现层数统计")
    }
}

/// 练习5：构建器实现
struct SequentialBuilder {
    layers: Vec<Box<dyn Module>>,
}

impl SequentialBuilder {
    /// 创建新的构建器
    fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// 添加层
    fn add_layer(mut self, layer: Box<dyn Module>) -> Self {
        // TODO: 实现构建器的层添加
        todo!("实现构建器添加层")
    }

    /// 构建Sequential
    fn build(self) -> Sequential {
        // TODO: 实现构建过程
        todo!("实现构建过程")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // 创建一个简单的测试层
    struct TestLayer;
    impl Module for TestLayer {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            x.mul_scalar(2.0)
        }
    }

    #[test]
    fn test_sequential_creation() {
        let sequential = Sequential::new();
        assert_eq!(sequential.num_layers(), 0);
    }

    #[test]
    fn test_layer_addition() {
        let mut sequential = Sequential::new();
        sequential.add(Box::new(TestLayer));
        assert_eq!(sequential.num_layers(), 1);
    }

    #[test]
    fn test_forward() -> Result<()> {
        let device = Device::Cpu;
        let mut sequential = Sequential::new();
        sequential.add(Box::new(TestLayer));
        sequential.add(Box::new(TestLayer));

        let input = Tensor::new(&[1.0f32], &device)?;
        let output = sequential.forward(&input)?;
        // Input: 1.0 -> First layer: 2.0 -> Second layer: 4.0
        assert_eq!(output.to_vec0::<f32>()?, 4.0);
        Ok(())
    }

    #[test]
    fn test_layer_access() {
        let mut sequential = Sequential::new();
        sequential.add(Box::new(TestLayer));
        assert!(sequential.layer(0).is_some());
        assert!(sequential.layer(1).is_none());
    }

    #[test]
    fn test_builder() -> Result<()> {
        let device = Device::Cpu;
        let sequential = Sequential::builder()
            .add_layer(Box::new(TestLayer))
            .add_layer(Box::new(TestLayer))
            .build();

        let input = Tensor::new(&[1.0f32], &device)?;
        let output = sequential.forward(&input)?;
        assert_eq!(output.to_vec0::<f32>()?, 4.0);
        Ok(())
    }
}

// 提示：
// - 使用Box<dyn Module>存储不同类型的层
// - 正确管理层的所有权
// - 实现优雅的错误处理
// - 提供直观的构建接口
// - 考虑迭代器的实现
