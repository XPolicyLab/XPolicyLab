//! 编译进内核的准入侧插件：`admission`（服务本体 + 四个内置能力）、`approval-auto`
//! （无人值守审批）、`mock-executor`（测试用执行器）。
pub mod admission;
pub mod approval_auto;
pub mod mock_executor;
