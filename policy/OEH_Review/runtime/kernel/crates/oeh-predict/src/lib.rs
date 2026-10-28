//! 世界模型槽位：一个"如果这么动，接下来会怎样"的接口，外加一个不需要任何学习的
//! 保底实现。
//!
//! 这个 crate 只依赖内核与协议（不依赖蒸馏、不依赖准入），方向是单向的：蒸馏器
//! （`distill-surprise`）和规则（`rule-dryrun`）依赖它，它谁也不依赖。换掉预测器
//! （学出来的动力学模型、厂商的仿真器）只是换一行 profile，读它的人一行都不用动。
pub mod kinematic;
pub mod predictor;

pub use kinematic::KinematicPredictor;
pub use predictor::{
    state_ref_from, ActionStep, PredictedState, Prediction, Predictor, PredictorService, StateRef,
};
