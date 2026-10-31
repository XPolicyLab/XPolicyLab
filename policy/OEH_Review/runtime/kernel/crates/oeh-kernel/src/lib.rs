pub mod brains;
pub mod builtin;
pub mod clock;
pub mod factories;
pub mod fiber;
pub mod frame_store;
pub mod fsm;
mod fsutil;
pub mod kernel;
pub mod observation;
pub mod op;
pub mod plugin;
pub mod profile;
pub mod proposal;
pub mod services;
pub mod session_log;
pub mod state;
pub mod wake;
pub mod world_log;

pub use brains::{
    BrainSink, BrainStatus, Brains, BrainsConfig, BrainsError, TurnHandle, MAX_INJECT_TEXT,
};
pub use clock::{Clock, ClockService, ManualClock, MonotonicClock};
pub use factories::{factory_ids, find_factory, PluginFactory};
pub use fiber::{FiberInfo, FiberState, FiberTree, SettleReport};
pub use frame_store::{FrameError, FrameStore, StreamStats};
pub use fsm::{FsmError, FsmState, RobotFsm, Transition};
pub use kernel::{BootError, Kernel, KernelConfig};
pub use observation::{
    build_observation, CollapsedEvent, Observation, ObservationBudget, WakeReason,
};
pub use op::{FiberAdded, KernelHandle, KernelSendError, Op};
pub use plugin::{Ctx, Disposer, Effects, Plugin, PluginError};
pub use profile::EntryRow;
pub use proposal::{BoxFuture, Outcome, Proposal, ProposalId, Proposer};
pub use services::{FiberId, ServiceRegistry};
pub use session_log::{read_session, SessionEntry, SessionLog, SessionReadError};
pub use state::{StateEntry, StateRegistry};
pub use wake::{BrainCursor, Decision, WakeAction, WakeBudget, WakePolicy};
pub use world_log::{
    read_segments, read_segments_after, resolve_log_dir, CursorTooOld, JsonlSegmentSink,
    LogReadError, LogSink, WorldLog,
};
