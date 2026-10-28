//! `distill-estop`：把"急停标志文件在不在"蒸馏成世界里的一个事实。
//!
//! 急停的权威在内核外面（独立的 `estop-listener` 进程：ESC → 写标志文件 → 直接给机器人
//! 发零力矩）。内核挂了不影响急停，所以这里只做两件事：把状态写进 `estop` 寄存器（观测
//! 构造读它），以及在**边沿**上发 `safety/estop{engaged}`——脑的唤醒策略把这条事件映射
//! 成 `interrupt`（见 `wake/fast.yaml`），所以它必须是边沿而不是每个周期一条，否则正在
//! 思考的脑会被自己的心跳反复打断。
//!
//! 起步假设是"没按下"：第一个周期发现文件不存在不发事件（那不是边沿），发现文件已经
//! 存在则立刻发一条 `engaged: true`——内核在急停期间启动，脑第一眼就该知道。
//!
//! 准入服务取快照时把这里的 `estop` 寄存器和它自己那个标志文件路径取**或**
//! （`Admission::estop_engaged`），`rule-estop` 读的是那个合并结果：两条路各有各的默认
//! 路径，一份只改了其中一个的 profile 不该悄悄把急停这条链解掉——fail-closed 的方向是
//! "任一说按下了就按下了"。寄存器最多晚 `period_ms`，但它只会让判定更紧，不会更松。
//!
//! 准入服务还会订阅上面那条 `safety/estop{engaged}`：按下的那一刻，**正在跑**的运动被
//! 当场撤销（`act/cancelled{reason:"estop"}`），而不是只挡住后续提案。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Ctx, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::json;

const SOURCE: &str = "distill-estop";

fn d_flag_path() -> PathBuf {
    PathBuf::from("/tmp/oeh_estop")
}
fn d_period() -> u64 {
    50
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_flag_path")]
    flag_path: PathBuf,
    #[serde(default = "d_period")]
    period_ms: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            flag_path: d_flag_path(),
            period_ms: d_period(),
        }
    }
}

pub struct EstopPlugin;
impl Plugin for EstopPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(SOURCE).inject(&["clock", "world-log", "state"])
    }
    fn apply(&self, ctx: &Ctx, config: &serde_json::Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let kind = EventKind::new("safety/estop").expect("static");
        let flag = cfg.flag_path;
        let shown = flag.display().to_string();
        // 上一次看到的状态。起步按"没按下"算，见模块文档。
        let last = Arc::new(Mutex::new(false));
        let period = Duration::from_millis(cfg.period_ms.clamp(1, 60_000));
        ctx.spawn_interval(period, move || {
            let engaged = flag.exists();
            state.set("estop", json!(engaged), SOURCE);
            // 边沿判断和记录新状态都在锁内一次做完；`append` 在锁外调用——它会同步跑
            // 世界日志的全部观察者，不能带着这把私有锁跨进去。
            let edge = {
                let mut g = last.lock();
                let edge = *g != engaged;
                *g = engaged;
                edge
            };
            if edge {
                log.append(
                    EventDraft::new(
                        kind.clone(),
                        SOURCE,
                        json!({"engaged": engaged, "flag_path": shown}),
                    )
                    .with_salience(255)
                    .with_collapse_key("estop"),
                );
            }
        });
        Ok(())
    }
}
inventory::submit! { PluginFactory { id: SOURCE, build: || Box::new(EstopPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_listener_contract() {
        let c = Config::default();
        assert_eq!(c.flag_path, PathBuf::from("/tmp/oeh_estop"));
        assert_eq!(c.period_ms, 50);
        let empty: Config = serde_json::from_value(json!({})).unwrap();
        assert_eq!(
            (empty.flag_path, empty.period_ms),
            (c.flag_path, c.period_ms)
        );
        assert!(serde_json::from_value::<Config>(json!({"flag": "/tmp/x"})).is_err());
    }
}
