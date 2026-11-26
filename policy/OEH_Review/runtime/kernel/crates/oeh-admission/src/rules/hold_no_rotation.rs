//! `rule-hold-no-rotation`：持物时禁止转腕，并把单次平移夹到 3 cm。
//!
//! 设计文档 §1.3 的那一行证据：持物期间 4/4 次转腕都把闹钟甩了出去；而单次超过 3 cm
//! 的平移在 25 Hz 下同样是一次加速度事件。前者没有"小一点就行"的版本，所以是 Deny；
//! 后者有，所以是 clamp。
//!
//! `holding.left` / `holding.right` 上 **`null` 的含义是"不知道"**（fail-closed，Deny），
//! 不是 `rule-scene` 的 `Dist::Far` 那种"看不见 ⇒ 无穷远 ⇒ 放行"；`eef_position.*` /
//! `eef_quaternion.*` 同理，读不出来就不许动。
//!
//! # 两遍结构是承重的，不要"化简"
//!
//! `evaluate_chain` 只在**入参判定已经是 Deny** 的时候才撤销一条规则对 params 的改动
//! （`verdict.rs` 的 `incoming_was_deny` 分支）。也就是说：从 `Allow` 进来、先夹了参数
//! 再自己 Deny 的规则，那次改动**会被保留下来并记成 `act/clamped`**——日志里于是出现
//! 一条"这一步被夹到 3 cm"的记录，而这一步其实根本没有跑。
//!
//! 所以这条规则先跑完整的第一遍（只读，任何一处越界或读不出来就当场 Deny、params 一个
//! 字节不动），确认没有 Deny 之后才跑第二遍（只写，把超出的平移缩到 3 cm）。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::admission::Admission;
use crate::capability::CapabilityClass;
use crate::rules::deny;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-hold-no-rotation";

const ARMS: [&str; 2] = ["left", "right"];
const DELTA_KEYS: [&str; 3] = ["dx", "dy", "dz"];

fn d_caps() -> Vec<String> {
    ["arx_x5_move_eef", "arx_x5_move_relative"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}
fn d_max_rot_rad() -> f64 {
    0.05
}
fn d_max_translation_m() -> f64 {
    0.03
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_caps")]
    caps: Vec<String>,
    #[serde(default = "d_max_rot_rad")]
    max_rot_rad: f64,
    #[serde(default = "d_max_translation_m")]
    max_translation_m: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            caps: d_caps(),
            max_rot_rad: d_max_rot_rad(),
            max_translation_m: d_max_translation_m(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HoldNoRotationRule {
    pub caps: Vec<String>,
    pub max_rot_rad: f64,
    pub max_translation_m: f64,
}

impl Default for HoldNoRotationRule {
    fn default() -> Self {
        Config::default().into()
    }
}

impl From<Config> for HoldNoRotationRule {
    fn from(c: Config) -> Self {
        Self {
            caps: c.caps,
            max_rot_rad: c.max_rot_rad,
            max_translation_m: c.max_translation_m,
        }
    }
}

/// `Ok(true)` = 至少一只手夹着东西，`Ok(false)` = 两只手都空着，`Err` = 不知道。
///
/// `ctx.value` 对不存在的键返回 `None`；`Value::Null` 落到 `_` 分支，因此同样是"不知道"。
fn any_holding(ctx: &RuleCtx) -> Result<bool, String> {
    let mut any = false;
    for arm in ARMS {
        let key = format!("holding.{arm}");
        match ctx.value(&key) {
            Some(Value::Bool(b)) => any |= *b,
            _ => return Err(format!("{key} is unknown")),
        }
    }
    Ok(any)
}

// ---------------------------------------------------------------- 数值读法

/// 有限数，别的一律"读不出来"。注意 `json!(f64::NAN)` 序列化成 `null`，走的也是这条路。
fn finite(v: &Value) -> Option<f64> {
    v.as_f64().filter(|d| d.is_finite())
}

/// 长度恰好 `N` 的有限数数组。长度不对、有一项不是有限数，都读不出来。
fn vec_n<const N: usize>(v: &Value) -> Option<[f64; N]> {
    let a = v.as_array()?;
    if a.len() != N {
        return None;
    }
    let mut out = [0.0f64; N];
    for (slot, item) in out.iter_mut().zip(a) {
        *slot = finite(item)?;
    }
    Some(out)
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ------------------------------------------------- 四元数（wxyz，无 scipy 对应实现）

fn q_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn q_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// 单位化；`‖q‖ < 1e-6` 或不是有限数一律读不出来。
fn q_unit(q: [f64; 4]) -> Option<[f64; 4]> {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if !n.is_finite() || n < 1e-6 {
        return None;
    }
    Some([q[0] / n, q[1] / n, q[2] / n, q[3] / n])
}

/// 两个单位四元数之间的夹角，落在 `[0, π]`：`2*atan2(‖r[1:]‖, |r[0]|)`。
fn q_angle(a: [f64; 4], b: [f64; 4]) -> f64 {
    let r = q_mul(a, q_conj(b));
    let xyz = (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
    2.0 * xyz.atan2(r[0].abs())
}

// ---------------------------------------------------------------- 第一遍的产物

/// 第二遍要把夹过的平移写回哪里。
enum WriteBack {
    /// 这条臂没给平移（比如只改了 `gripper`）：没什么可写的。
    None,
    /// `move_relative`：写回脑**实际用过的**那几个键（下标 + 键名），不补它没写的。
    Relative(Vec<(usize, &'static str)>),
    /// `move_eef`：写回 `position = 当前 EEF + 夹过的位移`。
    Absolute([f64; 3]),
}

struct ArmPlan {
    arm: &'static str,
    rotation_rad: f64,
    translation: [f64; 3],
    write: WriteBack,
}

/// 只读地把一条臂的参数解出来。任何读不出来的字段都是 `Err`（调用方据此 Deny）。
fn plan_arm(arm: &'static str, spec: &Value, ctx: &RuleCtx) -> Result<ArmPlan, String> {
    let obj = spec
        .as_object()
        .ok_or_else(|| format!("{arm} is not an object"))?;

    // --- 旋转量：相对的 drot_rotvec / dyaw_deg，绝对的 quaternion_wxyz。
    let mut rotation_rad = 0.0f64;
    if let Some(v) = obj.get("drot_rotvec") {
        let r: [f64; 3] = vec_n(v).ok_or_else(|| format!("{arm}.drot_rotvec is unreadable"))?;
        rotation_rad += norm3(r);
    }
    if let Some(v) = obj.get("dyaw_deg") {
        let d = finite(v).ok_or_else(|| format!("{arm}.dyaw_deg is unreadable"))?;
        rotation_rad += d.abs().to_radians();
    }
    if let Some(v) = obj.get("quaternion_wxyz") {
        let target = vec_n::<4>(v)
            .and_then(q_unit)
            .ok_or_else(|| format!("{arm}.quaternion_wxyz is unreadable"))?;
        let key = format!("eef_quaternion.{arm}");
        let current = ctx
            .value(&key)
            .and_then(vec_n::<4>)
            .and_then(q_unit)
            .ok_or_else(|| format!("{key} is unknown"))?;
        rotation_rad += q_angle(target, current);
    }

    // --- 平移量。`position` 是绝对目标，需要当前 EEF 才能算出这一步走多远。
    //
    // 先挡住"两种写法同时出现"这一种含糊形状。旧实现把 `position` 与 `dx/dy/dz` 当作
    // 二选一（有 `position` 就走绝对分支，`else` 才读增量），于是一个同时带两者的
    // params 会让增量**完全不被读到、不被量、也不被夹**；而 `arx_x5_move_relative` 的
    // 执行体 `_relative_target` 恰恰相反——它只读 `dx/dy/dz`，完全无视 `position`。
    // 两边对同一个对象的理解相反，中间又没有人拒绝这种形状（per-arm 子 schema 当时没有
    // `additionalProperties: false`，而校验器对缺省值按 true 处理），结果就是
    // `{"left": {"position": <当前 EEF>, "dx": 0.9}}` 在持物时被判 Allow，然后被当成
    // 一步 0.9 m 的平移执行。这里当场 Deny（params 一个字节不动）是"读不出来就不许动"
    // 的同一条纪律：形状本身有歧义，就没有"夹到 3 cm"的正确答案可言。
    //
    // 词汇关那一层也补了同一个洞（`arx_plugin.py` 的 `_ARM_ABS` / `_ARM_REL` 现在都是
    // `additionalProperties: false`），所以正常路径上这种 params 在 `stage:"schema"`
    // 就被拒了；这条规则是第二道，防的是"有人改了 schema 却没改规则"。
    let mixed: Vec<&str> = DELTA_KEYS
        .iter()
        .copied()
        .filter(|k| obj.contains_key(*k))
        .collect();
    if obj.contains_key("position") && !mixed.is_empty() {
        return Err(format!(
            "{arm} mixes absolute position with relative deltas ({})",
            mixed.join(", ")
        ));
    }

    let mut translation = [0.0f64; 3];
    let mut write = WriteBack::None;
    if let Some(v) = obj.get("position") {
        let p: [f64; 3] = vec_n(v).ok_or_else(|| format!("{arm}.position is unreadable"))?;
        let key = format!("eef_position.{arm}");
        let base: [f64; 3] = ctx
            .value(&key)
            .and_then(vec_n::<3>)
            .ok_or_else(|| format!("{key} is unknown"))?;
        translation = [p[0] - base[0], p[1] - base[1], p[2] - base[2]];
        write = WriteBack::Absolute(base);
    } else {
        let mut keys: Vec<(usize, &'static str)> = Vec::new();
        for (i, k) in DELTA_KEYS.iter().enumerate() {
            if let Some(v) = obj.get(*k) {
                translation[i] = finite(v).ok_or_else(|| format!("{arm}.{k} is unreadable"))?;
                keys.push((i, k));
            }
        }
        if !keys.is_empty() {
            write = WriteBack::Relative(keys);
        }
    }

    Ok(ArmPlan {
        arm,
        rotation_rad,
        translation,
        write,
    })
}

fn set_number(obj: &mut serde_json::Map<String, Value>, key: &str, v: f64) {
    if let Some(n) = serde_json::Number::from_f64(v) {
        obj.insert(key.to_string(), Value::Number(n));
    }
}

impl Rule for HoldNoRotationRule {
    fn name(&self) -> &str {
        NAME
    }

    fn evaluate(&self, params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        self.evaluate_detailed(params, ctx, incoming).0
    }

    fn evaluate_detailed(
        &self,
        params: &mut Value,
        ctx: &RuleCtx,
        incoming: Verdict,
    ) -> (Verdict, Option<Value>) {
        if ctx.spec.class != CapabilityClass::Motion || !self.caps.contains(&ctx.spec.name) {
            return (incoming, None);
        }
        match any_holding(ctx) {
            Err(why) => return (deny(incoming, format!("{NAME}: {why}")), None),
            Ok(false) => return (incoming, None),
            Ok(true) => {}
        }

        // ---- 第一遍：只读。任何越界或读不出来 ⇒ 当场 Deny，params 一个字节不动。
        let mut plans: Vec<ArmPlan> = Vec::with_capacity(ARMS.len());
        for arm in ARMS {
            let Some(spec) = params.get(arm) else {
                continue;
            };
            if spec.is_null() {
                // schema 里 `type: ["object","null"]` 是数组，`check_type` 会跳过它，但
                // `properties` 让 `is_object_schema` 为真，于是 `null` 在词汇校验那一关
                // 就被拒了。真走到这里只能说明上游放了个不该放的值——当未知处理。
                return (
                    deny(incoming, format!("{NAME}: {arm} is null (unknown target)")),
                    None,
                );
            }
            match plan_arm(arm, spec, ctx) {
                Ok(p) => plans.push(p),
                Err(why) => return (deny(incoming, format!("{NAME}: {why}")), None),
            }
        }

        let detail = json!({"arms": plans.iter().map(|p| json!({
            "arm": p.arm,
            "rotation_rad": p.rotation_rad,
            "translation_m": norm3(p.translation),
        })).collect::<Vec<_>>()});

        for p in &plans {
            if p.rotation_rad > self.max_rot_rad {
                return (
                    deny(
                        incoming,
                        format!(
                            "{NAME}: {} rotation {:.4} rad exceeds {} rad while holding",
                            p.arm, p.rotation_rad, self.max_rot_rad
                        ),
                    ),
                    Some(detail.clone()),
                );
            }
        }

        // ---- 第二遍：只写。到这里判定已经确定不会被这条规则收紧。
        for p in &plans {
            let d = norm3(p.translation);
            if d <= self.max_translation_m {
                continue;
            }
            let s = self.max_translation_m / d;
            let scaled = [
                p.translation[0] * s,
                p.translation[1] * s,
                p.translation[2] * s,
            ];
            let Some(arm_obj) = params.get_mut(p.arm).and_then(|v| v.as_object_mut()) else {
                continue;
            };
            match &p.write {
                WriteBack::None => {}
                WriteBack::Relative(keys) => {
                    for (i, k) in keys {
                        set_number(arm_obj, k, scaled[*i]);
                    }
                }
                WriteBack::Absolute(base) => {
                    let pos = vec![
                        json!(base[0] + scaled[0]),
                        json!(base[1] + scaled[1]),
                        json!(base[2] + scaled[2]),
                    ];
                    arm_obj.insert("position".into(), Value::Array(pos));
                }
            }
        }

        (incoming, Some(detail))
    }
}

pub struct HoldNoRotationRulePlugin;

impl Plugin for HoldNoRotationRulePlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(NAME).inject(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let adm = ctx.get::<Admission>("admission")?;
        ctx.on_dispose(adm.register_rule(Arc::new(HoldNoRotationRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(HoldNoRotationRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_carry_thresholds_from_the_lessons() {
        let r = HoldNoRotationRule::default();
        assert_eq!(r.caps, vec!["arx_x5_move_eef", "arx_x5_move_relative"]);
        assert!((r.max_rot_rad - 0.05).abs() < 1e-12);
        assert!((r.max_translation_m - 0.03).abs() < 1e-12);
    }

    #[test]
    fn a_partial_config_keeps_the_other_defaults() {
        let c: Config = serde_json::from_value(json!({"max_rot_rad": 0.01})).unwrap();
        assert!((c.max_rot_rad - 0.01).abs() < 1e-12);
        assert!((c.max_translation_m - 0.03).abs() < 1e-12);
        assert_eq!(c.caps, d_caps());
    }

    #[test]
    fn a_typo_is_a_boot_error_not_a_silently_ignored_key() {
        assert!(serde_json::from_value::<Config>(json!({"max_rot": 0.05})).is_err());
        assert!(serde_json::from_value::<Config>(json!({"max_translation": 0.03})).is_err());
    }

    #[test]
    fn quaternion_angle_is_the_relative_rotation() {
        let identity = [1.0, 0.0, 0.0, 0.0];
        assert!(q_angle(identity, identity).abs() < 1e-12);
        // 绕 z 转 90°。
        let h = std::f64::consts::FRAC_PI_4;
        let z90 = [h.cos(), 0.0, 0.0, h.sin()];
        assert!((q_angle(z90, identity) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        // 符号相反的同一个旋转（q 与 -q）夹角仍是 0。
        let neg = [-z90[0], -z90[1], -z90[2], -z90[3]];
        assert!(q_angle(z90, neg).abs() < 1e-9);
    }

    #[test]
    fn nan_reads_as_unreadable() {
        assert_eq!(finite(&json!(f64::NAN)), None, "NaN 序列化成 null");
        assert_eq!(finite(&json!("0.1")), None);
        assert_eq!(vec_n::<3>(&json!([1.0, 2.0])), None, "长度不对");
        assert_eq!(vec_n::<3>(&json!([1.0, 2.0, "x"])), None);
        assert_eq!(q_unit([0.0, 0.0, 0.0, 0.0]), None);
    }
}
