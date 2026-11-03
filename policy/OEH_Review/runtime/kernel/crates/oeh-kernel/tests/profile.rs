use oeh_kernel::profile::{
    compose, load_profile, load_profile_with_home, EntryRow, PatchOp, ProfileError, ProfileFile,
};
use serde_json::json;

fn row(id: &str) -> EntryRow {
    EntryRow::new(id)
}

#[test]
fn bundles_concatenate_in_order_then_patches_apply() {
    let base = vec![
        row("clock"),
        row("world-log").with_config(json!({"dir": "a", "segment_mb": 64})),
    ];
    let sim = vec![row("mock-stream").with_config(json!({"hz": 10}))];
    let layer = vec![
        PatchOp::Replace {
            id: "world-log".into(),
            config: Some(json!({"dir": "b"})),
            disabled: None,
        },
        PatchOp::Insert {
            insert: vec![row("frame-counter")],
        },
        PatchOp::Replace {
            id: "mock-stream".into(),
            config: None,
            disabled: Some(true),
        },
    ];
    let rows = compose(&[("base".into(), base), ("sim".into(), sim)], &[layer]).unwrap();
    let keys: Vec<_> = rows.iter().map(|r| r.key().to_string()).collect();
    assert_eq!(
        keys,
        vec!["clock", "world-log", "mock-stream", "frame-counter"]
    );
    // 整块替换，不是 deep merge：segment_mb 消失
    assert_eq!(rows[1].config, json!({"dir": "b"}));
    assert!(rows[2].disabled);
    assert_eq!(
        rows[2].config,
        json!({"hz": 10}),
        "disabled patch keeps config"
    );
}

#[test]
fn patch_targets_by_name_when_present() {
    let base = vec![
        row("brain-native").with_name("fast"),
        row("brain-native").with_name("slow"),
    ];
    let layer = vec![PatchOp::Replace {
        id: "slow".into(),
        config: Some(json!({"x": 1})),
        disabled: None,
    }];
    let rows = compose(&[("base".into(), base)], &[layer]).unwrap();
    assert_eq!(rows[1].config, json!({"x": 1}));
    assert_eq!(rows[0].config, serde_json::Value::Null);
    // 按 id 打两个同 id 行 → 歧义
    let layer = vec![PatchOp::Replace {
        id: "brain-native".into(),
        config: None,
        disabled: Some(true),
    }];
    let err = compose(
        &[(
            "base".into(),
            vec![
                row("brain-native").with_name("fast"),
                row("brain-native").with_name("slow"),
            ],
        )],
        &[layer],
    )
    .unwrap_err();
    assert!(matches!(err, ProfileError::PatchTargetAmbiguous(_)));
}

#[test]
fn single_named_row_is_still_addressable_by_raw_id() {
    let base = vec![row("brain-native").with_name("fast")];
    let layer = vec![PatchOp::Replace {
        id: "brain-native".into(),
        config: Some(json!({"v": 1})),
        disabled: None,
    }];
    let rows = compose(&[("base".into(), base)], &[layer]).unwrap();
    assert_eq!(rows[0].config, json!({"v": 1}));
}

#[test]
fn missing_patch_target_is_an_error() {
    let err = compose(
        &[("base".into(), vec![row("clock")])],
        &[vec![PatchOp::Replace {
            id: "nope".into(),
            config: None,
            disabled: Some(true),
        }]],
    )
    .unwrap_err();
    assert!(matches!(err, ProfileError::PatchTargetNotFound(ref s) if s == "nope"));
}

#[test]
fn later_layers_can_patch_rows_inserted_by_earlier_layers() {
    let l1 = vec![PatchOp::Insert {
        insert: vec![row("late")],
    }];
    let l2 = vec![PatchOp::Replace {
        id: "late".into(),
        config: Some(json!({"v": 2})),
        disabled: None,
    }];
    let rows = compose(&[], &[l1, l2]).unwrap();
    assert_eq!(rows[0].config, json!({"v": 2}));
}

#[test]
fn yaml_shapes_parse() {
    let bundle: Vec<EntryRow> =
        serde_yaml_ng::from_str("- { id: clock }\n- id: world-log\n  config: { dir: ./logs }\n")
            .unwrap();
    assert_eq!(bundle[1].config, json!({"dir": "./logs"}));
    let profile: ProfileFile = serde_yaml_ng::from_str("bundles: [base]\npatch:\n  - { id: world-log, config: { dir: x } }\n  - insert:\n      - { id: extra }\n  - { id: extra, disabled: true }\n").unwrap();
    assert_eq!(profile.bundles, vec!["base"]);
    assert_eq!(profile.patch.len(), 3);
    assert!(matches!(profile.patch[1], PatchOp::Insert { .. }));
}

#[test]
fn load_profile_layers_files_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let bundles = dir.path().join("bundles");
    std::fs::create_dir_all(&bundles).unwrap();
    std::fs::write(
        bundles.join("base.yaml"),
        "- { id: clock }\n- { id: world-log, config: { dir: a } }\n",
    )
    .unwrap();
    let profile = dir.path().join("p.yaml");
    std::fs::write(
        &profile,
        "bundles: [base]\npatch:\n  - { id: world-log, config: { dir: b } }\n",
    )
    .unwrap();
    let extra = dir.path().join("extra.yaml");
    std::fs::write(&extra, "- { id: world-log, config: { dir: c } }\n").unwrap();

    // Create a real home dir with patch.yaml
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(
        home.join("patch.yaml"),
        "- { id: world-log, config: { dir: home } }\n",
    )
    .unwrap();

    // Test 1: with oeh_home, no extra → home layer wins
    let rows = load_profile_with_home(&profile, &bundles, Some(&home), &[]).unwrap();
    assert_eq!(rows[1].config, json!({"dir": "home"}));

    // Test 2: with oeh_home AND extra → extra layer wins
    let rows = load_profile_with_home(
        &profile,
        &bundles,
        Some(&home),
        std::slice::from_ref(&extra),
    )
    .unwrap();
    assert_eq!(rows[1].config, json!({"dir": "c"}));

    // Test 3: no oeh_home, no extra → profile patch wins
    let rows = load_profile_with_home(&profile, &bundles, None, &[]).unwrap();
    assert_eq!(rows[1].config, json!({"dir": "b"}));

    let err = load_profile_with_home(&profile, &dir.path().join("nowhere"), None, &[]).unwrap_err();
    assert!(matches!(err, ProfileError::BundleNotFound(_)));
}

#[test]
fn load_profile_reads_env_wrapper_without_home() {
    let dir = tempfile::tempdir().unwrap();
    let bundles = dir.path().join("bundles");
    std::fs::create_dir_all(&bundles).unwrap();
    std::fs::write(
        bundles.join("base.yaml"),
        "- { id: clock }\n- { id: world-log, config: { dir: a } }\n",
    )
    .unwrap();
    let profile = dir.path().join("p.yaml");
    std::fs::write(
        &profile,
        "bundles: [base]\npatch:\n  - { id: world-log, config: { dir: b } }\n",
    )
    .unwrap();

    let rows = load_profile(&profile, &bundles, &[]).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].id, "clock");
}

/// 仓库里**真的**那三份 profile：`recover` 必须在 `rule-whitelist.allowed` 里。
///
/// 这一条是配置级的回归点，不是风格检查。`recover` 是
/// `EMERGENCY_STOP → RECOVERING → STANDING` 这条边唯一的驱动者（蒸馏器的 `fsm_driver`
/// 只驱动 `FAULT → RECOVERING`），而 `rule-whitelist` 对不在表里的名字是硬拒。少了它，
/// 一次急停就是一扇单向门：删掉 `./run/estop` 只会让 `rule-estop` 和 estop 寄存器放行，
/// FSM 原地不动，`rule-fsm` 从此拒掉每一条 motion，直到进程重启。三份 profile 各写一遍
/// 公共内核（"改一处要改三处"是那三个文件自己的选择），所以这里也三份都查。
///
/// 顺带把 `world-log.retain` 也查了：默认 4096 在这套 profile 上只够 ~5.7 分钟
/// （视觉 sidecar 12 Hz 一条 `obs/detections`），一个闲久了的脑每次醒来都要吃一条
/// `brain/cursor{reset:true}`。两条断言查的是同一件事——这些数字是**写出来的决定**。
#[test]
fn the_shipped_g1_profiles_allow_recover_and_size_the_world_log_ring() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/oeh-kernel 的上两级是仓库根")
        .to_path_buf();
    let bundles = repo.join("bundles");
    for name in ["g1-mock", "g1-sim", "g1-real"] {
        let rows = load_profile(&repo.join(format!("profiles/{name}.yaml")), &bundles, &[])
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        let wl = rows
            .iter()
            .find(|r| r.id == "rule-whitelist")
            .unwrap_or_else(|| panic!("{name} 里没有 rule-whitelist"));
        let allowed = wl.config["allowed"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: rule-whitelist.allowed 不是数组"));
        assert!(
            allowed.iter().any(|v| v == "recover"),
            "{name}: 急停之后没有出路（allowed = {allowed:?}）"
        );

        let log = rows
            .iter()
            .find(|r| r.id == "world-log")
            .unwrap_or_else(|| panic!("{name} 里没有 world-log"));
        let retain = log.config["retain"].as_u64().unwrap_or(0);
        assert!(
            retain >= 32768,
            "{name}: world-log.retain = {retain}，覆盖不了 12 Hz 的视觉产出"
        );
    }
}
