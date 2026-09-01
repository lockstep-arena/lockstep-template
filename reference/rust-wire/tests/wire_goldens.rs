//! The hand-written decoder against the SPEC's published golden encodings
//! (`tests/fixtures/*.bin`, vendored alongside `docs/wire.md` from the
//! platform's interface repo). If these pass, the decoder speaks the same wire as every engine.

use lockstep_wire_reference::wire::{encode_input, f32_bytes, Dtype, SeatInit, View};

const SEAT_INIT: &[u8] = include_bytes!("fixtures/seat_init.bin");
const VIEW: &[u8] = include_bytes!("fixtures/view.bin");
const INPUT: &[u8] = include_bytes!("fixtures/input.bin");

#[test]
fn seat_init_golden_decodes() {
    let init = SeatInit::decode(SEAT_INIT).expect("golden seat-init decodes");
    assert_eq!(init.seat, 1);
    assert_eq!(init.meta("control_hz"), Some("50"));
    assert_eq!(init.meta("task"), Some("golden"));

    let obs: Vec<&str> = init.obs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(obs, ["marquee", "agent"]);
    assert_eq!(init.obs[0].dtype, Dtype::U8);
    assert_eq!(init.obs[0].shape, [1, 2, 4]);
    assert_eq!(init.obs[1].dtype, Dtype::F32);
    // Per-element bounds override the scalars where declared.
    assert_eq!(init.obs[1].bounds_at(2), (-10.0, 10.0));
    assert_eq!(
        init.obs[1].slice("joint_pos").map(|s| (s.start, s.len)),
        Some((0, 2))
    );
    // The documentation channel rides the same bytes.
    assert_eq!(
        init.obs[0].doc,
        "a 2×4 grayscale strip, one channel, top row first"
    );
    assert_eq!(
        init.obs[1].slice("joint_vel").map(|s| s.unit.as_str()),
        Some("rad/s")
    );
    assert_eq!(init.brief.goal, "Hold the pose the cue asks for.");
    assert_eq!(
        init.brief.ends,
        "After 5 ticks, or the moment the body falls."
    );

    let action = init.action("action").expect("f32 action value");
    assert_eq!(action.dtype, Dtype::F32);
    assert_eq!(action.numel(), 3);
    assert_eq!(action.slice("torque").map(|s| s.len), Some(3));
    // Neutral = midpoint of finite bounds: [-1, 1] → 0.
    assert_eq!(action.neutral_f32(), [0.0, 0.0, 0.0]);

    let mode = init.action("mode").expect("i32 action value");
    assert_eq!(mode.dtype, Dtype::I32);
    // [0, 3] → 1.5 midpoint (kept as f32; encoding truncates per dtype).
    assert_eq!(mode.neutral_f32(), [1.5]);
}

#[test]
fn view_golden_decodes_zero_copy() {
    let view = View::decode(VIEW).expect("golden view decodes");
    assert_eq!(view.tick, 42);
    assert!((view.reward - -0.125).abs() < 1e-6);
    assert!(view.done);
    assert_eq!(view.values.len(), 2);
    assert_eq!(view.values[0], [0u8, 1, 2, 3, 4, 5, 6, 255]);
    let agent: Vec<f32> = view.values[1]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(agent, [0.5, -0.5, 1.25, -1.25, 0.75]);
}

#[test]
fn input_golden_reencodes_byte_for_byte() {
    // The golden input carries action = [0.25, -0.75, 1.0], mode = [2i32].
    let encoded = encode_input(&[f32_bytes(&[0.25, -0.75, 1.0]), 2i32.to_le_bytes().to_vec()]);
    assert_eq!(encoded, INPUT, "hand encoder must match the golden bytes");
}
