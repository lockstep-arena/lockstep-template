//! The hand-written decoder against the SPEC's published golden encodings
//! (`tests/fixtures/*.bin`, vendored alongside `docs/wire.md` from the
//! platform's interface repo). If these pass, the decoder speaks the same wire as every engine.

use lockstep_wire_reference::wire::{
    encode_input, f32_bytes, Dtype, MetricDirection, MetricKind, SeatInit, View,
};

const SEAT_INIT: &[u8] = include_bytes!("fixtures/seat_init.bin");
const SEAT_INIT_NOTAIL: &[u8] = include_bytes!("fixtures/seat_init_notail.bin");
const VIEW: &[u8] = include_bytes!("fixtures/view.bin");
const INPUT: &[u8] = include_bytes!("fixtures/input.bin");

/// Offset of the first `ValueSpec` region's `len` prefix: magic(4)
/// version(4) seat(4) n_obs(4).
const FIRST_REGION: usize = 16;

fn region_len(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize
}

#[test]
fn seat_init_golden_decodes() {
    let init = SeatInit::decode(SEAT_INIT).expect("golden seat-init decodes");
    assert_eq!(init.seat, 1);
    assert_eq!(init.meta("control_hz"), Some("50"));
    assert_eq!(init.meta("task"), Some("golden"));

    let obs: Vec<&str> = init.obs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(obs, ["marquee", "agent", "cues"]);
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
    assert!(
        init.obs[1].columns.is_empty(),
        "a rank-1 value has no columns"
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

/// A rank-2 value documents the columns of its last axis: exactly one per
/// column, each with a name, a doc and a unit, indexable by name.
#[test]
fn seat_init_golden_columns() {
    let init = SeatInit::decode(SEAT_INIT).unwrap();
    let cues = init.obs("cues").expect("rank-2 obs value");
    assert_eq!(cues.dtype, Dtype::F32);
    assert_eq!(cues.shape, [2, 3]);
    assert_eq!(cues.rank(), 2);
    assert_eq!(cues.doc, "the next 2 cues as rows, nearest first");
    assert!(cues.high.is_infinite() && cues.high > 0.0);
    let names: Vec<&str> = cues.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["move", "ticks_to_hit", "valid"]);
    assert_eq!(cues.columns.len(), cues.shape[1] as usize);
    assert_eq!(cues.column_index("ticks_to_hit"), Some(1));
    assert_eq!(cues.column_index("nope"), None);
    let mv = cues.column("move").unwrap();
    assert_eq!(mv.unit, "code");
    assert_eq!(mv.doc, "0 hold, 1 step, 2 turn");
    assert_eq!(cues.column("valid").map(|c| c.unit.as_str()), Some("flag"));
    // A column index is an offset along the last axis: row 1, column 1.
    let view = View::decode(VIEW).unwrap();
    let flat: Vec<f32> = view.values[2]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let col = cues.column_index("ticks_to_hit").unwrap();
    assert_eq!(flat[cues.shape[1] as usize + col], 0.0);
    assert_eq!(flat[col], 7.0);
}

/// The metric tail: three declared metrics with kind, direction, unit,
/// range and headline flag, looked up by key.
#[test]
fn seat_init_golden_metrics() {
    let init = SeatInit::decode(SEAT_INIT).unwrap();
    let keys: Vec<&str> = init.metrics.iter().map(|m| m.key.as_str()).collect();
    assert_eq!(keys, ["score", "misses", "scenario-tempo"]);

    let score = init.metric_spec("score").unwrap();
    assert_eq!(score.kind, MetricKind::Score);
    assert_eq!(score.direction, MetricDirection::Higher);
    assert_eq!((score.low, score.high), (0.0, 100.0));
    assert!(score.headline);
    assert_eq!(score.doc, "0–100 as in the brief");
    assert_eq!(score.unit, "");

    let misses = init.metric_spec("misses").unwrap();
    assert_eq!(misses.kind, MetricKind::Rate);
    assert_eq!(misses.direction, MetricDirection::Lower);
    assert_eq!(misses.unit, "%");
    assert!(
        misses.low.is_infinite() && misses.low < 0.0,
        "±inf = auto range"
    );
    assert!(misses.high.is_infinite() && misses.high > 0.0);
    assert!(!misses.headline);

    let tempo = init.metric_spec("scenario-tempo").unwrap();
    assert_eq!(tempo.kind, MetricKind::Scenario);
    assert_eq!(tempo.direction, MetricDirection::Neutral);
    assert_eq!((tempo.low, tempo.high), (60.0, 180.0));
    assert_eq!(tempo.unit, "bpm");

    assert_eq!(init.metric_spec("nope").map(|m| m.key.as_str()), None);
    let headline: Vec<&str> = init.headline_metrics().map(|m| m.key.as_str()).collect();
    assert_eq!(headline, ["score"]);
}

/// The frozen no-tail golden: the same declaration written before the
/// metrics section existed decodes with `metrics = []` and nothing else
/// different.
#[test]
fn seat_init_notail_golden_decodes_with_no_metrics() {
    let init = SeatInit::decode(SEAT_INIT_NOTAIL).expect("no-tail seat-init decodes");
    assert!(init.metrics.is_empty());
    assert_eq!(init.metric_spec("score").map(|m| m.key.as_str()), None);
    let full = SeatInit::decode(SEAT_INIT).unwrap();
    assert_eq!(init.seat, full.seat);
    assert_eq!(init.obs.len(), full.obs.len());
    assert_eq!(init.actions.len(), full.actions.len());
    assert_eq!(init.obs("cues").unwrap().column_index("valid"), Some(2));
    assert_eq!(init.brief.reward, full.brief.reward);
    // The no-tail golden IS the full golden minus its tail: the bytes up
    // to `ends` are identical.
    assert_eq!(&SEAT_INIT[..SEAT_INIT_NOTAIL.len()], SEAT_INIT_NOTAIL);
}

/// A tail that starts and stops short is an error, not an empty list.
#[test]
fn seat_init_short_tail_is_an_error() {
    // A count of 2 with nothing behind it.
    let mut short = SEAT_INIT_NOTAIL.to_vec();
    short.extend_from_slice(&2u32.to_le_bytes());
    assert!(SeatInit::decode(&short).is_err());
    // The full golden cut inside its last metric spec.
    assert!(SeatInit::decode(&SEAT_INIT[..SEAT_INIT.len() - 1]).is_err());
    // A count alone is fine: an explicit empty tail.
    let mut empty = SEAT_INIT_NOTAIL.to_vec();
    empty.extend_from_slice(&0u32.to_le_bytes());
    assert!(SeatInit::decode(&empty).unwrap().metrics.is_empty());
}

/// Bytes after the sections a reader knows are ignored: the declaration is
/// tail-extensible (Views and Inputs are not).
#[test]
fn seat_init_ignores_unknown_trailing_bytes() {
    let mut extended = SEAT_INIT.to_vec();
    extended.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
    let init = SeatInit::decode(&extended).unwrap();
    assert_eq!(init.obs.len(), 3);
    assert_eq!(init.metrics.len(), 3);
}

/// Every `ValueSpec` is a length-prefixed region: bytes a reader does not
/// know about at the end of the region are skipped, and a region that ends
/// before the fields the reader knows is an error.
#[test]
fn value_spec_region_skips_unknown_trailing_fields() {
    let len = region_len(SEAT_INIT, FIRST_REGION);
    let mut patched = SEAT_INIT.to_vec();
    let end = FIRST_REGION + 4 + len;
    patched.splice(end..end, [0xAA, 0xBB, 0xCC]);
    patched[FIRST_REGION..FIRST_REGION + 4].copy_from_slice(&((len + 3) as u32).to_le_bytes());
    let init = SeatInit::decode(&patched).expect("unknown region bytes are skipped");
    assert_eq!(init.obs[0].name, "marquee");
    assert_eq!(init.obs[1].name, "agent");
    assert_eq!(init.obs("cues").unwrap().column_index("valid"), Some(2));
    assert_eq!(init.metrics.len(), 3);

    // Shrink the region below its known fields.
    let mut truncated = SEAT_INIT.to_vec();
    truncated[FIRST_REGION..FIRST_REGION + 4].copy_from_slice(&((len - 5) as u32).to_le_bytes());
    assert!(SeatInit::decode(&truncated).is_err());

    // A region length that runs past the input.
    let mut overrun = SEAT_INIT.to_vec();
    overrun[FIRST_REGION..FIRST_REGION + 4].copy_from_slice(&(u32::MAX).to_le_bytes());
    assert!(SeatInit::decode(&overrun).is_err());
}

#[test]
fn view_golden_decodes_zero_copy() {
    let view = View::decode(VIEW).expect("golden view decodes");
    assert_eq!(view.tick, 42);
    assert!((view.reward - -0.125).abs() < 1e-6);
    assert!(view.done);
    assert_eq!(view.values.len(), 3);
    assert_eq!(view.values[0], [0u8, 1, 2, 3, 4, 5, 6, 255]);
    let agent: Vec<f32> = view.values[1]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(agent, [0.5, -0.5, 1.25, -1.25, 0.75]);
    // The rank-2 value arrives row-major: 2 rows × 3 columns of f32.
    assert_eq!(view.values[2].len(), 24);
}

#[test]
fn input_golden_reencodes_byte_for_byte() {
    // The golden input carries action = [0.25, -0.75, 1.0], mode = [2i32].
    let encoded = encode_input(&[f32_bytes(&[0.25, -0.75, 1.0]), 2i32.to_le_bytes().to_vec()]);
    assert_eq!(encoded, INPUT, "hand encoder must match the golden bytes");
}
