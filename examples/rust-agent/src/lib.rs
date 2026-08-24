//! The sway bot, in pure Rust — an agent with NO trained policy and NO
//! private dependencies, for ANY environment.
//!
//! Everything it consumes is public: the `lockstep:agent` WIT world
//! (vendored under `wit/`) and the tensor-wire SPEC (`docs/wire.md` in the
//! public lockstep-interface repo), re-implemented by hand in `src/wire.rs`
//! — no codegen, no schema files, no reference crate. Compare
//! `examples/scripted_agent.py`: same idea, but instead of riding the
//! generic ONNX shell this IS the component — bytes in, bytes out, no
//! inference host involved.
//!
//! The policy: read the seat-init's self-description at `init`, then
//!
//! - on dance-off's servo mode (recognized purely from the DECLARATION —
//!   meta `model = "dance-off"` plus an action tensor with `joint_targets`
//!   and `effort` slices) it sways the first two joints on a slow sine and
//!   spreads effort evenly: a bot that visibly dances;
//! - on ANY other environment it plays the wire's own neutral (the bounds
//!   midpoint of every action element) — the correct way to do nothing.
//!
//! Error posture: a decode failure answers with an EMPTY input rather than
//! a panic. The engine treats a malformed input as neutral (that is the
//! wire's rule), which is the correct way to misbehave.

pub mod wire;

#[cfg(target_arch = "wasm32")]
mod component {
    use crate::wire::{self, SeatInit, TensorSpec, View};
    use std::sync::Mutex;

    wit_bindgen::generate!({
        path: "wit",
        world: "agent",
        generate_all,
    });

    /// What `init` learned; `on_tick` replays it every tick.
    struct Plan {
        /// Pre-encoded neutral bytes, one blob per action tensor.
        neutral: Vec<Vec<u8>>,
        /// Present when the declaration matches dance-off's servo shape.
        sway: Option<Sway>,
    }

    struct Sway {
        /// Index of the swayed tensor within the action list.
        action_index: usize,
        spec: TensorSpec,
        joint_targets: (usize, usize), // start, len
        effort: (usize, usize),
    }

    static PLAN: Mutex<Option<Plan>> = Mutex::new(None);

    struct Component;

    impl Guest for Component {
        fn init(init_state: Vec<u8>) {
            let Ok(init) = SeatInit::decode(&init_state) else {
                return; // no plan → empty inputs → engine plays us neutral
            };
            let neutral = init.actions.iter().map(wire::neutral_bytes).collect();
            let sway = init.actions.iter().enumerate().find_map(|(i, spec)| {
                // Recognized from the declaration alone — nothing here
                // names a version or imports a schema.
                if init.meta("model") != Some("dance-off") {
                    return None;
                }
                let jt = spec.slice("joint_targets")?;
                let effort = spec.slice("effort")?;
                Some(Sway {
                    action_index: i,
                    spec: spec.clone(),
                    joint_targets: (jt.start as usize, jt.len as usize),
                    effort: (effort.start as usize, effort.len as usize),
                })
            });
            *PLAN.lock().unwrap() = Some(Plan { neutral, sway });
        }

        fn on_tick(view: Vec<u8>) -> Vec<u8> {
            let guard = PLAN.lock().unwrap();
            let Some(plan) = guard.as_ref() else {
                return Vec::new();
            };
            let tick = View::decode(&view).map(|v| v.tick).unwrap_or(0);

            let mut tensors = plan.neutral.clone();
            if let Some(sway) = &plan.sway {
                // The same lazy sway as the Python example: ~half-hertz sine
                // on the first two joints' x rotation, even effort.
                let seconds = tick as f32 / 60.0;
                let wave = 0.35 * libm::sinf(seconds * 3.0);
                let mut values = sway.spec.neutral_f32();
                let (jt_start, jt_len) = sway.joint_targets;
                // joint_targets is rotation VECTORS, 3 values per joint —
                // (x, y, z) per joint; sway x of joints 0 and 1.
                for joint in 0..2 {
                    let x = jt_start + joint * 3;
                    if x < jt_start + jt_len {
                        values[x] = wave;
                    }
                }
                let (e_start, e_len) = sway.effort;
                for v in values.iter_mut().skip(e_start).take(e_len) {
                    *v = 0.5;
                }
                tensors[sway.action_index] = wire::f32_bytes(&values);
            }
            wire::encode_input(&tensors)
        }
    }

    export!(Component);
}
