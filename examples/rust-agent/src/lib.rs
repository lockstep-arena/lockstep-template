//! The sway bot, in pure Rust — a dance-off agent with NO trained policy and
//! NO private dependencies.
//!
//! Everything it consumes is public: the `lockstep:agent` WIT world
//! (vendored under `wit/`) and the FlatBuffers wire contract
//! (`contract/dance-off.fbs`, published with every engine release as
//! `contract.fbs` on the CDN — `task contract` refreshes it). Compare
//! `examples/scripted_agent.py` — this is the same dumb policy, but instead
//! of riding the prebuilt ONNX shell it IS the component: bytes in, bytes
//! out, no inference host involved.
//!
//! Two things worth noticing:
//!
//! - the View is read ZERO-COPY: `ViewRef` walks the received buffer in
//!   place — the 16 KB marquee raster is never copied (or, here, even
//!   touched: this bot is deliberately marquee-blind).
//! - a decode failure answers with an EMPTY input rather than a panic: the
//!   engine zero-pads a short/absent action into "stand still", which is the
//!   correct way to misbehave.

#[allow(dead_code, clippy::all)]
#[rustfmt::skip]
mod wire;

use planus::ReadAsRoot;
use wire::dance_off::{ServoInput, Vec3, ViewRef};

wit_bindgen::generate!({
    path: "../rust-agent/wit",
    world: "agent",
    generate_all,
});

const NUM_JOINTS: usize = 12;

struct Component;

impl Guest for Component {
    fn init(_init_state: Vec<u8>) {
        // SeatInit carries only our slot; the sway needs nothing from it.
    }

    fn on_tick(view: Vec<u8>) -> Vec<u8> {
        // Zero-copy read of the tick; everything else in the View ignored.
        let tick = ViewRef::read_as_root(&view)
            .and_then(|v| v.tick())
            .unwrap_or(0);

        // The same lazy sway as the Python example: ~half-hertz sine on the
        // first two joints, neutral everywhere else, even effort.
        let seconds = tick as f32 / 60.0;
        let wave = 0.35 * libm::sinf(seconds * 3.0);

        let mut joints = vec![
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            NUM_JOINTS
        ];
        joints[0].x = wave;
        joints[1].x = wave;

        let input = ServoInput {
            joints,
            effort: vec![0.5; NUM_JOINTS],
        };
        let mut b = planus::Builder::new();
        b.finish(&input, None).to_vec()
    }
}

export!(Component);
