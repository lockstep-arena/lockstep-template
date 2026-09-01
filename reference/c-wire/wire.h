/* A HAND-WRITTEN Lockstep-wire (v1) reader/writer in C99 — the whole point.
 *
 * The wire is a *spec*, not a library: this pair (wire.h / wire.c)
 * re-implements it from the published document (docs/wire.md in this
 * template — the vendored copy of the platform's normative spec) with no
 * dependencies beyond libc. It is the C twin of reference/rust-wire's
 * wire.rs, tested by `task test` against the same golden encodings.
 *
 * Encoding rules (the short version — the spec is normative):
 * - everything little-endian; f32 is IEEE-754 binary32
 * - str = u16 length + UTF-8 bytes, no terminator
 * - every value and slice carries a doc string (slices a unit too), and
 *   the seat-init ends with the goal / reward / ends brief — the
 *   environment documents itself; decoders that don't care skip the strings
 * - a value's bytes are row-major, dtype size × product(shape) exactly
 * - dtype: 0 = f32 (4 B), 1 = u8 (1 B), 2 = i32 (4 B)
 * - values appear in DECLARED order with exact byte lengths
 * - SeatInit readers ignore unknown trailing bytes; View/Input are strict
 *
 * Ownership: decode fills structs that own heap allocations — free them
 * with the matching wire_*_free. wire_view_t borrows its blobs from the
 * buffer you decoded (zero-copy): keep that buffer alive while you read.
 */
#ifndef LOCKSTEP_WIRE_H
#define LOCKSTEP_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { WIRE_OK = 0, WIRE_ERR = 1 } wire_status_t;

typedef enum { WIRE_F32 = 0, WIRE_U8 = 1, WIRE_I32 = 2 } wire_dtype_t;

/* Bytes per element of a dtype. */
size_t wire_dtype_size(wire_dtype_t dtype);

/* A named documentation run inside a flat value (never a second layout). */
typedef struct {
    char *name; /* owned, NUL-terminated */
    char *doc;  /* what this run of elements IS, in the engine's words */
    char *unit; /* physical unit ("m", "rad/s"); empty when dimensionless */
    uint32_t start;
    uint32_t len;
} wire_slice_t;

/* One declared observation or action — a named value. */
typedef struct {
    char *name;
    char *doc;
    wire_dtype_t dtype;
    uint32_t *shape; /* owned, rank entries */
    uint8_t rank;
    float low;
    float high;
    /* Per-element bounds (numel each) when declared, else NULL. */
    float *elem_low;
    float *elem_high;
    wire_slice_t *slices; /* owned, n_slices entries */
    uint32_t n_slices;
    /* product(shape), min 1 — precomputed at decode. */
    size_t numel;
} wire_value_spec_t;

/* Element bounds at i: per-element when declared, else the scalars. */
void wire_bounds_at(const wire_value_spec_t *spec, size_t i, float *low, float *high);

/* The wire's own neutral: midpoint of finite bounds, else 0. `out` must
 * hold spec->numel floats. */
void wire_neutral_f32(const wire_value_spec_t *spec, float *out);

/* Find a documentation slice by name; NULL when absent. */
const wire_slice_t *wire_find_slice(const wire_value_spec_t *spec, const char *name);

/* The seat's brief — goal / reward / episode end, in the engine's words. */
typedef struct {
    char *goal;
    char *reward;
    char *ends;
} wire_brief_t;

typedef struct {
    char *key;
    char *value;
} wire_meta_t;

/* The declaration: decode once at init, read per-tick blobs against it. */
typedef struct {
    uint32_t seat;
    wire_value_spec_t *obs; /* owned */
    uint32_t n_obs;
    wire_value_spec_t *actions; /* owned */
    uint32_t n_actions;
    wire_meta_t *meta; /* owned */
    uint32_t n_meta;
    wire_brief_t brief;
} wire_seat_init_t;

wire_status_t wire_seat_init_decode(const uint8_t *bytes, size_t len, wire_seat_init_t *out);
void wire_seat_init_free(wire_seat_init_t *init);

/* Meta lookup; NULL when the key is absent. */
const char *wire_meta_get(const wire_seat_init_t *init, const char *key);

/* Find a declared action by name; NULL when absent. */
const wire_value_spec_t *wire_find_action(const wire_seat_init_t *init, const char *name);

/* One blob of a View: BORROWED from the decoded buffer (zero-copy). */
typedef struct {
    const uint8_t *ptr;
    size_t len;
} wire_blob_t;

/* One tick's observation. `values[i]` is the i-th DECLARED observation's
 * raw bytes, borrowed from the buffer passed to decode. */
typedef struct {
    uint32_t tick;
    float reward;
    int done;
    wire_blob_t *values; /* owned array of borrowed blobs */
    uint32_t n_values;
} wire_view_t;

wire_status_t wire_view_decode(const uint8_t *bytes, size_t len, wire_view_t *out);
void wire_view_free(wire_view_t *view);

/* Little-endian f32s out of a raw blob (an f32 value's bytes). */
void wire_read_f32(const uint8_t *bytes, size_t n, float *out);
/* Little-endian i32s out of a raw blob (an i32 value's bytes). */
void wire_read_i32(const uint8_t *bytes, size_t n, int32_t *out);

/* Build an Input: one blob per declared action, in declared order.
 * finish() mallocs the encoded message into (*ptr, *len) — exactly what a
 * wit-bindgen `input-payload` return wants (the canonical ABI frees it). */
typedef struct {
    uint8_t *buf;
    size_t len;
    size_t cap;
    uint32_t n_declared;
    uint32_t n_pushed;
} wire_input_builder_t;

void wire_input_builder_start(wire_input_builder_t *b, uint32_t n_actions);
/* Push one action's f32 values (spec->numel of them), encoded per the
 * spec's dtype (u8/i32 round). */
void wire_input_builder_push_f32(wire_input_builder_t *b, const wire_value_spec_t *spec,
                                 const float *values);
/* Push one action's raw element bytes verbatim (spec->numel * dtype size). */
void wire_input_builder_push_raw(wire_input_builder_t *b, const uint8_t *bytes, size_t len);
/* Hand ownership of the encoded message to the caller. */
void wire_input_builder_finish(wire_input_builder_t *b, uint8_t **ptr, size_t *len);

#ifdef __cplusplus
}
#endif

#endif /* LOCKSTEP_WIRE_H */
