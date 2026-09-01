/* The hand-written C decoder against the SPEC's published golden encodings
 * (the .bin files under ../rust-wire/tests/fixtures — the same files the
 * Rust reference pins). Built and run natively by `task test`:
 *
 *   cc -std=c99 -Wall -Wextra -o test_wire test_wire.c wire.c && ./test_wire
 */

#include "wire.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            failures++;                                                            \
        }                                                                          \
    } while (0)

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "short read on %s\n", path);
        exit(2);
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static char *fixture(const char *dir, const char *name) {
    char *p = malloc(strlen(dir) + strlen(name) + 2);
    sprintf(p, "%s/%s", dir, name);
    return p;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "../rust-wire/tests/fixtures";
    size_t si_len, view_len, input_len;
    uint8_t *si_bytes = read_file(fixture(dir, "seat_init.bin"), &si_len);
    uint8_t *view_bytes = read_file(fixture(dir, "view.bin"), &view_len);
    uint8_t *input_bytes = read_file(fixture(dir, "input.bin"), &input_len);

    /* ── seat-init golden decodes ── */
    wire_seat_init_t init;
    CHECK(wire_seat_init_decode(si_bytes, si_len, &init) == WIRE_OK);
    CHECK(init.seat == 1);
    CHECK(wire_meta_get(&init, "control_hz") && !strcmp(wire_meta_get(&init, "control_hz"), "50"));
    CHECK(wire_meta_get(&init, "task") && !strcmp(wire_meta_get(&init, "task"), "golden"));
    CHECK(init.n_obs == 2);
    CHECK(!strcmp(init.obs[0].name, "marquee") && init.obs[0].dtype == WIRE_U8);
    CHECK(init.obs[0].rank == 3 && init.obs[0].shape[0] == 1 && init.obs[0].shape[1] == 2 &&
          init.obs[0].shape[2] == 4);
    CHECK(!strcmp(init.obs[1].name, "agent") && init.obs[1].dtype == WIRE_F32);
    float lo, hi;
    wire_bounds_at(&init.obs[1], 2, &lo, &hi); /* per-element overrides the scalars */
    CHECK(lo == -10.0f && hi == 10.0f);
    const wire_slice_t *jp = wire_find_slice(&init.obs[1], "joint_pos");
    CHECK(jp && jp->start == 0 && jp->len == 2);
    CHECK(jp && !strcmp(jp->doc, "joint angles") && !strcmp(jp->unit, "rad"));
    CHECK(strlen(init.brief.goal) > 0 && strlen(init.brief.reward) > 0 &&
          strlen(init.brief.ends) > 0);
    CHECK(init.n_actions == 2);
    const wire_value_spec_t *action = wire_find_action(&init, "action");
    CHECK(action && action->dtype == WIRE_F32 && action->numel == 3);
    const wire_value_spec_t *mode = wire_find_action(&init, "mode");
    CHECK(mode && mode->dtype == WIRE_I32 && mode->numel == 1);

    /* SeatInit readers ignore unknown trailing bytes (spec rule). */
    uint8_t *extended = malloc(si_len + 3);
    memcpy(extended, si_bytes, si_len);
    memcpy(extended + si_len, "\xaa\xbb\xcc", 3);
    wire_seat_init_t init2;
    CHECK(wire_seat_init_decode(extended, si_len + 3, &init2) == WIRE_OK);
    CHECK(init2.n_obs == init.n_obs && init2.n_actions == init.n_actions);
    wire_seat_init_free(&init2);
    free(extended);

    /* Truncated / bad magic / bad version are errors, not crashes. */
    wire_seat_init_t bad;
    CHECK(wire_seat_init_decode(si_bytes, si_len - 1, &bad) == WIRE_ERR);
    CHECK(wire_seat_init_decode((const uint8_t *)"XXXX", 4, &bad) == WIRE_ERR);

    /* ── view golden decodes (zero-copy blobs) ── */
    wire_view_t view;
    CHECK(wire_view_decode(view_bytes, view_len, &view) == WIRE_OK);
    CHECK(view.tick == 42);
    CHECK(fabsf(view.reward - (-0.125f)) < 1e-9f);
    CHECK(view.done == 1);
    CHECK(view.n_values == 2);
    CHECK(view.values[0].len == 8); /* marquee u8[1,2,4] */
    CHECK(view.values[0].ptr[7] == 255);
    CHECK(view.values[1].len == 20); /* agent f32[5] */
    float agent[5];
    wire_read_f32(view.values[1].ptr, 5, agent);
    CHECK(agent[0] == 0.5f && agent[1] == -0.5f && agent[4] == 0.75f);
    /* Views are strict about trailing garbage. */
    uint8_t *vext = malloc(view_len + 1);
    memcpy(vext, view_bytes, view_len);
    vext[view_len] = 0;
    wire_view_t vbad;
    CHECK(wire_view_decode(vext, view_len + 1, &vbad) == WIRE_ERR);
    free(vext);

    /* ── input golden re-encodes byte-for-byte ── */
    wire_input_builder_t b;
    wire_input_builder_start(&b, 2);
    float act_vals[3] = {0.25f, -0.75f, 1.0f};
    wire_input_builder_push_f32(&b, action, act_vals);
    float mode_vals[1] = {2.0f};
    wire_input_builder_push_f32(&b, mode, mode_vals);
    uint8_t *enc;
    size_t enc_len;
    wire_input_builder_finish(&b, &enc, &enc_len);
    CHECK(enc_len == input_len);
    CHECK(enc && memcmp(enc, input_bytes, input_len) == 0);
    free(enc);

    /* ── neutral: midpoint of finite bounds, u8/i32 round ── */
    float neutral[3];
    wire_neutral_f32(action, neutral); /* bounds [-1, 1] → 0 */
    CHECK(neutral[0] == 0.0f && neutral[1] == 0.0f && neutral[2] == 0.0f);
    float mode_neutral[1];
    wire_neutral_f32(mode, mode_neutral); /* bounds [0, 3] → 1.5 */
    CHECK(mode_neutral[0] == 1.5f);
    wire_input_builder_start(&b, 1);
    wire_input_builder_push_f32(&b, mode, mode_neutral); /* i32 rounds → 2 */
    wire_input_builder_finish(&b, &enc, &enc_len);
    CHECK(enc_len == 4 + 4 + 4 + 4);
    int32_t rounded;
    wire_read_i32(enc + 12, 1, &rounded);
    CHECK(rounded == 2);
    free(enc);

    /* A push-count mismatch yields the EMPTY payload, never a malformed one. */
    wire_input_builder_start(&b, 2);
    wire_input_builder_push_f32(&b, action, act_vals);
    wire_input_builder_finish(&b, &enc, &enc_len);
    CHECK(enc == NULL && enc_len == 0);

    wire_view_free(&view);
    wire_seat_init_free(&init);
    free(si_bytes);
    free(view_bytes);
    free(input_bytes);

    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("c-wire: all golden checks passed\n");
    return 0;
}
