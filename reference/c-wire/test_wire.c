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

static uint32_t le_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void put_le_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

/* Offset of the first value spec's region length: magic(4) version(4)
 * seat(4) n_obs(4). */
#define FIRST_REGION 16

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "../rust-wire/tests/fixtures";
    size_t si_len, notail_len, view_len, input_len;
    uint8_t *si_bytes = read_file(fixture(dir, "seat_init.bin"), &si_len);
    uint8_t *notail_bytes = read_file(fixture(dir, "seat_init_notail.bin"), &notail_len);
    uint8_t *view_bytes = read_file(fixture(dir, "view.bin"), &view_len);
    uint8_t *input_bytes = read_file(fixture(dir, "input.bin"), &input_len);

    /* ── seat-init golden decodes ── */
    wire_seat_init_t init;
    CHECK(wire_seat_init_decode(si_bytes, si_len, &init) == WIRE_OK);
    CHECK(init.seat == 1);
    CHECK(wire_meta_get(&init, "control_hz") && !strcmp(wire_meta_get(&init, "control_hz"), "50"));
    CHECK(wire_meta_get(&init, "task") && !strcmp(wire_meta_get(&init, "task"), "golden"));
    CHECK(init.n_obs == 3);
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
    CHECK(init.obs[1].n_columns == 0 && init.obs[1].columns == NULL); /* rank-1: no columns */
    CHECK(strlen(init.brief.goal) > 0 && strlen(init.brief.reward) > 0 &&
          strlen(init.brief.ends) > 0);
    CHECK(init.n_actions == 2);
    const wire_value_spec_t *action = wire_find_action(&init, "action");
    CHECK(action && action->dtype == WIRE_F32 && action->numel == 3);
    const wire_value_spec_t *mode = wire_find_action(&init, "mode");
    CHECK(mode && mode->dtype == WIRE_I32 && mode->numel == 1);

    /* ── a rank-2 value documents the columns of its last axis ── */
    const wire_value_spec_t *cues = wire_find_obs(&init, "cues");
    CHECK(cues && cues->dtype == WIRE_F32 && cues->rank == 2);
    CHECK(cues && cues->shape[0] == 2 && cues->shape[1] == 3 && cues->numel == 6);
    CHECK(cues && !strcmp(cues->doc, "the next 2 cues as rows, nearest first"));
    CHECK(cues && cues->low == -1.0f && isinf(cues->high) && cues->high > 0);
    CHECK(cues && cues->n_columns == 3 && cues->n_columns == cues->shape[1]);
    CHECK(cues && !strcmp(cues->columns[0].name, "move") &&
          !strcmp(cues->columns[1].name, "ticks_to_hit") && !strcmp(cues->columns[2].name, "valid"));
    CHECK(cues && wire_column_index(cues, "ticks_to_hit") == 1);
    CHECK(cues && wire_column_index(cues, "nope") == -1);
    const wire_column_t *mv = cues ? wire_find_column(cues, "move") : NULL;
    CHECK(mv && !strcmp(mv->unit, "code") && !strcmp(mv->doc, "0 hold, 1 step, 2 turn"));
    const wire_column_t *valid = cues ? wire_find_column(cues, "valid") : NULL;
    CHECK(valid && !strcmp(valid->unit, "flag"));
    CHECK(cues && wire_find_column(cues, "nope") == NULL);

    /* ── the metric tail: three declared metrics, looked up by key ── */
    CHECK(init.n_metrics == 3);
    CHECK(!strcmp(init.metrics[0].key, "score") && !strcmp(init.metrics[1].key, "misses") &&
          !strcmp(init.metrics[2].key, "scenario-tempo"));
    const wire_metric_spec_t *score = wire_find_metric(&init, "score");
    CHECK(score && score->kind == WIRE_METRIC_SCORE && score->direction == WIRE_HIGHER_IS_BETTER);
    CHECK(score && score->low == 0.0f && score->high == 100.0f && score->headline);
    CHECK(score && !strcmp(score->doc, "0–100 as in the brief") && !strcmp(score->unit, ""));
    const wire_metric_spec_t *misses = wire_find_metric(&init, "misses");
    CHECK(misses && misses->kind == WIRE_METRIC_RATE && misses->direction == WIRE_LOWER_IS_BETTER);
    CHECK(misses && !strcmp(misses->unit, "%") && !misses->headline);
    CHECK(misses && isinf(misses->low) && misses->low < 0 && isinf(misses->high) &&
          misses->high > 0); /* ±inf = auto range */
    const wire_metric_spec_t *tempo = wire_find_metric(&init, "scenario-tempo");
    CHECK(tempo && tempo->kind == WIRE_METRIC_SCENARIO &&
          tempo->direction == WIRE_DIRECTION_NEUTRAL);
    CHECK(tempo && tempo->low == 60.0f && tempo->high == 180.0f && !strcmp(tempo->unit, "bpm"));
    CHECK(wire_find_metric(&init, "nope") == NULL);

    /* ── the frozen no-tail golden: same declaration, zero metrics ── */
    wire_seat_init_t notail;
    CHECK(wire_seat_init_decode(notail_bytes, notail_len, &notail) == WIRE_OK);
    CHECK(notail.n_metrics == 0 && notail.metrics == NULL);
    CHECK(wire_find_metric(&notail, "score") == NULL);
    CHECK(notail.seat == init.seat && notail.n_obs == init.n_obs &&
          notail.n_actions == init.n_actions);
    CHECK(wire_find_obs(&notail, "cues") && wire_column_index(wire_find_obs(&notail, "cues"), "valid") == 2);
    CHECK(!strcmp(notail.brief.reward, init.brief.reward));
    /* The no-tail golden IS the full golden minus its tail. */
    CHECK(notail_len < si_len && memcmp(si_bytes, notail_bytes, notail_len) == 0);
    wire_seat_init_free(&notail);

    /* A tail that starts and stops short is an error, not an empty list. */
    wire_seat_init_t bad;
    uint8_t *short_tail = malloc(notail_len + 4);
    memcpy(short_tail, notail_bytes, notail_len);
    put_le_u32(short_tail + notail_len, 2); /* a count of 2 with nothing behind it */
    CHECK(wire_seat_init_decode(short_tail, notail_len + 4, &bad) == WIRE_ERR);
    CHECK(wire_seat_init_decode(si_bytes, si_len - 1, &bad) == WIRE_ERR); /* cut inside a metric */
    put_le_u32(short_tail + notail_len, 0); /* an explicit empty tail is fine */
    CHECK(wire_seat_init_decode(short_tail, notail_len + 4, &bad) == WIRE_OK && bad.n_metrics == 0);
    wire_seat_init_free(&bad);
    free(short_tail);

    /* SeatInit readers ignore unknown trailing bytes after the sections
     * they know (spec rule). */
    uint8_t *extended = malloc(si_len + 3);
    memcpy(extended, si_bytes, si_len);
    memcpy(extended + si_len, "\xaa\xbb\xcc", 3);
    wire_seat_init_t init2;
    CHECK(wire_seat_init_decode(extended, si_len + 3, &init2) == WIRE_OK);
    CHECK(init2.n_obs == init.n_obs && init2.n_actions == init.n_actions &&
          init2.n_metrics == init.n_metrics);
    wire_seat_init_free(&init2);
    free(extended);

    /* Every value spec is a length-prefixed region: unknown bytes at the
     * end of a region are skipped; a region that ends before the known
     * fields, or runs past the input, is an error. */
    uint32_t region_len = le_u32(si_bytes + FIRST_REGION);
    size_t region_end = FIRST_REGION + 4 + region_len;
    uint8_t *patched = malloc(si_len + 3);
    memcpy(patched, si_bytes, region_end);
    memcpy(patched + region_end, "\xaa\xbb\xcc", 3);
    memcpy(patched + region_end + 3, si_bytes + region_end, si_len - region_end);
    put_le_u32(patched + FIRST_REGION, region_len + 3);
    wire_seat_init_t init3;
    CHECK(wire_seat_init_decode(patched, si_len + 3, &init3) == WIRE_OK);
    CHECK(!strcmp(init3.obs[0].name, "marquee") && !strcmp(init3.obs[1].name, "agent"));
    CHECK(wire_find_obs(&init3, "cues") && wire_column_index(wire_find_obs(&init3, "cues"), "valid") == 2);
    CHECK(init3.n_metrics == 3);
    wire_seat_init_free(&init3);
    free(patched);
    uint8_t *truncated = malloc(si_len);
    memcpy(truncated, si_bytes, si_len);
    put_le_u32(truncated + FIRST_REGION, region_len - 5);
    CHECK(wire_seat_init_decode(truncated, si_len, &bad) == WIRE_ERR);
    put_le_u32(truncated + FIRST_REGION, 0xFFFFFFFFu);
    CHECK(wire_seat_init_decode(truncated, si_len, &bad) == WIRE_ERR);
    free(truncated);

    /* Bad magic / bad version are errors, not crashes. */
    CHECK(wire_seat_init_decode((const uint8_t *)"XXXX", 4, &bad) == WIRE_ERR);

    /* ── view golden decodes (zero-copy blobs) ── */
    wire_view_t view;
    CHECK(wire_view_decode(view_bytes, view_len, &view) == WIRE_OK);
    CHECK(view.tick == 42);
    CHECK(fabsf(view.reward - (-0.125f)) < 1e-9f);
    CHECK(view.done == 1);
    CHECK(view.n_values == 3);
    CHECK(view.values[0].len == 8); /* marquee u8[1,2,4] */
    CHECK(view.values[0].ptr[7] == 255);
    CHECK(view.values[1].len == 20); /* agent f32[5] */
    float agent[5];
    wire_read_f32(view.values[1].ptr, 5, agent);
    CHECK(agent[0] == 0.5f && agent[1] == -0.5f && agent[4] == 0.75f);
    CHECK(view.values[2].len == 24); /* cues f32[2,3], row-major */
    float cue_vals[6];
    wire_read_f32(view.values[2].ptr, 6, cue_vals);
    /* A column index is an offset along the last axis: row r, column c is
     * flat[r * shape[1] + c]. */
    long tth = cues ? wire_column_index(cues, "ticks_to_hit") : -1;
    CHECK(tth == 1 && cue_vals[tth] == 7.0f && cue_vals[3 + tth] == 0.0f);
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
    free(notail_bytes);
    free(view_bytes);
    free(input_bytes);

    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("c-wire: all golden checks passed\n");
    return 0;
}
