/* See wire.h — the hand-written C reader/writer for the Lockstep wire (v1),
 * mirrored line-for-line from reference/rust-wire/src/wire.rs and pinned by
 * the same spec goldens. */

#include "wire.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── primitives ───────────────────────────────────────────────────────── */

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t pos;
} reader_t;

static int take(reader_t *r, size_t n, const uint8_t **out) {
    if (n > r->len - r->pos)
        return 0;
    *out = r->data + r->pos;
    r->pos += n;
    return 1;
}

static int magic(reader_t *r, const char expected[4]) {
    const uint8_t *p;
    return take(r, 4, &p) && memcmp(p, expected, 4) == 0;
}

static int read_u8(reader_t *r, uint8_t *v) {
    const uint8_t *p;
    if (!take(r, 1, &p))
        return 0;
    *v = p[0];
    return 1;
}

static int read_u16(reader_t *r, uint16_t *v) {
    const uint8_t *p;
    if (!take(r, 2, &p))
        return 0;
    *v = (uint16_t)(p[0] | (p[1] << 8));
    return 1;
}

static int read_u32(reader_t *r, uint32_t *v) {
    const uint8_t *p;
    if (!take(r, 4, &p))
        return 0;
    *v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
    return 1;
}

static int read_f32(reader_t *r, float *v) {
    uint32_t bits;
    if (!read_u32(r, &bits))
        return 0;
    memcpy(v, &bits, 4);
    return 1;
}

/* str = u16 len + UTF-8 bytes; returned NUL-terminated, malloc'd. */
static int read_str(reader_t *r, char **out) {
    uint16_t n;
    const uint8_t *p;
    if (!read_u16(r, &n) || !take(r, n, &p))
        return 0;
    char *s = malloc((size_t)n + 1);
    if (!s)
        return 0;
    memcpy(s, p, n);
    s[n] = '\0';
    *out = s;
    return 1;
}

size_t wire_dtype_size(wire_dtype_t dtype) {
    return dtype == WIRE_U8 ? 1 : 4;
}

/* ── declarations (SeatInit) ──────────────────────────────────────────── */

static void spec_free(wire_value_spec_t *t) {
    free(t->name);
    free(t->doc);
    free(t->shape);
    free(t->elem_low);
    free(t->elem_high);
    for (uint32_t i = 0; i < t->n_slices; i++) {
        free(t->slices[i].name);
        free(t->slices[i].doc);
        free(t->slices[i].unit);
    }
    free(t->slices);
    for (uint32_t i = 0; i < t->n_columns; i++) {
        free(t->columns[i].name);
        free(t->columns[i].doc);
        free(t->columns[i].unit);
    }
    free(t->columns);
    memset(t, 0, sizeof(*t));
}

/* The fields of one value spec, read out of its region. The region reader
 * `r` spans exactly the region, so every bounds check below is against the
 * region, never the whole message. */
static int spec_parse_fields(reader_t *r, wire_value_spec_t *t) {
    uint8_t dtype_byte, rank, has_elem_bounds;
    if (!read_str(r, &t->name) || !read_str(r, &t->doc) || !read_u8(r, &dtype_byte) ||
        dtype_byte > 2 || !read_u8(r, &rank))
        return 0;
    t->dtype = (wire_dtype_t)dtype_byte;
    t->rank = rank;
    t->shape = malloc(sizeof(uint32_t) * (rank ? rank : 1));
    if (!t->shape)
        return 0;
    t->numel = 1;
    for (uint8_t i = 0; i < rank; i++) {
        if (!read_u32(r, &t->shape[i]))
            return 0;
        t->numel *= t->shape[i];
    }
    if (t->numel == 0)
        t->numel = 1;
    if (!read_f32(r, &t->low) || !read_f32(r, &t->high) || !read_u8(r, &has_elem_bounds))
        return 0;
    if (has_elem_bounds) {
        /* Bounds-check before allocating: a hostile numel must not OOM. */
        if (t->numel * 8 > r->len - r->pos)
            return 0;
        t->elem_low = malloc(sizeof(float) * t->numel);
        t->elem_high = malloc(sizeof(float) * t->numel);
        if (!t->elem_low || !t->elem_high)
            return 0;
        for (size_t i = 0; i < t->numel; i++)
            if (!read_f32(r, &t->elem_low[i]))
                return 0;
        for (size_t i = 0; i < t->numel; i++)
            if (!read_f32(r, &t->elem_high[i]))
                return 0;
    }
    uint32_t n_slices;
    if (!read_u32(r, &n_slices))
        return 0;
    if (n_slices) {
        /* Each slice is at least 14 bytes on the wire. */
        if ((size_t)n_slices * 14 > r->len - r->pos)
            return 0;
        t->slices = calloc(n_slices, sizeof(wire_slice_t));
        if (!t->slices)
            return 0;
        t->n_slices = n_slices;
        for (uint32_t i = 0; i < n_slices; i++) {
            wire_slice_t *s = &t->slices[i];
            if (!read_str(r, &s->name) || !read_str(r, &s->doc) || !read_str(r, &s->unit) ||
                !read_u32(r, &s->start) || !read_u32(r, &s->len))
                return 0;
        }
    }
    uint32_t n_columns;
    if (!read_u32(r, &n_columns))
        return 0;
    if (n_columns) {
        /* Each column is at least 6 bytes on the wire. */
        if ((size_t)n_columns * 6 > r->len - r->pos)
            return 0;
        t->columns = calloc(n_columns, sizeof(wire_column_t));
        if (!t->columns)
            return 0;
        t->n_columns = n_columns;
        for (uint32_t i = 0; i < n_columns; i++) {
            wire_column_t *c = &t->columns[i];
            if (!read_str(r, &c->name) || !read_str(r, &c->doc) || !read_str(r, &c->unit))
                return 0;
        }
        /* Columns are exactly one per element of the last axis, or none —
         * and only a rank >= 2 value has them. */
        uint32_t expected = rank >= 2 ? t->shape[rank - 1] : 0;
        if (n_columns != expected)
            return 0;
    }
    /* Anything left in the region belongs to a later revision: skipped. */
    return 1;
}

/* One length-prefixed region: len u32, then the fields. The region is
 * taken whole, the known fields are read out of it, and whatever it holds
 * after them is skipped. A region that ends before the known fields (or a
 * len that runs past the input) is an error. */
static int spec_parse(reader_t *outer, wire_value_spec_t *t) {
    memset(t, 0, sizeof(*t));
    uint32_t len;
    const uint8_t *region;
    if (!read_u32(outer, &len) || !take(outer, len, &region))
        return 0;
    reader_t r = {region, len, 0};
    if (!spec_parse_fields(&r, t)) {
        spec_free(t);
        return 0;
    }
    return 1;
}

void wire_bounds_at(const wire_value_spec_t *spec, size_t i, float *low, float *high) {
    if (spec->elem_low && spec->elem_high && i < spec->numel) {
        *low = spec->elem_low[i];
        *high = spec->elem_high[i];
    } else {
        *low = spec->low;
        *high = spec->high;
    }
}

void wire_neutral_f32(const wire_value_spec_t *spec, float *out) {
    for (size_t i = 0; i < spec->numel; i++) {
        float low, high;
        wire_bounds_at(spec, i, &low, &high);
        out[i] = (isfinite(low) && isfinite(high)) ? (low + high) / 2.0f : 0.0f;
    }
}

const wire_slice_t *wire_find_slice(const wire_value_spec_t *spec, const char *name) {
    for (uint32_t i = 0; i < spec->n_slices; i++)
        if (strcmp(spec->slices[i].name, name) == 0)
            return &spec->slices[i];
    return NULL;
}

const wire_column_t *wire_find_column(const wire_value_spec_t *spec, const char *name) {
    for (uint32_t i = 0; i < spec->n_columns; i++)
        if (strcmp(spec->columns[i].name, name) == 0)
            return &spec->columns[i];
    return NULL;
}

long wire_column_index(const wire_value_spec_t *spec, const char *name) {
    for (uint32_t i = 0; i < spec->n_columns; i++)
        if (strcmp(spec->columns[i].name, name) == 0)
            return (long)i;
    return -1;
}

static int specs_parse(reader_t *r, wire_value_spec_t **out, uint32_t *n_out) {
    uint32_t n;
    if (!read_u32(r, &n))
        return 0;
    /* Each spec is at least 4 (len) + 25 bytes on the wire. */
    if ((size_t)n * 29 > r->len - r->pos)
        return 0;
    wire_value_spec_t *specs = n ? calloc(n, sizeof(wire_value_spec_t)) : NULL;
    if (n && !specs)
        return 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!spec_parse(r, &specs[i])) {
            for (uint32_t j = 0; j < i; j++)
                spec_free(&specs[j]);
            free(specs);
            return 0;
        }
    }
    *out = specs;
    *n_out = n;
    return 1;
}

static int metric_parse(reader_t *r, wire_metric_spec_t *m) {
    uint8_t kind, direction, headline;
    if (!read_str(r, &m->key) || !read_str(r, &m->doc) || !read_str(r, &m->unit) ||
        !read_u8(r, &kind) || kind > 6 || !read_u8(r, &direction) || direction > 2 ||
        !read_f32(r, &m->low) || !read_f32(r, &m->high) || !read_u8(r, &headline))
        return 0;
    m->kind = (wire_metric_kind_t)kind;
    m->direction = (wire_metric_direction_t)direction;
    m->headline = headline != 0;
    return 1;
}

wire_status_t wire_seat_init_decode(const uint8_t *bytes, size_t len, wire_seat_init_t *out) {
    memset(out, 0, sizeof(*out));
    reader_t r = {bytes, len, 0};
    uint32_t version;
    if (!magic(&r, "LSTI") || !read_u32(&r, &version) || version != 1 ||
        !read_u32(&r, &out->seat))
        goto fail;
    if (!specs_parse(&r, &out->obs, &out->n_obs))
        goto fail;
    if (!specs_parse(&r, &out->actions, &out->n_actions))
        goto fail;
    if (!read_u32(&r, &out->n_meta))
        goto fail;
    if (out->n_meta) {
        if ((size_t)out->n_meta * 4 > r.len - r.pos)
            goto fail;
        out->meta = calloc(out->n_meta, sizeof(wire_meta_t));
        if (!out->meta)
            goto fail;
        for (uint32_t i = 0; i < out->n_meta; i++)
            if (!read_str(&r, &out->meta[i].key) || !read_str(&r, &out->meta[i].value))
                goto fail;
    }
    if (!read_str(&r, &out->brief.goal) || !read_str(&r, &out->brief.reward) ||
        !read_str(&r, &out->brief.ends))
        goto fail;
    /* The tail. Nothing after `ends` = a declaration written before the
     * metrics section existed: zero metrics. A tail that starts is read
     * whole — a count with too few specs behind it is an error. Bytes after
     * the sections this reader knows are IGNORED (spec rule): a later
     * revision may append sections. Views and inputs stay strict. */
    if (r.pos < r.len) {
        uint32_t n_metrics;
        if (!read_u32(&r, &n_metrics))
            goto fail;
        if (n_metrics) {
            /* Each metric is at least 17 bytes on the wire. */
            if ((size_t)n_metrics * 17 > r.len - r.pos)
                goto fail;
            out->metrics = calloc(n_metrics, sizeof(wire_metric_spec_t));
            if (!out->metrics)
                goto fail;
            out->n_metrics = n_metrics;
            for (uint32_t i = 0; i < n_metrics; i++)
                if (!metric_parse(&r, &out->metrics[i]))
                    goto fail;
        }
    }
    return WIRE_OK;
fail:
    wire_seat_init_free(out);
    return WIRE_ERR;
}

void wire_seat_init_free(wire_seat_init_t *init) {
    for (uint32_t i = 0; i < init->n_obs; i++)
        spec_free(&init->obs[i]);
    free(init->obs);
    for (uint32_t i = 0; i < init->n_actions; i++)
        spec_free(&init->actions[i]);
    free(init->actions);
    for (uint32_t i = 0; i < init->n_meta; i++) {
        free(init->meta[i].key);
        free(init->meta[i].value);
    }
    free(init->meta);
    free(init->brief.goal);
    free(init->brief.reward);
    free(init->brief.ends);
    for (uint32_t i = 0; i < init->n_metrics; i++) {
        free(init->metrics[i].key);
        free(init->metrics[i].doc);
        free(init->metrics[i].unit);
    }
    free(init->metrics);
    memset(init, 0, sizeof(*init));
}

const char *wire_meta_get(const wire_seat_init_t *init, const char *key) {
    for (uint32_t i = 0; i < init->n_meta; i++)
        if (strcmp(init->meta[i].key, key) == 0)
            return init->meta[i].value;
    return NULL;
}

const wire_value_spec_t *wire_find_obs(const wire_seat_init_t *init, const char *name) {
    for (uint32_t i = 0; i < init->n_obs; i++)
        if (strcmp(init->obs[i].name, name) == 0)
            return &init->obs[i];
    return NULL;
}

const wire_value_spec_t *wire_find_action(const wire_seat_init_t *init, const char *name) {
    for (uint32_t i = 0; i < init->n_actions; i++)
        if (strcmp(init->actions[i].name, name) == 0)
            return &init->actions[i];
    return NULL;
}

const wire_metric_spec_t *wire_find_metric(const wire_seat_init_t *init, const char *key) {
    for (uint32_t i = 0; i < init->n_metrics; i++)
        if (strcmp(init->metrics[i].key, key) == 0)
            return &init->metrics[i];
    return NULL;
}

/* ── per-tick messages ────────────────────────────────────────────────── */

wire_status_t wire_view_decode(const uint8_t *bytes, size_t len, wire_view_t *out) {
    memset(out, 0, sizeof(*out));
    reader_t r = {bytes, len, 0};
    uint8_t done;
    const uint8_t *pad;
    if (!magic(&r, "LSTV") || !read_u32(&r, &out->tick) || !read_f32(&r, &out->reward) ||
        !read_u8(&r, &done) || !take(&r, 3, &pad))
        return WIRE_ERR;
    out->done = done != 0;
    uint32_t n;
    if (!read_u32(&r, &n))
        return WIRE_ERR;
    if ((size_t)n * 4 > r.len - r.pos)
        return WIRE_ERR;
    out->values = n ? calloc(n, sizeof(wire_blob_t)) : NULL;
    if (n && !out->values)
        return WIRE_ERR;
    out->n_values = n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t blen;
        const uint8_t *p;
        if (!read_u32(&r, &blen) || !take(&r, blen, &p)) {
            wire_view_free(out);
            return WIRE_ERR;
        }
        out->values[i].ptr = p;
        out->values[i].len = blen;
    }
    if (r.pos != r.len) { /* trailing garbage — Views are strict */
        wire_view_free(out);
        return WIRE_ERR;
    }
    return WIRE_OK;
}

void wire_view_free(wire_view_t *view) {
    free(view->values);
    memset(view, 0, sizeof(*view));
}

void wire_read_f32(const uint8_t *bytes, size_t n, float *out) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)bytes[4 * i] | ((uint32_t)bytes[4 * i + 1] << 8) |
                        ((uint32_t)bytes[4 * i + 2] << 16) | ((uint32_t)bytes[4 * i + 3] << 24);
        memcpy(&out[i], &bits, 4);
    }
}

void wire_read_i32(const uint8_t *bytes, size_t n, int32_t *out) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)bytes[4 * i] | ((uint32_t)bytes[4 * i + 1] << 8) |
                        ((uint32_t)bytes[4 * i + 2] << 16) | ((uint32_t)bytes[4 * i + 3] << 24);
        memcpy(&out[i], &bits, 4);
    }
}

/* ── input builder ────────────────────────────────────────────────────── */

static void grow(wire_input_builder_t *b, size_t need) {
    if (b->len + need <= b->cap)
        return;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->len + need)
        cap *= 2;
    b->buf = realloc(b->buf, cap);
    b->cap = cap;
}

static void put(wire_input_builder_t *b, const void *p, size_t n) {
    grow(b, n);
    memcpy(b->buf + b->len, p, n);
    b->len += n;
}

static void put_u32(wire_input_builder_t *b, uint32_t v) {
    uint8_t le[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
    put(b, le, 4);
}

void wire_input_builder_start(wire_input_builder_t *b, uint32_t n_actions) {
    memset(b, 0, sizeof(*b));
    b->n_declared = n_actions;
    put(b, "LSTA", 4);
    put_u32(b, n_actions);
}

void wire_input_builder_push_raw(wire_input_builder_t *b, const uint8_t *bytes, size_t len) {
    put_u32(b, (uint32_t)len);
    put(b, bytes, len);
    b->n_pushed++;
}

void wire_input_builder_push_f32(wire_input_builder_t *b, const wire_value_spec_t *spec,
                                 const float *values) {
    size_t len = spec->numel * wire_dtype_size(spec->dtype);
    put_u32(b, (uint32_t)len);
    grow(b, len);
    uint8_t *dst = b->buf + b->len;
    switch (spec->dtype) {
    case WIRE_F32:
        for (size_t i = 0; i < spec->numel; i++) {
            uint32_t bits;
            memcpy(&bits, &values[i], 4);
            dst[4 * i] = (uint8_t)bits;
            dst[4 * i + 1] = (uint8_t)(bits >> 8);
            dst[4 * i + 2] = (uint8_t)(bits >> 16);
            dst[4 * i + 3] = (uint8_t)(bits >> 24);
        }
        break;
    case WIRE_U8:
        for (size_t i = 0; i < spec->numel; i++)
            dst[i] = (uint8_t)roundf(values[i]);
        break;
    case WIRE_I32:
        for (size_t i = 0; i < spec->numel; i++) {
            int32_t v = (int32_t)roundf(values[i]);
            uint32_t bits;
            memcpy(&bits, &v, 4);
            dst[4 * i] = (uint8_t)bits;
            dst[4 * i + 1] = (uint8_t)(bits >> 8);
            dst[4 * i + 2] = (uint8_t)(bits >> 16);
            dst[4 * i + 3] = (uint8_t)(bits >> 24);
        }
        break;
    }
    b->len += len;
    b->n_pushed++;
}

void wire_input_builder_finish(wire_input_builder_t *b, uint8_t **ptr, size_t *len) {
    /* A count mismatch is a bug in the caller; hand back an EMPTY payload
     * (never a malformed one) so the engine plays the neutral action. */
    if (b->n_pushed != b->n_declared) {
        free(b->buf);
        *ptr = NULL;
        *len = 0;
        memset(b, 0, sizeof(*b));
        return;
    }
    *ptr = b->buf;
    *len = b->len;
    memset(b, 0, sizeof(*b));
}
