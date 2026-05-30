static inline uniform uint64 repeat_byte(uniform uint8 value) {
    uniform uint64 fill = value;
    fill |= fill << 8;
    fill |= fill << 16;
    fill |= fill << 32;
    return fill;
}

static inline uniform uint8 toggle_pattern(uniform uint8 pattern) {
    return pattern == 0x55 ? 0xAA : 0x55;
}

export uniform uint64 membench_ispc_read(uniform uint64 words[],
                                         uniform int64 word_count,
                                         uniform uint8 tail[],
                                         uniform int64 tail_size,
                                         uniform int32 passes,
                                         uniform uint64 seed) {
    uniform int64 lane_count = (uniform int64)programCount;
    uniform int64 vector_count = word_count - (word_count % lane_count);
    uint64 accumulator = seed + (uint64)programIndex;
    uniform uint64 scalar_accumulator = 0;

    for (uniform int32 pass = 0; pass < passes; ++pass) {
        for (uniform int64 base = 0; base < vector_count; base += lane_count) {
            accumulator += streaming_load(&words[base]);
        }
        for (uniform int64 index = vector_count; index < word_count; ++index) {
            scalar_accumulator += words[index];
        }
        for (uniform int64 index = 0; index < tail_size; ++index) {
            scalar_accumulator += tail[index];
        }
    }

    return reduce_add(accumulator) + scalar_accumulator;
}

export void membench_ispc_write(uniform uint64 words[],
                                uniform int64 word_count,
                                uniform uint8 tail[],
                                uniform int64 tail_size,
                                uniform int32 passes,
                                uniform uint8 base_pattern) {
    uniform int64 lane_count = (uniform int64)programCount;
    uniform int64 vector_count = word_count - (word_count % lane_count);
    uniform uint8 pattern = base_pattern;

    for (uniform int32 pass = 0; pass < passes; ++pass) {
        uniform uint64 fill = repeat_byte(pattern);
        for (uniform int64 base = 0; base < vector_count; base += lane_count) {
            streaming_store(&words[base], fill);
        }
        for (uniform int64 index = vector_count; index < word_count; ++index) {
            words[index] = fill;
        }
        for (uniform int64 index = 0; index < tail_size; ++index) {
            tail[index] = pattern;
        }
        pattern = toggle_pattern(pattern);
    }
}

export void membench_ispc_copy(uniform uint64 a_words[],
                               uniform uint64 b_words[],
                               uniform int64 word_count,
                               uniform uint8 a_tail[],
                               uniform uint8 b_tail[],
                               uniform int64 tail_size,
                               uniform int32 passes) {
    uniform int64 lane_count = (uniform int64)programCount;
    uniform int64 vector_count = word_count - (word_count % lane_count);

    for (uniform int32 pass = 0; pass < passes; ++pass) {
        if ((pass & 1) == 0) {
            for (uniform int64 base = 0; base < vector_count; base += lane_count) {
                uint64 values = streaming_load(&a_words[base]);
                streaming_store(&b_words[base], values);
            }
            for (uniform int64 index = vector_count; index < word_count; ++index) {
                b_words[index] = a_words[index];
            }
            for (uniform int64 index = 0; index < tail_size; ++index) {
                b_tail[index] = a_tail[index];
            }
        } else {
            for (uniform int64 base = 0; base < vector_count; base += lane_count) {
                uint64 values = streaming_load(&b_words[base]);
                streaming_store(&a_words[base], values);
            }
            for (uniform int64 index = vector_count; index < word_count; ++index) {
                a_words[index] = b_words[index];
            }
            for (uniform int64 index = 0; index < tail_size; ++index) {
                a_tail[index] = b_tail[index];
            }
        }
    }
}
