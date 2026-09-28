#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kScaleBits = 12;
constexpr uint32_t kTotal = 1u << kScaleBits;
constexpr uint32_t kSymbols = 16;
constexpr uint32_t kRansL = 1u << 23;

struct Table {
    uint32_t freq[kSymbols];
    uint32_t cum[kSymbols + 1];
};

Table build_table(const std::vector<uint8_t>& symbols, const std::vector<uint8_t>& ctx,
                  uint32_t ctx_count, uint32_t bucket) {
    Table t{};
    uint64_t counts[kSymbols] = {};
    uint64_t total = 0;
    const size_t n = symbols.size();
    for (size_t i = 0; i < n; ++i) {
        if (ctx_count > 1 && ctx[i] != bucket) continue;
        counts[symbols[i]]++;
        total++;
    }
    if (total == 0) {
        for (uint32_t s = 0; s < kSymbols; ++s) t.freq[s] = kTotal / kSymbols;
    } else {
        uint32_t assign = 0;
        for (uint32_t s = 0; s < kSymbols; ++s) {
            uint32_t f = static_cast<uint32_t>(
                (static_cast<long double>(counts[s]) * kTotal) / total);
            if (f == 0 && counts[s] > 0) f = 1;
            t.freq[s] = f;
            assign += f;
        }
        int64_t diff = static_cast<int64_t>(kTotal) - assign;
        uint32_t s = 0;
        while (diff != 0) {
            if (diff > 0) {
                t.freq[s % kSymbols]++;
                diff--;
            } else if (t.freq[s % kSymbols] > 1) {
                t.freq[s % kSymbols]--;
                diff++;
            }
            s++;
        }
    }
    uint32_t c = 0;
    for (uint32_t i = 0; i < kSymbols; ++i) {
        t.cum[i] = c;
        c += t.freq[i];
    }
    t.cum[kSymbols] = c;
    return t;
}

struct RansEncoder {
    std::vector<uint32_t> state;
    std::vector<uint8_t> out;

    void init(uint32_t s) { state.assign(s, kRansL); }

    void put(uint32_t idx, uint32_t start, uint32_t freq) {
        uint32_t x = state[idx];
        uint32_t x_max = ((kRansL >> kScaleBits) << 8) * freq;
        while (x >= x_max) {
            out.push_back(static_cast<uint8_t>(x & 0xFFu));
            x >>= 8;
        }
        state[idx] = ((x / freq) << kScaleBits) + (x % freq) + start;
#ifdef PSQ4E_DEBUG
        std::fprintf(stderr, "enc idx=%u start=%u freq=%u -> x=%u\n", idx, start, freq,
                     state[idx]);
#endif
    }

    void flush() {
        for (int i = static_cast<int>(state.size()) - 1; i >= 0; --i) {
            uint32_t x = state[static_cast<size_t>(i)];
            for (int b = 0; b < 4; ++b) {
                out.push_back(static_cast<uint8_t>(x & 0xFFu));
                x >>= 8;
            }
        }
    }
};

struct RansDecoder {
    std::vector<uint32_t> state;
    const uint8_t* ptr = nullptr;
    const uint8_t* begin = nullptr;

    void init(const uint8_t* buf, size_t bytes, uint32_t s) {
        begin = buf;
        ptr = buf + bytes;
        state.assign(s, 0);
        for (uint32_t i = 0; i < s; ++i) {
            uint32_t x = 0;
            for (int b = 0; b < 4; ++b) {
                x = (x << 8) | static_cast<uint32_t>(*--ptr);
            }
            state[i] = x;
        }
    }

    uint8_t get(uint32_t idx, const Table& t) {
        uint32_t x = state[idx];
        const uint32_t slot = x & (kTotal - 1u);
        uint32_t s = 0;
        while (t.cum[s + 1] <= slot) s++;
        uint32_t x2 = t.freq[s] * (x >> kScaleBits) + slot - t.cum[s];
        if (x2 < kRansL) {
            x2 = (x2 << 8) | static_cast<uint32_t>(*--ptr);
        }
#ifdef PSQ4E_DEBUG
        std::fprintf(stderr, "dec idx=%u x=%u slot=%u s=%u x2=%u\n", idx, x, slot, s, x2);
#endif
        state[idx] = x2;
        return static_cast<uint8_t>(s);
    }
};

std::vector<uint8_t> load_file(const char* path, size_t* size) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::fprintf(stderr, "short read\n");
        std::exit(2);
    }
    std::fclose(f);
    *size = buf.size();
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s codes.bin rows k_padded S mode(none|kprev)\n", argv[0]);
        return 2;
    }
    const uint32_t rows = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
    const uint32_t kp = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
    const uint32_t S = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
    const std::string mode = argv[5];

    size_t bytes = 0;
    std::vector<uint8_t> codes = load_file(argv[1], &bytes);
    const uint64_t nb = kp / 32u;
    const uint64_t expected = static_cast<uint64_t>(rows) * nb * 16u;
    if (bytes != expected) {
        std::fprintf(stderr, "codes size %zu != expected %llu\n", bytes,
                     static_cast<unsigned long long>(expected));
        return 2;
    }

    const uint64_t weights = static_cast<uint64_t>(rows) * kp;
    std::vector<uint8_t> sym(weights);
    std::vector<uint8_t> ctx(weights, 0);
    uint64_t w = 0;
    for (uint64_t o = 0; o < rows; ++o) {
        const uint8_t* row = codes.data() + o * nb * 16u;
        for (uint64_t ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + ib * 16u;
            for (uint32_t j = 0; j < 16u; ++j) {
                sym[w + j] = blk[j] & 0x0Fu;
                sym[w + 16u + j] = (blk[j] >> 4u) & 0x0Fu;
            }
            if (mode == "kprev") {
                for (uint32_t j = 0; j < 16u; ++j) ctx[w + j] = 0u;
                for (uint32_t j = 0; j < 16u; ++j) {
                    ctx[w + 16u + j] = sym[w + j];
                }
            }
            w += 32u;
        }
    }

    const bool ctx_mode = (mode == "kprev");
    const uint32_t table_count = ctx_mode ? 16u : 1u;
    std::vector<Table> tables(table_count);
    if (ctx_mode) {
        for (uint32_t c = 0; c < 16u; ++c) tables[c] = build_table(sym, ctx, 16u, c);
    } else {
        std::vector<uint8_t> none;
        tables[0] = build_table(sym, none, 1u, 0u);
    }

    RansEncoder enc;
    enc.init(S);
#ifdef PSQ4E_DEBUG
    std::fprintf(stderr, "freq:");
    for (uint32_t i = 0; i < 16; ++i) std::fprintf(stderr, " %u", tables[0].freq[i]);
    std::fprintf(stderr, "\nsym:");
    for (uint64_t i = 0; i < (weights < 40u ? weights : 40u); ++i)
        std::fprintf(stderr, " %u", sym[i]);
    std::fprintf(stderr, "\n");
#endif
    for (uint64_t i = weights; i-- > 0;) {
        const Table& t = tables[ctx_mode ? ctx[i] : 0u];
        enc.put(static_cast<uint32_t>(i % S), t.cum[sym[i]], t.freq[sym[i]]);
    }
    enc.flush();
    const uint64_t payload_bytes = enc.out.size();
    const uint64_t header_bytes = static_cast<uint64_t>(S) * 4u;
    const double payload_bpw = static_cast<double>(payload_bytes) * 8.0 / weights;
    const double header_bpw = static_cast<double>(static_cast<uint64_t>(S) * 4u) * 8.0 /
                              static_cast<double>(kp);
    const double offset_bpw = 33.0 * 4.0 * 8.0 / (16.0 * static_cast<double>(kp));
    const double scale_bpw = 0.5;
    const double perm_bpw = (argc > 6) ? std::atof(argv[6]) : 0.0;
    const double total_bpw =
        payload_bpw + header_bpw + offset_bpw + scale_bpw + perm_bpw;

    RansDecoder dec;
    std::vector<uint8_t> out(weights);
    dec.init(enc.out.data(), enc.out.size(), S);
    uint64_t mismatch = 0;
    for (uint64_t i = 0; i < weights; ++i) {
        out[i] = dec.get(static_cast<uint32_t>(i % S), tables[ctx_mode ? ctx[i] : 0u]);
        if (out[i] != sym[i]) mismatch++;
    }
    std::printf("rows=%u kp=%u S=%u mode=%s payload_bytes=%llu header_bytes=%llu "
                "payload_bpw=%.5f header_bpw=%.5f offset_bpw=%.5f scale_bpw=%.2f "
                "perm_bpw=%.5f total_bpw=%.5f mismatch=%llu\n",
                rows, kp, S, mode.c_str(), static_cast<unsigned long long>(payload_bytes),
                static_cast<unsigned long long>(header_bytes), payload_bpw, header_bpw,
                offset_bpw, scale_bpw, perm_bpw, total_bpw,
                static_cast<unsigned long long>(mismatch));
    return mismatch == 0 ? 0 : 1;
}
