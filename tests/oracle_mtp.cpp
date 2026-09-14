// oracle_mtp.cpp -- CPU oracle for the Qwen3.8-27B MTP / NextN draft block.
//
// usage: oracle-mtp <model.gguf> <tok1> <tok2> ... [--steps N]
//
// Produces, on stdout:
//   1. a per-node dump of ONE MTP (draft block) step in exactly the textual format
//      of llama.cpp's eval-callback example (common_debug_cb_eval in common/debug.cpp),
//      so that tests/check_graph_gpu.hip::parse_oracle can consume it;
//   2. a block of grep-able `MTPORACLE: ` labelled raw numbers;
//   3. a draft-and-score loop measuring the greedy acceptance rate of the MTP head
//      against the trunk's own greedy continuation;
//   4. the exact llama_get_logits_ith readback used for one step.
//
// No GPU: n_gpu_layers = 0.
//
// Build (from the worktree root):
//   g++ -O2 -std=c++17 tests/oracle_mtp.cpp \
//     -I/home/marcelo/Projetos/llama.cpp/include -I/home/marcelo/Projetos/llama.cpp/ggml/include \
//     -L/home/marcelo/Projetos/llama.cpp/build/bin -lllama -lggml -lggml-base \
//     -Wl,-rpath,/home/marcelo/Projetos/llama.cpp/build/bin -o /tmp/oracle_mtp

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// staging API from /home/marcelo/Projetos/llama.cpp/src/llama-ext.h
// (declared here with plain C++ linkage: the exported symbols are C++-mangled)
// ---------------------------------------------------------------------------
void   llama_set_embeddings_nextn    (struct llama_context * ctx, bool value, bool masked);
float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// ---------------------------------------------------------------------------
// eval-callback dump, copied/trimmed from llama.cpp common/debug.cpp
// ---------------------------------------------------------------------------
#define INDENT "    "

struct cb_data {
    bool enabled = false;
    std::vector<uint8_t> data;
    int n_printed = 0;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static float get_f32(const uint8_t * data, const size_t * nb, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    return *(const float *) (data + i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]);
}

// same layout as common_debug_print_tensor(..., n = 3, abort_on_nan = false)
static void print_tensor_f32(const uint8_t * data, const int64_t * ne, const size_t * nb, int64_t n) {
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    sum += get_f32(data, nb, i0, i1, i2, i3);
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf(INDENT "[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) { printf(INDENT INDENT "..., \n"); i2 = ne[2] - n; }
            printf(INDENT INDENT "[\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) { printf(INDENT INDENT INDENT "..., \n"); i1 = ne[1] - n; }
                printf(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) { printf("   ..., "); i0 = ne[0] - n; }
                    printf("%12.4f", get_f32(data, nb, i0, i1, i2, i3));
                    if (i0 < ne[0] - 1) { printf(", "); }
                }
                printf("  ],\n");
            }
            printf(INDENT INDENT "],\n");
        }
        printf(INDENT "]\n");
        printf(INDENT "sum = %f\n", sum);
    }
}

// our node of interest: the cb() nodes of llama_model_qwen35::graph_mtp (+ the h input)
static bool name_is_mtp_node(const char * name) {
    if (strncmp(name, "mtp_", 4) == 0) return true;
    if (strcmp(name, "h_nextn") == 0)       return true;
    if (strcmp(name, "result_output") == 0) return true;
    return false;
}

static bool cb_eval_mtp(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cd = (cb_data *) user_data;

    if (ask) {
        return true;
    }
    if (!cd->enabled || !name_is_mtp_node(t->name)) {
        return true;
    }

    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];

    char src1_str[128] = { 0 };
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    printf("common_debug_cb_eval: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n",
            t->name, ggml_type_name(t->type), ggml_op_desc(t),
            src0 ? src0->name : "?", src0 ? ggml_ne_string(src0).c_str() : "?",
            src1 ? src1_str : "", ggml_ne_string(t).c_str());

    const size_t n_bytes = ggml_nbytes(t);

    if (t->type != GGML_TYPE_F32 || n_bytes == 0) {
        printf("    [ values not dumped: type %s, %zu bytes ]\n", ggml_type_name(t->type), n_bytes);
        printf("    sum = 0.000000\n");
        cd->n_printed++;
        return true;
    }

    const bool is_host = t->buffer && ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data = nullptr;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        cd->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cd->data.data(), 0, n_bytes);
        data = cd->data.data();
    }

    print_tensor_f32(data, t->ne, t->nb, 3);
    cd->n_printed++;

    return true;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static float row_sum(const float * v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double) v[i];
    return (float) s;
}

static int argmax_f32(const float * v, int n) {
    int bi = 0;
    float bv = v[0];
    for (int i = 1; i < n; ++i) {
        if (v[i] > bv) { bv = v[i]; bi = i; }
    }
    return bi;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <tok1> <tok2> ... [--steps N]\n", argv[0]);
        return 2;
    }

    std::vector<llama_token> prompt;
    int steps = 24;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else {
            prompt.push_back((llama_token) atoi(argv[i]));
        }
    }
    if (prompt.empty()) {
        fprintf(stderr, "no prompt tokens given\n");
        return 2;
    }
    const int n_prompt = (int) prompt.size();

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.load_mtp     = true;
    mparams.load_mode    = LLAMA_LOAD_MODE_MMAP;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "failed to load model %s\n", argv[1]);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    const int n_embd_out = llama_model_n_embd_out(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_layer_nextn = llama_model_n_layer_nextn(model);
    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_embd_inp  = llama_model_n_embd_inp(model);

    if (n_layer_nextn <= 0) {
        fprintf(stderr, "model reports n_layer_nextn = %d: no MTP block\n", n_layer_nextn);
        return 1;
    }

    const int n_ctx = 512;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = 512;
    cparams.n_ubatch   = 512;
    cparams.n_seq_max  = 1;
    cparams.n_threads  = 8;
    cparams.n_threads_batch = 8;
    cparams.embeddings = false;
    cparams.ctx_type   = LLAMA_CONTEXT_TYPE_DEFAULT;

    llama_context * ctx_tgt = llama_init_from_model(model, cparams);
    if (!ctx_tgt) { fprintf(stderr, "failed to create target context\n"); return 1; }

    cb_data cd;
    cparams.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    cparams.cb_eval         = cb_eval_mtp;
    cparams.cb_eval_user_data = &cd;

    llama_context * ctx_dft = llama_init_from_model(model, cparams);
    if (!ctx_dft) {
        fprintf(stderr, "failed to create MTP context (ctx_type = LLAMA_CONTEXT_TYPE_MTP)\n");
        return 1;
    }

    // unmasked: h_nextn rows are dense over the tokens of each decode call
    llama_set_embeddings_nextn(ctx_tgt, true, /*masked=*/ false);
    llama_set_embeddings_nextn(ctx_dft, true, /*masked=*/ false);

    fprintf(stderr, "MTPORACLE: model loaded: n_embd=%d n_embd_out=%d n_embd_inp=%d n_vocab=%d n_layer=%d n_layer_nextn=%d n_ctx_train=%d\n",
            n_embd, n_embd_out, n_embd_inp, n_vocab, n_layer, n_layer_nextn, n_ctx_train);

    // -----------------------------------------------------------------------
    // 1. trunk prefill
    //    ORACLE_PREFILL_BATCH=1 -> the whole prompt in ONE batch (what llama.cpp's
    //      own driver does in process()); with masked=false the h rows are then dense
    //      and indexed by position i (n_tokens_prev is reset on every llama_decode
    //      call, so all prompt rows must come from a single call).
    //    default -> one token per decode, i.e. the matvec path the rdna4-infer engine
    //      actually runs and the path of the `-ub 1` oracle dumps.
    // -----------------------------------------------------------------------
    std::vector<float> h((size_t) (n_prompt + steps + 4) * n_embd, 0.0f);
    auto h_row = [&](int p) -> float * { return h.data() + (size_t) p * n_embd; };

    std::vector<int>   tok((size_t) n_prompt + steps + 8, 0);
    std::vector<float> trunk_logits((size_t) n_prompt + steps + 8, 0.0f); // top-1 logit only
    std::vector<int>   trunk_argmax((size_t) n_prompt + steps + 8, -1);

    const bool batch_prefill = getenv("ORACLE_PREFILL_BATCH") != nullptr;
    printf("MTPORACLE: prefill_mode %s\n", batch_prefill ? "batch" : "token-by-token");

    {
        if (batch_prefill) {
            llama_batch b = llama_batch_init(n_prompt, 0, 1);
            b.n_tokens = n_prompt;
            for (int i = 0; i < n_prompt; ++i) {
                b.token[i]      = prompt[i];
                b.pos[i]        = i;
                b.n_seq_id[i]   = 1;
                b.seq_id[i][0]  = 0;
                b.logits[i]     = 1;
                tok[i]          = prompt[i];
            }

            const int rc = llama_decode(ctx_tgt, b);
            printf("MTPORACLE: trunk_prefill_rc %d n_tokens %d\n", rc, n_prompt);
            if (rc != 0) {
                fprintf(stderr, "trunk prefill failed rc=%d\n", rc);
                return 1;
            }

            for (int i = 0; i < n_prompt; ++i) {
                const float * hr = llama_get_embeddings_nextn_ith(ctx_tgt, i);
                memcpy(h_row(i), hr, (size_t) n_embd * sizeof(float));

                const float * lg = llama_get_logits_ith(ctx_tgt, i);
                trunk_argmax[i]  = argmax_f32(lg, n_vocab);
                trunk_logits[i]  = lg[trunk_argmax[i]];
            }
            llama_batch_free(b);
        } else {
            int rc_all = 0;
            for (int i = 0; i < n_prompt; ++i) {
                llama_batch b = llama_batch_init(1, 0, 1);
                b.n_tokens     = 1;
                b.token[0]     = prompt[i];
                b.pos[0]       = i;
                b.n_seq_id[0]  = 1;
                b.seq_id[0][0] = 0;
                b.logits[0]    = 1;
                tok[i]         = prompt[i];

                const int rc = llama_decode(ctx_tgt, b);
                if (rc != 0) { rc_all = rc; llama_batch_free(b); break; }

                // one token per call -> unmasked h rows always land at index 0
                const float * hr = llama_get_embeddings_nextn_ith(ctx_tgt, 0);
                memcpy(h_row(i), hr, (size_t) n_embd * sizeof(float));

                const float * lg = llama_get_logits_ith(ctx_tgt, 0);
                trunk_argmax[i]  = argmax_f32(lg, n_vocab);
                trunk_logits[i]  = lg[trunk_argmax[i]];

                llama_batch_free(b);
            }
            printf("MTPORACLE: trunk_prefill_rc %d n_tokens %d (one decode per token)\n", rc_all, n_prompt);
            if (rc_all != 0) {
                fprintf(stderr, "trunk prefill failed rc=%d\n", rc_all);
                return 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // 2. fill the MTP context's own KV cache for the prompt positions 0..n-2
    //    (position p gets token p and h_{p-1}; h_{-1} := 0, exactly the
    //     pending_h initialisation of llama.cpp's draft-mtp driver)
    // -----------------------------------------------------------------------
    auto make_mtp_batch = [&](int n) {
        llama_batch b = llama_batch_init(n, n_embd, 1);
        b.token = (llama_token *) malloc(sizeof(llama_token) * n); // llama_batch_init allocates only one of token/embd
        b.n_tokens = n;
        return b;
    };

    {
        const int n_pre = n_prompt - 1; // positions 0 .. n_prompt-2
        if (n_pre > 0) {
            int rc = 0;
            if (batch_prefill) {
                llama_batch b = make_mtp_batch(n_pre);
                for (int i = 0; i < n_pre; ++i) {
                    b.token[i]    = prompt[i];
                    b.pos[i]      = i;
                    b.n_seq_id[i] = 1;
                    b.seq_id[i][0] = 0;
                    b.logits[i]   = (i == n_pre - 1) ? 1 : 0;
                    // for i == 0 the paired h is h_{-1}, which does not exist -> zero vector
                    // (same as llama.cpp's draft-mtp driver, pending_h initialised to 0)
                    if (i > 0) {
                        memcpy(b.embd + (size_t) i * n_embd, h_row(i - 1), (size_t) n_embd * sizeof(float));
                    }
                }
                memset(b.embd, 0, (size_t) n_embd * sizeof(float));

                rc = llama_decode(ctx_dft, b);
                llama_batch_free(b);
            } else {
                for (int i = 0; i < n_pre; ++i) {
                    llama_batch b = make_mtp_batch(1);
                    b.token[0]     = prompt[i];
                    b.pos[0]       = i;
                    b.n_seq_id[0]  = 1;
                    b.seq_id[0][0] = 0;
                    b.logits[0]    = 1;
                    if (i > 0) {
                        memcpy(b.embd, h_row(i - 1), (size_t) n_embd * sizeof(float));
                    } else {
                        memset(b.embd, 0, (size_t) n_embd * sizeof(float));
                    }
                    rc = llama_decode(ctx_dft, b);
                    llama_batch_free(b);
                    if (rc != 0) break;
                }
            }
            printf("MTPORACLE: mtp_prefill_rc %d n_tokens %d (draft KV positions 0..%d)\n", rc, n_pre, n_pre - 1);
            if (rc != 0) {
                fprintf(stderr, "MTP draft prefill failed rc=%d\n", rc);
                return 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. ONE dumped MTP step, at the last prompt position p = n_prompt-1
    // -----------------------------------------------------------------------
    const int p_first = n_prompt - 1; // 7 for the 8-token oracle prompt

    llama_batch b1 = make_mtp_batch(1);

    auto mtp_step = [&](int p, bool dump) -> const float * {
        b1.n_tokens      = 1;
        b1.token[0]      = tok[p];
        b1.pos[0]        = p;
        b1.n_seq_id[0]   = 1;
        b1.seq_id[0][0]  = 0;
        b1.logits[0]     = 1;
        memcpy(b1.embd, h_row(p - 1), (size_t) n_embd * sizeof(float));

        cd.enabled = dump;
        const int rc = llama_decode(ctx_dft, b1);
        cd.enabled = false;

        if (rc != 0) {
            fprintf(stderr, "MTP decode at pos %d failed rc=%d\n", p, rc);
            return nullptr;
        }
        return llama_get_logits_ith(ctx_dft, 0);
    };

    printf("\n");
    printf("MTPORACLE: begin_dump mtp_step pos %d token %d (input h = trunk h at pos %d)\n", p_first, tok[p_first], p_first - 1);

    const float * dft_logits_first = mtp_step(p_first, /*dump=*/ true);
    if (!dft_logits_first) {
        fprintf(stderr, "MTP step refused to decode\n");
        return 1;
    }
    // keep a private copy (the buffer may be reallocated by later decodes)
    std::vector<float> dft_logits_first_copy(dft_logits_first, dft_logits_first + n_vocab);

    printf("MTPORACLE: end_dump nodes_printed %d\n", cd.n_printed);
    printf("\n");

    // -----------------------------------------------------------------------
    // 4. labelled raw numbers
    // -----------------------------------------------------------------------
    char meta[256] = { 0 };
    const int meta_rc = llama_model_meta_val_str(model, "qwen35.nextn_predict_layers", meta, sizeof(meta));

    printf("MTPORACLE: n_embd %d\n", n_embd);
    printf("MTPORACLE: n_embd_out %d\n", n_embd_out);
    printf("MTPORACLE: n_embd_inp %d\n", n_embd_inp);
    printf("MTPORACLE: n_vocab %d\n", n_vocab);
    printf("MTPORACLE: n_layer %d\n", n_layer);
    printf("MTPORACLE: n_layer_nextn %d\n", n_layer_nextn);
    printf("MTPORACLE: n_ctx_train %d\n", n_ctx_train);
    printf("MTPORACLE: gguf_meta qwen35.nextn_predict_layers rc %d val '%s'\n", meta_rc, meta);
    printf("MTPORACLE: prompt_n %d", n_prompt);
    for (int i = 0; i < n_prompt; ++i) printf(" %d", prompt[i]);
    printf("\n");

    for (int i = 0; i < n_prompt; ++i) {
        const float * hr = h_row(i);
        printf("MTPORACLE: h %d %.6f %.6f %.6f %.6f %.6f\n",
                i, hr[0], hr[1], hr[2], hr[3], row_sum(hr, n_embd));
    }
    for (int i = 0; i < n_prompt; ++i) {
        printf("MTPORACLE: trunk_argmax %d %d %.6f\n", i, trunk_argmax[i], trunk_logits[i]);
    }

    {
        const int darg = argmax_f32(dft_logits_first_copy.data(), n_vocab);
        printf("MTPORACLE: draft_step %d token %d h_argmax %d draft_argmax %d\n",
                p_first, tok[p_first], trunk_argmax[p_first], darg);
        printf("MTPORACLE: draft_step_detail p %d trunk_pos %d predicts_pos %d h_argmax_id %d trunk_top_logit %.6f h_row_pos_used %d token_used %d draft_argmax_id %d draft_top_logit %.6f\n",
                p_first, p_first, p_first + 1, trunk_argmax[p_first], trunk_logits[p_first],
                p_first - 1, tok[p_first], darg, dft_logits_first_copy[darg]);

        double s = 0.0;
        for (int i = 0; i < n_vocab; ++i) s += (double) dft_logits_first_copy[i];
        printf("MTPORACLE: draft_logits_sum %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                (float) s,
                dft_logits_first_copy[0], dft_logits_first_copy[1], dft_logits_first_copy[2], dft_logits_first_copy[3],
                dft_logits_first_copy[n_vocab - 4], dft_logits_first_copy[n_vocab - 3],
                dft_logits_first_copy[n_vocab - 2], dft_logits_first_copy[n_vocab - 1]);
    }

    // -----------------------------------------------------------------------
    // 5. draft-and-score loop
    //    at position p the head sees (token at p, trunk h at p-1) and predicts
    //    position p+1; the trunk predicts position p+1 from its logits at p.
    //    the trunk is advanced with its OWN greedy token (draft-and-score, not
    //    speculative decoding): tok[p+1] = argmax(trunk logits at p).
    // -----------------------------------------------------------------------
    printf("\nMTPORACLE: begin_score steps %d\n", steps);

    int accepted = 0;
    int total    = 0;
    int p        = p_first;

    // first score line reuses the already-computed step at p_first
    {
        const int darg = argmax_f32(dft_logits_first_copy.data(), n_vocab);
        const int targ = trunk_argmax[p];
        const int match = (darg == targ) ? 1 : 0;
        accepted += match; total++;
        printf("MTPORACLE: score %d tok %d trunk %d draft %d match %d\n", p, tok[p], targ, darg, match);
    }

    for (int s = 1; s < steps; ++s) {
        // advance the trunk with its own greedy token, from position p to p+1
        const int next_tok = trunk_argmax[p];
        {
            llama_batch b = llama_batch_init(1, 0, 1);
            b.n_tokens     = 1;
            b.token[0]     = next_tok;
            b.pos[0]       = p + 1;
            b.n_seq_id[0]  = 1;
            b.seq_id[0][0] = 0;
            b.logits[0]    = 1;

            const int rc = llama_decode(ctx_tgt, b);
            if (rc != 0) {
                fprintf(stderr, "trunk decode at pos %d failed rc=%d\n", p + 1, rc);
                printf("MTPORACLE: aborted_at_pos %d rc %d\n", p + 1, rc);
                break;
            }
            tok[p + 1] = next_tok;

            const float * hr = llama_get_embeddings_nextn_ith(ctx_tgt, 0);
            memcpy(h_row(p + 1), hr, (size_t) n_embd * sizeof(float));

            const float * lg = llama_get_logits_ith(ctx_tgt, 0);
            trunk_argmax[p + 1] = argmax_f32(lg, n_vocab);
            trunk_logits[p + 1] = lg[trunk_argmax[p + 1]];

            llama_batch_free(b);
        }

        p += 1;

        // one MTP step at the new position p
        const float * lg = mtp_step(p, /*dump=*/ false);
        if (!lg) {
            printf("MTPORACLE: aborted_at_pos %d mtp_decode_failed\n", p);
            break;
        }
        const int darg = argmax_f32(lg, n_vocab);
        const int targ = trunk_argmax[p];
        const int match = (darg == targ) ? 1 : 0;
        accepted += match; total++;

        printf("MTPORACLE: score %d tok %d trunk %d draft %d match %d\n", p, tok[p], targ, darg, match);
    }

    printf("MTPORACLE: acceptance %d/%d %.2f%%\n", accepted, total,
            total > 0 ? 100.0 * accepted / total : 0.0);

    // -----------------------------------------------------------------------
    // 6. exact logits readback for one step (the dumped step)
    // -----------------------------------------------------------------------
    printf("\n");
    printf("MTPORACLE: logits_readback step_pos %d api llama_get_logits_ith(ctx_dft, 0) n_vocab %d\n",
            p_first, n_vocab);
    {
        const float * lg = dft_logits_first_copy.data();
        const int am = argmax_f32(lg, n_vocab);
        double s = 0.0;
        for (int i = 0; i < n_vocab; ++i) s += (double) lg[i];
        printf("MTPORACLE: logits_readback argmax %d argmax_logit %.6f sum %.6f mean %.8f\n",
                am, lg[am], (float) s, (float) (s / n_vocab));
        printf("MTPORACLE: logits_readback first8");
        for (int i = 0; i < 8; ++i) printf(" %.6f", lg[i]);
        printf("\n");
        printf("MTPORACLE: logits_readback last4");
        for (int i = n_vocab - 4; i < n_vocab; ++i) printf(" %.6f", lg[i]);
        printf("\n");
        printf("MTPORACLE: logits_readback first8_4dp");
        for (int i = 0; i < 8; ++i) printf(" %.4f", lg[i]);
        printf("\n");
        printf("MTPORACLE: logits_readback last4_4dp");
        for (int i = n_vocab - 4; i < n_vocab; ++i) printf(" %.4f", lg[i]);
        printf("\n");

        // top-5
        std::vector<int> idx(n_vocab);
        for (int i = 0; i < n_vocab; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                [&](int a, int b) { return lg[a] > lg[b]; });
        printf("MTPORACLE: logits_readback top5");
        for (int k = 0; k < 5; ++k) printf(" %d:%.6f", idx[k], lg[idx[k]]);
        printf("\n");
    }

    // (nothing further: the trunk logits at p_first were captured during prefill)

    llama_batch_free(b1);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();

    printf("\nMTPORACLE: done\n");
    return 0;
}
