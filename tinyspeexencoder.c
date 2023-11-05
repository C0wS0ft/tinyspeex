#include <stdio.h>
#include "tinyspeexencoder.h"

const spx_word16_t lag_window[11] = {
        1.00000f, 0.99716f, 0.98869f, 0.97474f, 0.95554f, 0.93140f, 0.90273f, 0.86998f, 0.83367f, 0.79434f, 0.75258f
};

const spx_word16_t lpc_window[200] = {
        0.080000f, 0.080158f, 0.080630f, 0.081418f, 0.082520f, 0.083935f, 0.085663f, 0.087703f,
        0.090052f, 0.092710f, 0.095674f, 0.098943f, 0.102514f, 0.106385f, 0.110553f, 0.115015f,
        0.119769f, 0.124811f, 0.130137f, 0.135744f, 0.141628f, 0.147786f, 0.154212f, 0.160902f,
        0.167852f, 0.175057f, 0.182513f, 0.190213f, 0.198153f, 0.206328f, 0.214731f, 0.223357f,
        0.232200f, 0.241254f, 0.250513f, 0.259970f, 0.269619f, 0.279453f, 0.289466f, 0.299651f,
        0.310000f, 0.320507f, 0.331164f, 0.341965f, 0.352901f, 0.363966f, 0.375151f, 0.386449f,
        0.397852f, 0.409353f, 0.420943f, 0.432615f, 0.444361f, 0.456172f, 0.468040f, 0.479958f,
        0.491917f, 0.503909f, 0.515925f, 0.527959f, 0.540000f, 0.552041f, 0.564075f, 0.576091f,
        0.588083f, 0.600042f, 0.611960f, 0.623828f, 0.635639f, 0.647385f, 0.659057f, 0.670647f,
        0.682148f, 0.693551f, 0.704849f, 0.716034f, 0.727099f, 0.738035f, 0.748836f, 0.759493f,
        0.770000f, 0.780349f, 0.790534f, 0.800547f, 0.810381f, 0.820030f, 0.829487f, 0.838746f,
        0.847800f, 0.856643f, 0.865269f, 0.873672f, 0.881847f, 0.889787f, 0.897487f, 0.904943f,
        0.912148f, 0.919098f, 0.925788f, 0.932214f, 0.938372f, 0.944256f, 0.949863f, 0.955189f,
        0.960231f, 0.964985f, 0.969447f, 0.973615f, 0.977486f, 0.981057f, 0.984326f, 0.987290f,
        0.989948f, 0.992297f, 0.994337f, 0.996065f, 0.997480f, 0.998582f, 0.999370f, 0.999842f,
        1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
        1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
        1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
        1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
        1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
        1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
        1.000000f, 1.000000f, 1.000000f, 0.998640f, 0.994566f, 0.987787f, 0.978324f, 0.966203f,
        0.951458f, 0.934131f, 0.914270f, 0.891931f, 0.867179f, 0.840084f, 0.810723f, 0.779182f,
        0.745551f, 0.709930f, 0.672424f, 0.633148f, 0.592223f, 0.549781f, 0.505964f, 0.460932f,
        0.414863f, 0.367968f, 0.320511f, 0.272858f, 0.225569f, 0.179655f, 0.137254f, 0.103524f
};

const float vbr_nb_thresh[9][11] = {
        {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f}, /*   CNG   */
        {4.0f,  2.5f,  2.0f,  1.2f,  0.5f,  -0.25f, -0.5f, -0.7f, -0.8f, -0.9f, -1.0f}, /*  2 kbps */
        {10.0f, 6.5f,  5.2f,  4.5f,  3.9f,  3.7f,   3.0f,  2.5f,  2.3f,  1.8f,  1.0f}, /*  6 kbps */
        {11.0f, 8.8f,  7.5f,  6.5f,  5.0f,  4.2f,   3.9f,  3.9f,  3.5f,  3.0f,  1.0f}, /*  8 kbps */
        {11.0f, 11.0f, 9.9f,  8.5f,  7.0f,  5.25f,  4.5f,  4.0f,  4.0f,  4.0f,  2.0f}, /* 11 kbps */
        {11.0f, 11.0f, 11.0f, 11.0f, 9.5f,  9.25f,  8.0f,  7.0f,  5.0f,  4.0f,  3.0f}, /* 15 kbps */
        {11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f,  9.5f,  8.5f,  6.2f,  5.2f,  5.0f}, /* 18 kbps */
        {11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f,  11.0f, 11.0f, 10.0f, 9.8f,  7.5f}, /* 24 kbps */
        {7.0f,  4.5f,  3.7f,  3.0f,  2.5f,  1.0f,   1.8f,  1.5f,  1.0f,  0.0f,  0.0f}  /*  4 kbps */
};

//
void lsp_interpolate(spx_lsp_t *old_lsp, spx_lsp_t *new_lsp, spx_lsp_t *lsp, int len, int subframe, int nb_subframes, spx_word16_t margin) {
    int i;
    float tmp = (1.0f + subframe) / nb_subframes;
    for (i = 0; i < len; i++)
        lsp[i] = (1 - tmp) * old_lsp[i] + tmp * new_lsp[i];
    /* Enforce margin to sure the LSPs are stable*/
    if (lsp[0] < LSP_SCALING * margin)
        lsp[0] = LSP_SCALING * margin;
    if (lsp[len - 1] > LSP_SCALING * (M_PI - margin))
        lsp[len - 1] = LSP_SCALING * (M_PI - margin);
    for (i = 1; i < len - 1; i++) {
        if (lsp[i] < lsp[i - 1] + LSP_SCALING * margin)
            lsp[i] = lsp[i - 1] + LSP_SCALING * margin;

        if (lsp[i] > lsp[i + 1] - LSP_SCALING * margin)
            lsp[i] = .5f * (lsp[i] + lsp[i + 1] - LSP_SCALING * margin);
    }
}

//
void signal_mul(const spx_sig_t *x, spx_sig_t *y, spx_word32_t scale, int len) {
    int i;
    for (i = 0; i < len; i++)
        y[i] = scale * x[i];
}

//
void signal_div(const spx_sig_t *x, spx_sig_t *y, spx_word32_t scale, int len) {
    int i;
    float scale_1 = 1 / scale;
    for (i = 0; i < len; i++)
        y[i] = scale_1 * x[i];
}

//
void iir_mem16(const spx_word16_t *x, const spx_coef_t *den, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack) {
    int i, j;
    spx_word16_t yi, nyi;

    for (i = 0; i < N; i++) {
        yi = EXTRACT16(SATURATE(ADD32(EXTEND32(x[i]), PSHR32(mem[0], LPC_SHIFT)), 32767));
        nyi = NEG16(yi);
        for (j = 0; j < ord - 1; j++) {
            mem[j] = MAC16_16(mem[j + 1], den[j], nyi);
        }
        mem[ord - 1] = MULT16_16(den[ord - 1], nyi);
        y[i] = yi;
    }
}

//
spx_word32_t inner_prod(const spx_word16_t *x, const spx_word16_t *y, int len) {
    spx_word32_t sum = 0;
    len >>= 2;
    while (len--) {
        spx_word32_t part = 0;
        part = MAC16_16(part, *x++, *y++);
        part = MAC16_16(part, *x++, *y++);
        part = MAC16_16(part, *x++, *y++);
        part = MAC16_16(part, *x++, *y++);
        /* HINT: If you had a 40-bit accumulator, you could shift only at the end */
        sum = ADD32(sum, SHR32(part, 6));
    }
    return sum;
}

//
static void pitch_xcorr(const spx_word16_t *_x, const spx_word16_t *_y, spx_word32_t *corr, int len, int nb_pitch, char *stack) {
    int i;
    for (i = 0; i < nb_pitch; i++) {
        /* Compute correlation*/
        corr[nb_pitch - 1 - i] = inner_prod(_x, _y + i, len);
    }
}

//
void open_loop_nbest_pitch(spx_word16_t *sw, int start, int end, int len, int *pitch, spx_word16_t *gain, int N, char *stack) {
    int i, j, k;
    VARDECL(spx_word32_t *best_score);
    VARDECL(spx_word32_t *best_ener);
    spx_word32_t e0;
    VARDECL(spx_word32_t *corr);
#ifdef FIXED_POINT
    /* In fixed-point, we need only one (temporary) array of 32-bit values and two (corr16, ener16)
      arrays for (normalized) 16-bit values */
   VARDECL(spx_word16_t *corr16);
   VARDECL(spx_word16_t *ener16);
   spx_word32_t *energy;
   int cshift=0, eshift=0;
   int scaledown = 0;
   ALLOC(corr16, end-start+1, spx_word16_t);
   ALLOC(ener16, end-start+1, spx_word16_t);
   ALLOC(corr, end-start+1, spx_word32_t);
   energy = corr;
#else
    /* In floating-point, we need to float arrays and no normalized copies */
    VARDECL(spx_word32_t *energy);
    spx_word16_t *corr16;
    spx_word16_t *ener16;
    ALLOC(energy, end - start + 2, spx_word32_t);
    ALLOC(corr, end - start + 1, spx_word32_t);
    corr16 = corr;
    ener16 = energy;
#endif

    ALLOC(best_score, N, spx_word32_t);
    ALLOC(best_ener, N, spx_word32_t);
    for (i = 0; i < N; i++) {
        best_score[i] = -1;
        best_ener[i] = 0;
        pitch[i] = start;
    }

#ifdef FIXED_POINT
    for (i=-end;i<len;i++)
   {
      if (ABS16(sw[i])>16383)
      {
         scaledown=1;
         break;
      }
   }
   /* If the weighted input is close to saturation, then we scale it down */
   if (scaledown)
   {
      for (i=-end;i<len;i++)
      {
         sw[i]=SHR16(sw[i],1);
      }
   }
#endif
    energy[0] = inner_prod(sw - start, sw - start, len);
    e0 = inner_prod(sw, sw, len);
    for (i = start; i < end; i++) {
        /* Update energy for next pitch*/
        energy[i - start + 1] = SUB32(ADD32(energy[i - start], SHR32(MULT16_16(sw[-i - 1], sw[-i - 1]), 6)),
                                      SHR32(MULT16_16(sw[-i + len - 1], sw[-i + len - 1]), 6));
        if (energy[i - start + 1] < 0)
            energy[i - start + 1] = 0;
    }

#ifdef FIXED_POINT
    eshift = normalize16(energy, ener16, 32766, end-start+1);
#endif

    /* In fixed-point, this actually overrites the energy array (aliased to corr) */
    pitch_xcorr(sw, sw - end, corr, len, end - start + 1, stack);

#ifdef FIXED_POINT
    /* Normalize to 180 so we can square it and it still fits in 16 bits */
   cshift = normalize16(corr, corr16, 180, end-start+1);
   /* If we scaled weighted input down, we need to scale it up again (OK, so we've just lost the LSB, who cares?) */
   if (scaledown)
   {
      for (i=-end;i<len;i++)
      {
         sw[i]=SHL16(sw[i],1);
      }
   }
#endif

    /* Search for the best pitch prediction gain */
    for (i = start; i <= end; i++) {
        spx_word16_t tmp = MULT16_16_16(corr16[i - start], corr16[i - start]);
        /* Instead of dividing the tmp by the energy, we multiply on the other side */
        if (MULT16_16(tmp, best_ener[N - 1]) > MULT16_16(best_score[N - 1], ADD16(1, ener16[i - start]))) {
            /* We can safely put it last and then check */
            best_score[N - 1] = tmp;
            best_ener[N - 1] = ener16[i - start] + 1;
            pitch[N - 1] = i;
            /* Check if it comes in front of others */
            for (j = 0; j < N - 1; j++) {
                if (MULT16_16(tmp, best_ener[j]) > MULT16_16(best_score[j], ADD16(1, ener16[i - start]))) {
                    for (k = N - 1; k > j; k--) {
                        best_score[k] = best_score[k - 1];
                        best_ener[k] = best_ener[k - 1];
                        pitch[k] = pitch[k - 1];
                    }
                    best_score[j] = tmp;
                    best_ener[j] = ener16[i - start] + 1;
                    pitch[j] = i;
                    break;
                }
            }
        }
    }

    /* Compute open-loop gain if necessary */
    if (gain) {
        for (j = 0; j < N; j++) {
            spx_word16_t g;
            i = pitch[j];
            g = DIV32(SHL32(EXTEND32(corr16[i - start]), cshift),
                      10 + SHR32(MULT16_16(spx_sqrt(e0), spx_sqrt(SHL32(EXTEND32(ener16[i - start]), eshift))), 6));
            /* FIXME: g = max(g,corr/energy) */
            if (g < 0)
                g = 0;
            gain[j] = g;
        }
    }
}

//
static float cheb_poly_eva(spx_word32_t *coef, spx_word16_t x, int m, char *stack) {
    int k;
    float b0, b1, tmp;

    /* Initial conditions */
    b0 = 0; /* b_(m+1) */
    b1 = 0; /* b_(m+2) */

    x *= 2;

    /* Calculate the b_(k) */
    for (k = m; k > 0; k--) {
        tmp = b0;                           /* tmp holds the previous value of b0 */
        b0 = x * b0 - b1 + coef[m - k];    /* b0 holds its new value based on b0 and b1 */
        b1 = tmp;                           /* b1 holds the previous value of b0 */
    }

    return (-b1 + .5 * x * b0 + coef[m]);
}

//
void compute_impulse_response(const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord, char *stack) {
    int i, j;
    spx_word16_t y1, ny1i, ny2i;
    VARDECL(spx_mem_t *mem1);
    VARDECL(spx_mem_t *mem2);
    ALLOC(mem1, ord, spx_mem_t);
    ALLOC(mem2, ord, spx_mem_t);

    y[0] = LPC_SCALING;
    for (i = 0; i < ord; i++)
        y[i + 1] = awk1[i];
    i++;
    for (; i < N; i++)
        y[i] = VERY_SMALL;
    for (i = 0; i < ord; i++)
        mem1[i] = mem2[i] = 0;
    for (i = 0; i < N; i++) {
        y1 = ADD16(y[i], EXTRACT16(PSHR32(mem1[0], LPC_SHIFT)));
        ny1i = NEG16(y1);
        y[i] = PSHR32(ADD32(SHL32(EXTEND32(y1), LPC_SHIFT + 1), mem2[0]), LPC_SHIFT);
        ny2i = NEG16(y[i]);
        for (j = 0; j < ord - 1; j++) {
            mem1[j] = MAC16_16(mem1[j + 1], awk2[j], ny1i);
            mem2[j] = MAC16_16(mem2[j + 1], ak[j], ny2i);
        }
        mem1[ord - 1] = MULT16_16(awk2[ord - 1], ny1i);
        mem2[ord - 1] = MULT16_16(ak[ord - 1], ny2i);
    }
}

//
void bw_lpc(spx_word16_t gamma, const spx_coef_t *lpc_in, spx_coef_t *lpc_out, int order) {
    int i;
    spx_word16_t tmp = gamma;
    for (i = 0; i < order; i++) {
        lpc_out[i] = MULT16_16_P15(tmp, lpc_in[i]);
        tmp = MULT16_16_P15(tmp, gamma);
    }
}

//
void fir_mem16(const spx_word16_t *x, const spx_coef_t *num, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack) {
    int i, j;
    spx_word16_t xi, yi;

    for (i = 0; i < N; i++) {
        xi = x[i];
        yi = EXTRACT16(SATURATE(ADD32(EXTEND32(x[i]), PSHR32(mem[0], LPC_SHIFT)), 32767));
        for (j = 0; j < ord - 1; j++) {
            mem[j] = MAC16_16(mem[j + 1], num[j], xi);
        }
        mem[ord - 1] = MULT16_16(num[ord - 1], xi);
        y[i] = yi;
    }
}

//
void filter_mem16(const spx_word16_t *x, const spx_coef_t *num, const spx_coef_t *den, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack) {
    int i, j;
    spx_word16_t xi, yi, nyi;

    for (i = 0; i < N; i++) {
        xi = x[i];
        yi = EXTRACT16(SATURATE(ADD32(EXTEND32(x[i]), PSHR32(mem[0], LPC_SHIFT)), 32767));
        nyi = NEG16(yi);

        for (j = 0; j < ord - 1; j++) {
            mem[j] = MAC16_16(MAC16_16(mem[j + 1], num[j], xi), den[j], nyi);
        }

        mem[ord - 1] = ADD32(MULT16_16(num[ord - 1], xi), MULT16_16(den[ord - 1], nyi));
        y[i] = yi;
    }
}

//
void
residue_percep_zero16(const spx_word16_t *xx, const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord,
                      char *stack) {
    int i;
    VARDECL(spx_mem_t *mem);
    ALLOC(mem, ord, spx_mem_t);
    for (i = 0; i < ord; i++)
        mem[i] = 0;
    filter_mem16(xx, ak, awk1, y, N, ord, mem, stack);
    for (i = 0; i < ord; i++)
        mem[i] = 0;
    fir_mem16(y, awk2, y, N, ord, mem, stack);
}

//
void noise_codebook_quant(
        spx_word16_t target[],            /* target vector */
        spx_coef_t ak[],            /* LPCs for this subframe */
        spx_coef_t awk1[],            /* Weighted LPCs for this subframe */
        spx_coef_t awk2[],            /* Weighted LPCs for this subframe */
        const void *par,                      /* Codebook/search parameters*/
        int p,                        /* number of LPC coeffs */
        int nsf,                      /* number of samples in subframe */
        spx_sig_t *exc,
        spx_word16_t *r,
        SpeexBits *bits,
        char *stack,
        int complexity,
        int update_target
) {
    int i;
    VARDECL(spx_word16_t *tmp);
    ALLOC(tmp, nsf, spx_word16_t);
    residue_percep_zero16(target, ak, awk1, awk2, tmp, nsf, p, stack);

    for (i = 0; i < nsf; i++)
        exc[i] += SHL32(EXTEND32(tmp[i]), 8);
    SPEEX_MEMSET(target, 0, nsf);
}

//
spx_word16_t compute_rms(const spx_sig_t *x, int len) {
    int i;
    float sum = 0;
    for (i = 0; i < len; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(.1 + sum / len);
}

//
spx_word16_t compute_rms16(const spx_word16_t *x, int len) {
    return compute_rms(x, len);
}

//
void vbr_init(VBRState *vbr) {
    int i;

    vbr->average_energy = 1600000;
    vbr->last_energy = 1;
    vbr->accum_sum = 0;
    vbr->soft_pitch = 0;
    vbr->last_pitch_coef = 0;
    vbr->last_quality = 0;

    vbr->noise_accum = .05 * pow(MIN_ENERGY, NOISE_POW);
    vbr->noise_accum_count = .05;
    vbr->noise_level = vbr->noise_accum / vbr->noise_accum_count;
    vbr->consec_noise = 0;

    for (i = 0; i < VBR_MEMORY_SIZE; i++)
        vbr->last_log_energy[i] = log(MIN_ENERGY);
}

//
float vbr_analysis(VBRState *vbr, spx_word16_t *sig, int len, int pitch, float pitch_coef) {
    int i;
    float ener = 0, ener1 = 0, ener2 = 0;
    float qual = 7;
    float log_energy;
    float non_st = 0;
    float voicing;
    float pow_ener;

    for (i = 0; i < len >> 1; i++)
        ener1 += ((float) sig[i]) * sig[i];

    for (i = len >> 1; i < len; i++)
        ener2 += ((float) sig[i]) * sig[i];
    ener = ener1 + ener2;

    log_energy = log(ener + MIN_ENERGY);
    for (i = 0; i < VBR_MEMORY_SIZE; i++)
        non_st += sqr(log_energy - vbr->last_log_energy[i]);
    non_st = non_st / (30 * VBR_MEMORY_SIZE);
    if (non_st > 1)
        non_st = 1;

    voicing = 3 * (pitch_coef - .4) * fabs(pitch_coef - .4);
    vbr->average_energy = 0.9 * vbr->average_energy + .1 * ener;
    vbr->noise_level = vbr->noise_accum / vbr->noise_accum_count;
    pow_ener = pow(ener, NOISE_POW);
    if (vbr->noise_accum_count < .06 && ener > MIN_ENERGY)
        vbr->noise_accum = .05 * pow_ener;

    if ((voicing < .3 && non_st < .2 && pow_ener < 1.2 * vbr->noise_level)
        || (voicing < .3 && non_st < .05 && pow_ener < 1.5 * vbr->noise_level)
        || (voicing < .4 && non_st < .05 && pow_ener < 1.2 * vbr->noise_level)
        || (voicing < 0 && non_st < .05)) {
        float tmp;

        vbr->consec_noise++;
        if (pow_ener > 3 * vbr->noise_level)
            tmp = 3 * vbr->noise_level;
        else
            tmp = pow_ener;
        if (vbr->consec_noise >= 4) {
            vbr->noise_accum = .95 * vbr->noise_accum + .05 * tmp;
            vbr->noise_accum_count = .95 * vbr->noise_accum_count + .05;
        }
    } else {
        vbr->consec_noise = 0;
    }

    if (pow_ener < vbr->noise_level && ener > MIN_ENERGY) {
        vbr->noise_accum = .95 * vbr->noise_accum + .05 * pow_ener;
        vbr->noise_accum_count = .95 * vbr->noise_accum_count + .05;
    }

    /* Checking for very low absolute energy */
    if (ener < 30000) {
        qual -= .7;
        if (ener < 10000)
            qual -= .7;
        if (ener < 3000)
            qual -= .7;
    } else {
        float short_diff, long_diff;
        short_diff = log((ener + 1) / (1 + vbr->last_energy));
        long_diff = log((ener + 1) / (1 + vbr->average_energy));
        /*fprintf (stderr, "%f %f\n", short_diff, long_diff);*/

        if (long_diff < -5)
            long_diff = -5;
        if (long_diff > 2)
            long_diff = 2;

        if (long_diff > 0)
            qual += .6 * long_diff;
        if (long_diff < 0)
            qual += .5 * long_diff;
        if (short_diff > 0) {
            if (short_diff > 5)
                short_diff = 5;
            qual += 1 * short_diff;
        }
        /* Checking for energy increases */
        if (ener2 > 1.6 * ener1)
            qual += .5;
    }
    vbr->last_energy = ener;
    vbr->soft_pitch = .8 * vbr->soft_pitch + .2 * pitch_coef;
    qual += 2.2 * ((pitch_coef - .4) + (vbr->soft_pitch - .4));

    if (qual < vbr->last_quality)
        qual = .5 * qual + .5 * vbr->last_quality;
    if (qual < 4)
        qual = 4;
    if (qual > 10)
        qual = 10;

    /*
    if (vbr->consec_noise>=2)
       qual-=1.3;
    if (vbr->consec_noise>=5)
       qual-=1.3;
    if (vbr->consec_noise>=12)
       qual-=1.3;
    */
    if (vbr->consec_noise >= 3)
        qual = 4;

    if (vbr->consec_noise)
        qual -= 1.0 * (log(3.0 + vbr->consec_noise) - log(3));
    if (qual < 0)
        qual = 0;

    if (ener < 1600000) {
        if (vbr->consec_noise > 2)
            qual -= 0.5 * (log(3.0 + vbr->consec_noise) - log(3));
        if (ener < 10000 && vbr->consec_noise > 2)
            qual -= 0.5 * (log(3.0 + vbr->consec_noise) - log(3));
        if (qual < 0)
            qual = 0;
        qual += .3 * log(.0001 + ener / 1600000.0);
    }
    if (qual < -1)
        qual = -1;

    /*printf ("%f %f %f %f\n", qual, voicing, non_st, pow_ener/(.01+vbr->noise_level));*/

    vbr->last_pitch_coef = pitch_coef;
    vbr->last_quality = qual;

    for (i = VBR_MEMORY_SIZE - 1; i > 0; i--)
        vbr->last_log_energy[i] = vbr->last_log_energy[i - 1];
    vbr->last_log_energy[0] = log_energy;

    /*printf ("VBR: %f %f %f %f\n", (float)(log_energy-log(vbr->average_energy+MIN_ENERGY)), non_st, voicing, vbr->noise_level);*/

    return qual;
}

//
void vbr_destroy(VBRState *vbr) {
}

void highpass(const spx_word16_t *x, spx_word16_t *y, int len, int filtID, spx_mem_t *mem) {
    int i;
#ifdef FIXED_POINT
    const spx_word16_t Pcoef[5][3] = {{16384, -31313, 14991}, {16384, -31569, 15249}, {16384, -31677, 15328}, {16384, -32313, 15947}, {16384, -22446, 6537}};
   const spx_word16_t Zcoef[5][3] = {{15672, -31344, 15672}, {15802, -31601, 15802}, {15847, -31694, 15847}, {16162, -32322, 16162}, {14418, -28836, 14418}};
#else
    const spx_word16_t Pcoef[5][3] = {{1.00000f, -1.91120f, 0.91498f},
                                      {1.00000f, -1.92683f, 0.93071f},
                                      {1.00000f, -1.93338f, 0.93553f},
                                      {1.00000f, -1.97226f, 0.97332f},
                                      {1.00000f, -1.37000f, 0.39900f}};
    const spx_word16_t Zcoef[5][3] = {{0.95654f, -1.91309f, 0.95654f},
                                      {0.96446f, -1.92879f, 0.96446f},
                                      {0.96723f, -1.93445f, 0.96723f},
                                      {0.98645f, -1.97277f, 0.98645f},
                                      {0.88000f, -1.76000f, 0.88000f}};
#endif
    const spx_word16_t *den, *num;
    if (filtID > 4)
        filtID = 4;

    den = Pcoef[filtID];
    num = Zcoef[filtID];
    /*return;*/
    for (i = 0; i < len; i++) {
        spx_word16_t yi;
        spx_word32_t vout = ADD32(MULT16_16(num[0], x[i]), mem[0]);
        yi = EXTRACT16(SATURATE(PSHR32(vout, 14), 32767));
        mem[0] = ADD32(MAC16_16(mem[1], num[1], x[i]), SHL32(MULT16_32_Q15(-den[1], vout), 1));
        mem[1] = ADD32(MULT16_16(num[2], x[i]), SHL32(MULT16_32_Q15(-den[2], vout), 1));
        y[i] = yi;
    }
}

void *nb_encoder_init(const SpeexMode *m) {
    EncState *st;
    const SpeexNBMode *mode;
    int i;

    mode = (const SpeexNBMode *) m->mode;
    st = (EncState *) speex_alloc(sizeof(EncState));
    if (!st)
        return NULL;
#if defined(VAR_ARRAYS) || defined (USE_ALLOCA)
    st->stack = NULL;
#else
    st->stack = (char *) speex_alloc_scratch(NB_ENC_STACK);
#endif

    st->mode = m;

    st->gamma1 = mode->gamma1;
    st->gamma2 = mode->gamma2;
    st->lpc_floor = mode->lpc_floor;

    st->submodes = mode->submodes;
    st->submodeID = st->submodeSelect = mode->defaultSubmode;
    st->bounded_pitch = 1;

    st->encode_submode = 1;

#ifdef VORBIS_PSYCHO
    st->psy = vorbis_psy_init(8000, 256);
   st->curve = (float*)speex_alloc(128*sizeof(float));
   st->old_curve = (float*)speex_alloc(128*sizeof(float));
   st->psy_window = (float*)speex_alloc(256*sizeof(float));
#endif

    st->cumul_gain = 1024;

    st->window = lpc_window;

    /* Create the window for autocorrelation (lag-windowing) */
    st->lagWindow = lag_window;

    st->first = 1;
    for (i = 0; i < NB_ORDER; i++)
        st->old_lsp[i] = DIV32(MULT16_16(QCONST16(3.1415927f, LSP_SHIFT), i + 1), NB_ORDER + 1);

    st->innov_rms_save = NULL;

#ifndef DISABLE_VBR
    vbr_init(&st->vbr);
    st->vbr_quality = 8;
    st->vbr_enabled = 0;
    st->vbr_max = 0;
    st->vad_enabled = 0;
    st->dtx_enabled = 0;
    st->dtx_count = 0;
    st->abr_enabled = 0;
    st->abr_drift = 0;
    st->abr_drift2 = 0;
#endif /* #ifndef DISABLE_VBR */

    st->plc_tuning = 2;
    st->complexity = 2;
    st->sampling_rate = 8000;
    st->isWideband = 0;
    st->highpass_enabled = 1;

#ifdef ENABLE_VALGRIND
    VALGRIND_MAKE_READABLE(st, NB_ENC_STACK);
#endif
    return st;
}

int nb_encoder_ctl(void *state, int request, void *ptr) {
    EncState *st;
    st = (EncState *) state;
    switch (request) {
        case SPEEX_GET_FRAME_SIZE:
            (*(spx_int32_t *) ptr) = NB_FRAME_SIZE;
            break;
        case SPEEX_SET_LOW_MODE:
        case SPEEX_SET_MODE:
            st->submodeSelect = st->submodeID = (*(spx_int32_t *) ptr);
            break;
        case SPEEX_GET_LOW_MODE:
        case SPEEX_GET_MODE:
            (*(spx_int32_t *) ptr) = st->submodeID;
            break;
#ifndef DISABLE_VBR
        case SPEEX_SET_VBR:
            st->vbr_enabled = (*(spx_int32_t *) ptr);
            break;
        case SPEEX_GET_VBR:
            (*(spx_int32_t *) ptr) = st->vbr_enabled;
            break;
        case SPEEX_SET_VAD:
            st->vad_enabled = (*(spx_int32_t *) ptr);
            break;
        case SPEEX_GET_VAD:
            (*(spx_int32_t *) ptr) = st->vad_enabled;
            break;
        case SPEEX_SET_DTX:
            st->dtx_enabled = (*(spx_int32_t *) ptr);
            break;
        case SPEEX_GET_DTX:
            (*(spx_int32_t *) ptr) = st->dtx_enabled;
            break;
        case SPEEX_SET_ABR:
            st->abr_enabled = (*(spx_int32_t *) ptr);
            st->vbr_enabled = st->abr_enabled != 0;
            if (st->vbr_enabled) {
                spx_int32_t i = 10;
                spx_int32_t rate, target;
                float vbr_qual;
                target = (*(spx_int32_t *) ptr);
                while (i >= 0) {
                    nb_encoder_ctl(st, SPEEX_SET_QUALITY, &i);
                    nb_encoder_ctl(st, SPEEX_GET_BITRATE, &rate);
                    if (rate <= target)
                        break;
                    i--;
                }
                vbr_qual = i;
                if (vbr_qual < 0)
                    vbr_qual = 0;
                nb_encoder_ctl(st, SPEEX_SET_VBR_QUALITY, &vbr_qual);
                st->abr_count = 0;
                st->abr_drift = 0;
                st->abr_drift2 = 0;
            }

            break;
        case SPEEX_GET_ABR:
            (*(spx_int32_t *) ptr) = st->abr_enabled;
            break;
#endif /* #ifndef DISABLE_VBR */
#if !defined(DISABLE_VBR) && !defined(DISABLE_FLOAT_API)
        case SPEEX_SET_VBR_QUALITY:
            st->vbr_quality = (*(float *) ptr);
            break;
        case SPEEX_GET_VBR_QUALITY:
            (*(float *) ptr) = st->vbr_quality;
            break;
#endif /* !defined(DISABLE_VBR) && !defined(DISABLE_FLOAT_API) */
        case SPEEX_SET_QUALITY: {
            int quality = (*(spx_int32_t *) ptr);
            if (quality < 0)
                quality = 0;
            if (quality > 10)
                quality = 10;
            st->submodeSelect = st->submodeID = ((const SpeexNBMode *) (st->mode->mode))->quality_map[quality];
        }
            break;
        case SPEEX_SET_COMPLEXITY:
            st->complexity = (*(spx_int32_t *) ptr);
            if (st->complexity < 0)
                st->complexity = 0;
            break;
        case SPEEX_GET_COMPLEXITY:
            (*(spx_int32_t *) ptr) = st->complexity;
            break;
        case SPEEX_SET_BITRATE: {
            spx_int32_t i = 10;
            spx_int32_t rate, target;
            target = (*(spx_int32_t *) ptr);
            while (i >= 0) {
                nb_encoder_ctl(st, SPEEX_SET_QUALITY, &i);
                nb_encoder_ctl(st, SPEEX_GET_BITRATE, &rate);
                if (rate <= target)
                    break;
                i--;
            }
        }
            break;
        case SPEEX_GET_BITRATE:
            if (st->submodes[st->submodeID])
                (*(spx_int32_t *) ptr) = st->sampling_rate * SUBMODE(bits_per_frame) / NB_FRAME_SIZE;
            else
                (*(spx_int32_t *) ptr) = st->sampling_rate * (NB_SUBMODE_BITS + 1) / NB_FRAME_SIZE;
            break;
        case SPEEX_SET_SAMPLING_RATE:
            st->sampling_rate = (*(spx_int32_t *) ptr);
            break;
        case SPEEX_GET_SAMPLING_RATE:
            (*(spx_int32_t *) ptr) = st->sampling_rate;
            break;
        case SPEEX_RESET_STATE: {
            int i;
            st->bounded_pitch = 1;
            st->first = 1;
            for (i = 0; i < NB_ORDER; i++)
                st->old_lsp[i] = DIV32(MULT16_16(QCONST16(3.1415927f, LSP_SHIFT), i + 1), NB_ORDER + 1);
            for (i = 0; i < NB_ORDER; i++)
                st->mem_sw[i] = st->mem_sw_whole[i] = st->mem_sp[i] = st->mem_exc[i] = 0;
            for (i = 0; i < NB_FRAME_SIZE + NB_PITCH_END + 1; i++)
                st->excBuf[i] = st->swBuf[i] = 0;
            for (i = 0; i < NB_WINDOW_SIZE - NB_FRAME_SIZE; i++)
                st->winBuf[i] = 0;
        }
            break;
        case SPEEX_SET_SUBMODE_ENCODING:
            st->encode_submode = (*(spx_int32_t *) ptr);
            break;
        case SPEEX_GET_SUBMODE_ENCODING:
            (*(spx_int32_t *) ptr) = st->encode_submode;
            break;
        case SPEEX_GET_LOOKAHEAD:
            (*(spx_int32_t *) ptr) = (NB_WINDOW_SIZE - NB_FRAME_SIZE);
            break;
        case SPEEX_SET_PLC_TUNING:
            st->plc_tuning = (*(spx_int32_t *) ptr);
            if (st->plc_tuning > 100)
                st->plc_tuning = 100;
            break;
        case SPEEX_GET_PLC_TUNING:
            (*(spx_int32_t *) ptr) = (st->plc_tuning);
            break;
#ifndef DISABLE_VBR
        case SPEEX_SET_VBR_MAX_BITRATE:
            st->vbr_max = (*(spx_int32_t *) ptr);
            break;
        case SPEEX_GET_VBR_MAX_BITRATE:
            (*(spx_int32_t *) ptr) = st->vbr_max;
            break;
#endif /* #ifndef DISABLE_VBR */
        case SPEEX_SET_HIGHPASS:
            st->highpass_enabled = (*(spx_int32_t *) ptr);
            break;
        case SPEEX_GET_HIGHPASS:
            (*(spx_int32_t *) ptr) = st->highpass_enabled;
            break;

            /* This is all internal stuff past this point */
        case SPEEX_GET_PI_GAIN: {
            int i;
            spx_word32_t *g = (spx_word32_t *) ptr;
            for (i = 0; i < NB_NB_SUBFRAMES; i++)
                g[i] = st->pi_gain[i];
        }
            break;
        case SPEEX_GET_EXC: {
            int i;
            for (i = 0; i < NB_NB_SUBFRAMES; i++)
                ((spx_word16_t *) ptr)[i] = compute_rms16(st->exc + i * NB_SUBFRAME_SIZE, NB_SUBFRAME_SIZE);
        }
            break;
#ifndef DISABLE_VBR
        case SPEEX_GET_RELATIVE_QUALITY:
            (*(float *) ptr) = st->relative_quality;
            break;
#endif /* #ifndef DISABLE_VBR */
        case SPEEX_SET_INNOVATION_SAVE:
            st->innov_rms_save = (spx_word16_t *) ptr;
            break;
        case SPEEX_SET_WIDEBAND:
            st->isWideband = *((spx_int32_t *) ptr);
            break;
        case SPEEX_GET_STACK:
            *((char **) ptr) = st->stack;
            break;
        default:
            //speex_warning_int("Unknown nb_ctl request: ", request);
            return -1;
    }
    return 0;
}

void nb_encoder_destroy(void *state) {
    EncState *st = (EncState *) state;
    /* Free all allocated memory */
#if !(defined(VAR_ARRAYS) || defined (USE_ALLOCA))
    speex_free_scratch(st->stack);
#endif

#ifndef DISABLE_VBR
    vbr_destroy(&st->vbr);
#endif /* #ifndef DISABLE_VBR */

#ifdef VORBIS_PSYCHO
    vorbis_psy_destroy(st->psy);
   speex_free (st->curve);
   speex_free (st->old_curve);
   speex_free (st->psy_window);
#endif

    /*Free state memory... should be last*/
    speex_free(st);
}

//
int scal_quant(spx_word16_t in, const spx_word16_t *boundary, int entries) {
    int i = 0;
    while (i < entries - 1 && in > boundary[0]) {
        boundary++;
        i++;
    }
    return i;
}

//
void _spx_autocorr(
        const spx_word16_t *x,   /*  in: [0...n-1] samples x   */
        float *ac,  /* out: [0...lag-1] ac values */
        int lag,
        int n
) {
    float d;
    int i;
    while (lag--) {
        for (i = lag, d = 0; i < n; i++)
            d += x[i] * x[i - lag];
        ac[lag] = d;
    }
    ac[0] += 10;
}

//
spx_word32_t _spx_lpc(
        spx_coef_t *lpc, /* out: [0...p-1] LPC coefficients      */
        const spx_word16_t *ac,  /* in:  [0...p] autocorrelation values  */
        int p
) {
    int i, j;
    spx_word16_t r;
    spx_word16_t error = ac[0];

    for (i = 0; i < p; i++) {

        /* Sum up this iteration's reflection coefficient */
        spx_word32_t rr = NEG32(SHL32(EXTEND32(ac[i + 1]), 13));
        for (j = 0; j < i; j++)
            rr = SUB32(rr, MULT16_16(lpc[j], ac[i - j]));
#ifdef FIXED_POINT
        r = DIV32_16(rr+PSHR32(error,1),ADD16(error,8));
#else
        r = rr / (error + .003 * ac[0]);
#endif
        /*  Update LPC coefficients and total error */
        lpc[i] = r;
        for (j = 0; j < (i + 1) >> 1; j++) {
            spx_word16_t tmp1, tmp2;
            /* It could be that j == i-1-j, in which case, we're updating the same value twice, which is OK */
            tmp1 = lpc[j];
            tmp2 = lpc[i - 1 - j];
            lpc[j] = MAC16_16_P13(tmp1, r, tmp2);
            lpc[i - 1 - j] = MAC16_16_P13(tmp2, r, tmp1);
        }

        error = SUB16(error, MULT16_16_Q13(r, MULT16_16_Q13(error, r)));
    }
    return error;
}

//
void lsp_to_lpc(const spx_lsp_t *freq, spx_coef_t *ak, int lpcrdr, char *stack)
/*  float *freq 	array of LSP frequencies in the x domain	*/
/*  float *ak 		array of LPC coefficients 			*/
/*  int lpcrdr  	order of LPC coefficients 			*/
{
    int i, j;
    float xout1, xout2, xin1, xin2;
    VARDECL(float *Wp);
    float *pw, *n1, *n2, *n3, *n4 = NULL;
    VARDECL(float *x_freq);
    int m = lpcrdr >> 1;

    ALLOC(Wp, 4 * m + 2, float);
    pw = Wp;

    /* initialise contents of array */

    for (i = 0; i <= 4 * m + 1; i++) {        /* set contents of buffer to 0 */
        *pw++ = 0.0;
    }

    /* Set pointers up */

    pw = Wp;
    xin1 = 1.0;
    xin2 = 1.0;

    ALLOC(x_freq, lpcrdr, float);
    for (i = 0; i < lpcrdr; i++)
        x_freq[i] = ANGLE2X(freq[i]);

    /* reconstruct P(z) and Q(z) by  cascading second order
      polynomials in form 1 - 2xz(-1) +z(-2), where x is the
      LSP coefficient */

    for (j = 0; j <= lpcrdr; j++) {
        int i2 = 0;
        for (i = 0; i < m; i++, i2 += 2) {
            n1 = pw + (i * 4);
            n2 = n1 + 1;
            n3 = n2 + 1;
            n4 = n3 + 1;
            xout1 = xin1 - 2.f * x_freq[i2] * *n1 + *n2;
            xout2 = xin2 - 2.f * x_freq[i2 + 1] * *n3 + *n4;
            *n2 = *n1;
            *n4 = *n3;
            *n1 = xin1;
            *n3 = xin2;
            xin1 = xout1;
            xin2 = xout2;
        }
        xout1 = xin1 + *(n4 + 1);
        xout2 = xin2 - *(n4 + 2);
        if (j > 0)
            ak[j - 1] = (xout1 + xout2) * 0.5f;
        *(n4 + 1) = xin1;
        *(n4 + 2) = xin2;

        xin1 = 0.0;
        xin2 = 0.0;
    }
}

int lpc_to_lsp (spx_coef_t *a,int lpcrdr,spx_lsp_t *freq,int nb,spx_word16_t delta, char *stack)
/*  float *a 		     	lpc coefficients			*/
/*  int lpcrdr			order of LPC coefficients (10) 		*/
/*  float *freq 	      	LSP frequencies in the x domain       	*/
/*  int nb			number of sub-intervals (4) 		*/
/*  float delta			grid spacing interval (0.02) 		*/


{
    spx_word16_t temp_xr,xl,xr,xm=0;
    spx_word32_t psuml,psumr,psumm,temp_psumr/*,temp_qsumr*/;
    int i,j,m,k;
    VARDECL(spx_word32_t *Q);                 	/* ptrs for memory allocation 		*/
    VARDECL(spx_word32_t *P);
    VARDECL(spx_word16_t *Q16);         /* ptrs for memory allocation 		*/
    VARDECL(spx_word16_t *P16);
    spx_word32_t *px;                	/* ptrs of respective P'(z) & Q'(z)	*/
    spx_word32_t *qx;
    spx_word32_t *p;
    spx_word32_t *q;
    spx_word16_t *pt;                	/* ptr used for cheb_poly_eval()
				whether P' or Q' 			*/
    int roots=0;              	/* DR 8/2/94: number of roots found 	*/
    m = lpcrdr/2;            	/* order of P'(z) & Q'(z) polynomials 	*/

    /* Allocate memory space for polynomials */
    ALLOC(Q, (m+1), spx_word32_t);
    ALLOC(P, (m+1), spx_word32_t);

    /* determine P'(z)'s and Q'(z)'s coefficients where
      P'(z) = P(z)/(1 + z^(-1)) and Q'(z) = Q(z)/(1-z^(-1)) */

    px = P;                      /* initialise ptrs 			*/
    qx = Q;
    p = px;
    q = qx;

#ifdef FIXED_POINT
    *px++ = LPC_SCALING;
    *qx++ = LPC_SCALING;
    for(i=0;i<m;i++){
       *px++ = SUB32(ADD32(EXTEND32(a[i]),EXTEND32(a[lpcrdr-i-1])), *p++);
       *qx++ = ADD32(SUB32(EXTEND32(a[i]),EXTEND32(a[lpcrdr-i-1])), *q++);
    }
    px = P;
    qx = Q;
    for(i=0;i<m;i++)
    {
       /*if (fabs(*px)>=32768)
          speex_warning_int("px", *px);
       if (fabs(*qx)>=32768)
       speex_warning_int("qx", *qx);*/
       *px = PSHR32(*px,2);
       *qx = PSHR32(*qx,2);
       px++;
       qx++;
    }
    /* The reason for this lies in the way cheb_poly_eva() is implemented for fixed-point */
    P[m] = PSHR32(P[m],3);
    Q[m] = PSHR32(Q[m],3);
#else
    *px++ = LPC_SCALING;
    *qx++ = LPC_SCALING;
    for(i=0;i<m;i++){
        *px++ = (a[i]+a[lpcrdr-1-i]) - *p++;
        *qx++ = (a[i]-a[lpcrdr-1-i]) + *q++;
    }
    px = P;
    qx = Q;
    for(i=0;i<m;i++){
        *px = 2**px;
        *qx = 2**qx;
        px++;
        qx++;
    }
#endif

    px = P;             	/* re-initialise ptrs 			*/
    qx = Q;

    /* now that we have computed P and Q convert to 16 bits to
       speed up cheb_poly_eval */

    ALLOC(P16, m+1, spx_word16_t);
    ALLOC(Q16, m+1, spx_word16_t);

    for (i=0;i<m+1;i++)
    {
        P16[i] = P[i];
        Q16[i] = Q[i];
    }

    /* Search for a zero in P'(z) polynomial first and then alternate to Q'(z).
    Keep alternating between the two polynomials as each zero is found 	*/

    xr = 0;             	/* initialise xr to zero 		*/
    xl = FREQ_SCALE;               	/* start at point xl = 1 		*/

    for(j=0;j<lpcrdr;j++){
        if(j&1)            	/* determines whether P' or Q' is eval. */
            pt = Q16;
        else
            pt = P16;

        psuml = cheb_poly_eva(pt,xl,m,stack);	/* evals poly. at xl 	*/

        while(xr >= -FREQ_SCALE){
            spx_word16_t dd;
            /* Modified by JMV to provide smaller steps around x=+-1 */
#ifdef FIXED_POINT
            dd = MULT16_16_Q15(delta,SUB16(FREQ_SCALE, MULT16_16_Q14(MULT16_16_Q14(xl,xl),14000)));
           if (psuml<512 && psuml>-512)
              dd = PSHR16(dd,1);
#else
            dd=delta*(1-.9*xl*xl);
            if (fabs(psuml)<.2)
                dd *= .5;
#endif
            xr = SUB16(xl, dd);                        	/* interval spacing 	*/
            psumr = cheb_poly_eva(pt,xr,m,stack);/* poly(xl-delta_x) 	*/
            temp_psumr = psumr;
            temp_xr = xr;

            /* if no sign change increment xr and re-evaluate poly(xr). Repeat til
            sign change.
            if a sign change has occurred the interval is bisected and then
            checked again for a sign change which determines in which
            interval the zero lies in.
            If there is no sign change between poly(xm) and poly(xl) set interval
            between xm and xr else set interval between xl and xr and repeat till
            root is located within the specified limits 			*/

            if(SIGN_CHANGE(psumr,psuml))
            {
                roots++;

                psumm=psuml;
                for(k=0;k<=nb;k++){
#ifdef FIXED_POINT
                    xm = ADD16(PSHR16(xl,1),PSHR16(xr,1));        	/* bisect the interval 	*/
#else
                    xm = .5*(xl+xr);        	/* bisect the interval 	*/
#endif
                    psumm=cheb_poly_eva(pt,xm,m,stack);
                    /*if(psumm*psuml>0.)*/
                    if(!SIGN_CHANGE(psumm,psuml))
                    {
                        psuml=psumm;
                        xl=xm;
                    } else {
                        psumr=psumm;
                        xr=xm;
                    }
                }

                /* once zero is found, reset initial interval to xr 	*/
                freq[j] = X2ANGLE(xm);
                xl = xm;
                break;
            }
            else{
                psuml=temp_psumr;
                xl=temp_xr;
            }
        }
    }
    return(roots);
}

int nb_encode(void *state, void *vin, SpeexBits *bits) {
    EncState *st;
    int i, sub, roots;
    int ol_pitch;
    spx_word16_t ol_pitch_coef;
    spx_word32_t ol_gain;

    VARDECL(spx_word16_t *target);
    VARDECL(spx_sig_t *innov);
    VARDECL(spx_word32_t *exc32);
    VARDECL(spx_mem_t *mem);
    VARDECL(spx_coef_t *bw_lpc1);
    VARDECL(spx_coef_t *bw_lpc2);
    VARDECL(spx_coef_t *lpc);
    VARDECL(spx_lsp_t *lsp);
    VARDECL(spx_lsp_t *qlsp);
    VARDECL(spx_lsp_t *interp_lsp);
    VARDECL(spx_lsp_t *interp_qlsp);
    VARDECL(spx_coef_t *interp_lpc);
    VARDECL(spx_coef_t *interp_qlpc);
    char *stack;
    VARDECL(spx_word16_t *syn_resp);

    spx_word32_t ener = 0;
    spx_word16_t fine_gain;
    spx_word16_t *in = (spx_word16_t *) vin;

    st = (EncState *) state;
    stack = st->stack;

    ALLOC(lpc, NB_ORDER, spx_coef_t);
    ALLOC(bw_lpc1, NB_ORDER, spx_coef_t);
    ALLOC(bw_lpc2, NB_ORDER, spx_coef_t);
    ALLOC(lsp, NB_ORDER, spx_lsp_t);
    ALLOC(qlsp, NB_ORDER, spx_lsp_t);
    ALLOC(interp_lsp, NB_ORDER, spx_lsp_t);
    ALLOC(interp_qlsp, NB_ORDER, spx_lsp_t);
    ALLOC(interp_lpc, NB_ORDER, spx_coef_t);
    ALLOC(interp_qlpc, NB_ORDER, spx_coef_t);

    st->exc = st->excBuf + NB_PITCH_END + 2;
    st->sw = st->swBuf + NB_PITCH_END + 2;
    /* Move signals 1 frame towards the past */
    SPEEX_MOVE(st->excBuf, st->excBuf + NB_FRAME_SIZE, NB_PITCH_END + 2);
    SPEEX_MOVE(st->swBuf, st->swBuf + NB_FRAME_SIZE, NB_PITCH_END + 2);

    if (st->highpass_enabled)
        highpass(in, in, NB_FRAME_SIZE, (st->isWideband ? HIGHPASS_WIDEBAND : HIGHPASS_NARROWBAND) | HIGHPASS_INPUT, st->mem_hp);

    {
        VARDECL(spx_word16_t *w_sig);
        VARDECL(spx_word16_t *autocorr);
        ALLOC(w_sig, NB_WINDOW_SIZE, spx_word16_t);
        ALLOC(autocorr, NB_ORDER + 1, spx_word16_t);
        /* Window for analysis */
        for (i = 0; i < NB_WINDOW_SIZE - NB_FRAME_SIZE; i++)
            w_sig[i] = MULT16_16_Q15(st->winBuf[i], st->window[i]);
        for (; i < NB_WINDOW_SIZE; i++)
            w_sig[i] = MULT16_16_Q15(in[i - NB_WINDOW_SIZE + NB_FRAME_SIZE], st->window[i]);
        /* Compute auto-correlation */
        _spx_autocorr(w_sig, autocorr, NB_ORDER + 1, NB_WINDOW_SIZE);
        autocorr[0] = ADD16(autocorr[0], MULT16_16_Q15(autocorr[0], st->lpc_floor)); /* Noise floor in auto-correlation domain */

        /* Lag windowing: equivalent to filtering in the power-spectrum domain */
        for (i = 0; i < NB_ORDER + 1; i++)
            autocorr[i] = MULT16_16_Q15(autocorr[i], st->lagWindow[i]);
        autocorr[0] = ADD16(autocorr[0], 1);

        /* Levinson-Durbin */
        _spx_lpc(lpc, autocorr, NB_ORDER);
        /* LPC to LSPs (x-domain) transform */
        roots = lpc_to_lsp(lpc, NB_ORDER, lsp, 10, LSP_DELTA1, stack);
        /* Check if we found all the roots */
        if (roots != NB_ORDER) {
            /*If we can't find all LSP's, do some damage control and use previous filter*/
            for (i = 0; i < NB_ORDER; i++) {
                lsp[i] = st->old_lsp[i];
            }
        }
    }




    /* Whole frame analysis (open-loop estimation of pitch and excitation gain) */
    {
        int diff = NB_WINDOW_SIZE - NB_FRAME_SIZE;
        if (st->first)
            for (i = 0; i < NB_ORDER; i++)
                interp_lsp[i] = lsp[i];
        else
            lsp_interpolate(st->old_lsp, lsp, interp_lsp, NB_ORDER, NB_NB_SUBFRAMES, NB_NB_SUBFRAMES << 1, LSP_MARGIN);

        /* Compute interpolated LPCs (unquantized) for whole frame*/
        lsp_to_lpc(interp_lsp, interp_lpc, NB_ORDER, stack);


        /*Open-loop pitch*/
        if (!st->submodes[st->submodeID] || (st->complexity > 2 && SUBMODE(have_subframe_gain) < 3) || SUBMODE(forced_pitch_gain) ||
            SUBMODE(lbr_pitch) != -1
            #ifndef DISABLE_VBR
            || st->vbr_enabled || st->vad_enabled
#endif
                ) {
            int nol_pitch[6];
            spx_word16_t nol_pitch_coef[6];

            bw_lpc(0.9, interp_lpc, bw_lpc1, NB_ORDER);
            bw_lpc(0.55, interp_lpc, bw_lpc2, NB_ORDER);

            SPEEX_COPY(st->sw, st->winBuf, diff);
            SPEEX_COPY(st->sw + diff, in, NB_FRAME_SIZE - diff);
            filter10(st->sw, bw_lpc1, bw_lpc2, st->sw, NB_FRAME_SIZE, st->mem_sw_whole, stack);

            open_loop_nbest_pitch(st->sw, NB_PITCH_START, NB_PITCH_END, NB_FRAME_SIZE,
                                  nol_pitch, nol_pitch_coef, 6, stack);
            ol_pitch = nol_pitch[0];
            ol_pitch_coef = nol_pitch_coef[0];
            /*Try to remove pitch multiples*/
            for (i = 1; i < 6; i++) {
#ifdef FIXED_POINT
                if ((nol_pitch_coef[i]>MULT16_16_Q15(nol_pitch_coef[0],27853)) &&
#else
                if ((nol_pitch_coef[i] > .85 * nol_pitch_coef[0]) &&
                    #endif
                    (ABS(2 * nol_pitch[i] - ol_pitch) <= 2 || ABS(3 * nol_pitch[i] - ol_pitch) <= 3 ||
                     ABS(4 * nol_pitch[i] - ol_pitch) <= 4 || ABS(5 * nol_pitch[i] - ol_pitch) <= 5)) {
                    /*ol_pitch_coef=nol_pitch_coef[i];*/
                    ol_pitch = nol_pitch[i];
                }
            }
            /*if (ol_pitch>50)
              ol_pitch/=2;*/
            /*ol_pitch_coef = sqrt(ol_pitch_coef);*/

        } else {
            ol_pitch = 0;
            ol_pitch_coef = 0;
        }

        /*Compute "real" excitation*/
        /*SPEEX_COPY(st->exc, st->winBuf, diff);
        SPEEX_COPY(st->exc+diff, in, NB_FRAME_SIZE-diff);*/
        fir_mem16(st->winBuf, interp_lpc, st->exc, diff, NB_ORDER, st->mem_exc, stack);
        fir_mem16(in, interp_lpc, st->exc + diff, NB_FRAME_SIZE - diff, NB_ORDER, st->mem_exc, stack);

        /* Compute open-loop excitation gain */
        {
            spx_word16_t g = compute_rms16(st->exc, NB_FRAME_SIZE);
            if (st->submodeID != 1 && ol_pitch > 0)
                ol_gain = MULT16_16(g, MULT16_16_Q14(QCONST16(1.1, 14),
                                                     spx_sqrt(QCONST32(1., 28) -
                                                              MULT16_32_Q15(QCONST16(.8, 15), SHL32(MULT16_16(ol_pitch_coef, ol_pitch_coef), 16)))));
            else
                ol_gain = SHL32(EXTEND32(g), SIG_SHIFT);
        }
    }

#ifdef VORBIS_PSYCHO
    SPEEX_MOVE(st->psy_window, st->psy_window+NB_FRAME_SIZE, 256-NB_FRAME_SIZE);
   SPEEX_COPY(&st->psy_window[256-NB_FRAME_SIZE], in, NB_FRAME_SIZE);
   compute_curve(st->psy, st->psy_window, st->curve);
   /*print_vec(st->curve, 128, "curve");*/
   if (st->first)
      SPEEX_COPY(st->old_curve, st->curve, 128);
#endif

    /*VBR stuff*/
#ifndef DISABLE_VBR
    if (st->vbr_enabled || st->vad_enabled) {
        float lsp_dist = 0;
        for (i = 0; i < NB_ORDER; i++)
            lsp_dist += (st->old_lsp[i] - lsp[i]) * (st->old_lsp[i] - lsp[i]);
        lsp_dist /= LSP_SCALING * LSP_SCALING;

        if (st->abr_enabled) {
            float qual_change = 0;
            if (st->abr_drift2 * st->abr_drift > 0) {
                /* Only adapt if long-term and short-term drift are the same sign */
                qual_change = -.00001 * st->abr_drift / (1 + st->abr_count);
                if (qual_change > .05)
                    qual_change = .05;
                if (qual_change < -.05)
                    qual_change = -.05;
            }
            st->vbr_quality += qual_change;
            if (st->vbr_quality > 10)
                st->vbr_quality = 10;
            if (st->vbr_quality < 0)
                st->vbr_quality = 0;
        }

        st->relative_quality = vbr_analysis(&st->vbr, in, NB_FRAME_SIZE, ol_pitch, GAIN_SCALING_1 * ol_pitch_coef);
        /*if (delta_qual<0)*/
        /*  delta_qual*=.1*(3+st->vbr_quality);*/
        if (st->vbr_enabled) {
            spx_int32_t mode;
            int choice = 0;
            float min_diff = 100;
            mode = 8;
            while (mode) {
                int v1;
                float thresh;
                v1 = (int) floor(st->vbr_quality);
                if (v1 == 10)
                    thresh = vbr_nb_thresh[mode][v1];
                else
                    thresh = (st->vbr_quality - v1) * vbr_nb_thresh[mode][v1 + 1] + (1 + v1 - st->vbr_quality) * vbr_nb_thresh[mode][v1];
                if (st->relative_quality > thresh &&
                    st->relative_quality - thresh < min_diff) {
                    choice = mode;
                    min_diff = st->relative_quality - thresh;
                }
                mode--;
            }
            mode = choice;
            if (mode == 0) {
                if (st->dtx_count == 0 || lsp_dist > .05 || !st->dtx_enabled || st->dtx_count > 20) {
                    mode = 1;
                    st->dtx_count = 1;
                } else {
                    mode = 0;
                    st->dtx_count++;
                }
            } else {
                st->dtx_count = 0;
            }

            nb_encoder_ctl(state, SPEEX_SET_MODE, &mode);
            if (st->vbr_max > 0) {
                spx_int32_t rate;
                nb_encoder_ctl(state, SPEEX_GET_BITRATE, &rate);
                if (rate > st->vbr_max) {
                    rate = st->vbr_max;
                    nb_encoder_ctl(state, SPEEX_SET_BITRATE, &rate);
                }
            }

            if (st->abr_enabled) {
                spx_int32_t bitrate;
                nb_encoder_ctl(state, SPEEX_GET_BITRATE, &bitrate);
                st->abr_drift += (bitrate - st->abr_enabled);
                st->abr_drift2 = .95 * st->abr_drift2 + .05 * (bitrate - st->abr_enabled);
                st->abr_count += 1.0;
            }

        } else {
            /*VAD only case*/
            int mode;
            if (st->relative_quality < 2) {
                if (st->dtx_count == 0 || lsp_dist > .05 || !st->dtx_enabled || st->dtx_count > 20) {
                    st->dtx_count = 1;
                    mode = 1;
                } else {
                    mode = 0;
                    st->dtx_count++;
                }
            } else {
                st->dtx_count = 0;
                mode = st->submodeSelect;
            }
            /*speex_encoder_ctl(state, SPEEX_SET_MODE, &mode);*/
            st->submodeID = mode;
        }
    } else {
        st->relative_quality = -1;
    }
#endif /* #ifndef DISABLE_VBR */

    if (st->encode_submode) {
        /* First, transmit a zero for narrowband */
        speex_bits_pack(bits, 0, 1);

        /* Transmit the sub-mode we use for this frame */
        speex_bits_pack(bits, st->submodeID, NB_SUBMODE_BITS);

    }

    /* If null mode (no transmission), just set a couple things to zero*/
    if (st->submodes[st->submodeID] == NULL) {
        for (i = 0; i < NB_FRAME_SIZE; i++)
            st->exc[i] = st->sw[i] = VERY_SMALL;

        for (i = 0; i < NB_ORDER; i++)
            st->mem_sw[i] = 0;
        st->first = 1;
        st->bounded_pitch = 1;

        SPEEX_COPY(st->winBuf, in + 2 * NB_FRAME_SIZE - NB_WINDOW_SIZE, NB_WINDOW_SIZE - NB_FRAME_SIZE);

        /* Clear memory (no need to really compute it) */
        for (i = 0; i < NB_ORDER; i++)
            st->mem_sp[i] = 0;
        return 0;

    }

    /* LSP Quantization */
    if (st->first) {
        for (i = 0; i < NB_ORDER; i++)
            st->old_lsp[i] = lsp[i];
    }


        /*Quantize LSPs*/
#if 1 /*0 for unquantized*/
    SUBMODE(lsp_quant)(lsp, qlsp, NB_ORDER, bits);
#else
    for (i=0;i<NB_ORDER;i++)
     qlsp[i]=lsp[i];
#endif

    /*If we use low bit-rate pitch mode, transmit open-loop pitch*/
    if (SUBMODE(lbr_pitch) != -1) {
        speex_bits_pack(bits, ol_pitch - NB_PITCH_START, 7);
    }

    if (SUBMODE(forced_pitch_gain)) {
        int quant;
        /* This just damps the pitch a bit, because it tends to be too aggressive when forced */
        ol_pitch_coef = MULT16_16_Q15(QCONST16(.9, 15), ol_pitch_coef);
#ifdef FIXED_POINT
        quant = PSHR16(MULT16_16_16(15, ol_pitch_coef),GAIN_SHIFT);
#else
        quant = (int) floor(.5 + 15 * ol_pitch_coef * GAIN_SCALING_1);
#endif
        if (quant > 15)
            quant = 15;
        if (quant < 0)
            quant = 0;
        speex_bits_pack(bits, quant, 4);
        ol_pitch_coef = MULT16_16_P15(QCONST16(0.066667, 15), SHL16(quant, GAIN_SHIFT));
    }


    /*Quantize and transmit open-loop excitation gain*/
#ifdef FIXED_POINT
    {
      int qe = scal_quant32(ol_gain, ol_gain_table, 32);
      /*ol_gain = exp(qe/3.5)*SIG_SCALING;*/
      ol_gain = MULT16_32_Q15(28406,ol_gain_table[qe]);
      speex_bits_pack(bits, qe, 5);
   }
#else
    {
        int qe = (int) (floor(.5 + 3.5 * log(ol_gain * 1.0 / SIG_SCALING)));
        if (qe < 0)
            qe = 0;
        if (qe > 31)
            qe = 31;
        ol_gain = exp(qe / 3.5) * SIG_SCALING;
        speex_bits_pack(bits, qe, 5);
    }
#endif



    /* Special case for first frame */
    if (st->first) {
        for (i = 0; i < NB_ORDER; i++)
            st->old_qlsp[i] = qlsp[i];
    }

    /* Target signal */
    ALLOC(target, NB_SUBFRAME_SIZE, spx_word16_t);
    ALLOC(innov, NB_SUBFRAME_SIZE, spx_sig_t);
    ALLOC(exc32, NB_SUBFRAME_SIZE, spx_word32_t);
    ALLOC(syn_resp, NB_SUBFRAME_SIZE, spx_word16_t);
    ALLOC(mem, NB_ORDER, spx_mem_t);

    /* Loop on sub-frames */
    for (sub = 0; sub < NB_NB_SUBFRAMES; sub++) {
        int offset;
        spx_word16_t *sw;
        spx_word16_t *exc, *inBuf;
        int pitch;
        int response_bound = NB_SUBFRAME_SIZE;

        /* Offset relative to start of frame */
        offset = NB_SUBFRAME_SIZE * sub;
        /* Excitation */
        exc = st->exc + offset;
        /* Weighted signal */
        sw = st->sw + offset;

        /* LSP interpolation (quantized and unquantized) */
        lsp_interpolate(st->old_lsp, lsp, interp_lsp, NB_ORDER, sub, NB_NB_SUBFRAMES, LSP_MARGIN);
        lsp_interpolate(st->old_qlsp, qlsp, interp_qlsp, NB_ORDER, sub, NB_NB_SUBFRAMES, LSP_MARGIN);

        /* Compute interpolated LPCs (quantized and unquantized) */
        lsp_to_lpc(interp_lsp, interp_lpc, NB_ORDER, stack);

        lsp_to_lpc(interp_qlsp, interp_qlpc, NB_ORDER, stack);

        /* Compute analysis filter gain at w=pi (for use in SB-CELP) */
        {
            spx_word32_t pi_g = LPC_SCALING;
            for (i = 0; i < NB_ORDER; i += 2) {
                /*pi_g += -st->interp_qlpc[i] +  st->interp_qlpc[i+1];*/
                pi_g = ADD32(pi_g, SUB32(EXTEND32(interp_qlpc[i + 1]), EXTEND32(interp_qlpc[i])));
            }
            st->pi_gain[sub] = pi_g;
        }

#ifdef VORBIS_PSYCHO
        {
         float curr_curve[128];
         float fact = ((float)sub+1.0f)/NB_NB_SUBFRAMES;
         for (i=0;i<128;i++)
            curr_curve[i] = (1.0f-fact)*st->old_curve[i] + fact*st->curve[i];
         curve_to_lpc(st->psy, curr_curve, bw_lpc1, bw_lpc2, 10);
      }
#else
        /* Compute bandwidth-expanded (unquantized) LPCs for perceptual weighting */
        bw_lpc(st->gamma1, interp_lpc, bw_lpc1, NB_ORDER);
        bw_lpc(st->gamma2, interp_lpc, bw_lpc2, NB_ORDER);
        /*print_vec(st->bw_lpc1, 10, "bw_lpc");*/
#endif

        /*FIXME: This will break if we change the window size */
                speex_assert(NB_WINDOW_SIZE - NB_FRAME_SIZE == NB_SUBFRAME_SIZE);
        if (sub == 0)
            inBuf = st->winBuf;
        else
            inBuf = &in[((sub - 1) * NB_SUBFRAME_SIZE)];
        for (i = 0; i < NB_SUBFRAME_SIZE; i++)
            sw[i] = inBuf[i];

        if (st->complexity == 0)
            response_bound >>= 1;
        compute_impulse_response(interp_qlpc, bw_lpc1, bw_lpc2, syn_resp, response_bound, NB_ORDER, stack);
        for (i = response_bound; i < NB_SUBFRAME_SIZE; i++)
            syn_resp[i] = VERY_SMALL;

        /* Compute zero response of A(z/g1) / ( A(z/g2) * A(z) ) */
        for (i = 0; i < NB_ORDER; i++)
            mem[i] = SHL32(st->mem_sp[i], 1);
        for (i = 0; i < NB_SUBFRAME_SIZE; i++)
            exc[i] = VERY_SMALL;
#ifdef SHORTCUTS2
        iir_mem16(exc, interp_qlpc, exc, response_bound, NB_ORDER, mem, stack);
      for (i=0;i<NB_ORDER;i++)
         mem[i]=SHL32(st->mem_sw[i],1);
      filter10(exc, st->bw_lpc1, st->bw_lpc2, exc, response_bound, mem, stack);
      SPEEX_MEMSET(&exc[response_bound], 0, NB_SUBFRAME_SIZE-response_bound);
#else
        iir_mem16(exc, interp_qlpc, exc, NB_SUBFRAME_SIZE, NB_ORDER, mem, stack);
        for (i = 0; i < NB_ORDER; i++)
            mem[i] = SHL32(st->mem_sw[i], 1);
        filter10(exc, bw_lpc1, bw_lpc2, exc, NB_SUBFRAME_SIZE, mem, stack);
#endif

        /* Compute weighted signal */
        for (i = 0; i < NB_ORDER; i++)
            mem[i] = st->mem_sw[i];
        filter10(sw, bw_lpc1, bw_lpc2, sw, NB_SUBFRAME_SIZE, mem, stack);

        if (st->complexity == 0)
            for (i = 0; i < NB_ORDER; i++)
                st->mem_sw[i] = mem[i];

        /* Compute target signal (saturation prevents overflows on clipped input speech) */
        for (i = 0; i < NB_SUBFRAME_SIZE; i++)
            target[i] = EXTRACT16(SATURATE(SUB32(sw[i], PSHR32(exc[i], 1)), 32767));

        for (i = 0; i < NB_SUBFRAME_SIZE; i++)
            exc[i] = inBuf[i];
        fir_mem16(exc, interp_qlpc, exc, NB_SUBFRAME_SIZE, NB_ORDER, st->mem_exc2, stack);
        /* If we have a long-term predictor (otherwise, something's wrong) */
                speex_assert (SUBMODE(ltp_quant));
        {
            int pit_min, pit_max;
            /* Long-term prediction */
            if (SUBMODE(lbr_pitch) != -1) {
                /* Low bit-rate pitch handling */
                int margin;
                margin = SUBMODE(lbr_pitch);
                if (margin) {
                    if (ol_pitch < NB_PITCH_START + margin - 1)
                        ol_pitch = NB_PITCH_START + margin - 1;
                    if (ol_pitch > NB_PITCH_END - margin)
                        ol_pitch = NB_PITCH_END - margin;
                    pit_min = ol_pitch - margin + 1;
                    pit_max = ol_pitch + margin;
                } else {
                    pit_min = pit_max = ol_pitch;
                }
            } else {
                pit_min = NB_PITCH_START;
                pit_max = NB_PITCH_END;
            }

            /* Force pitch to use only the current frame if needed */
            if (st->bounded_pitch && pit_max > offset)
                pit_max = offset;

            /* Perform pitch search */
            pitch = SUBMODE(ltp_quant)(target, sw, interp_qlpc, bw_lpc1, bw_lpc2,
                                       exc32, SUBMODE(ltp_params), pit_min, pit_max, ol_pitch_coef,
                                       NB_ORDER, NB_SUBFRAME_SIZE, bits, stack,
                                       exc, syn_resp, st->complexity, 0, st->plc_tuning, &st->cumul_gain);

            st->pitch[sub] = pitch;
        }
        /* Quantization of innovation */
        SPEEX_MEMSET(innov, 0, NB_SUBFRAME_SIZE);

        /* FIXME: Make sure this is safe from overflows (so far so good) */
        for (i = 0; i < NB_SUBFRAME_SIZE; i++)
            exc[i] = EXTRACT16(SUB32(EXTEND32(exc[i]), PSHR32(exc32[i], SIG_SHIFT - 1)));

        ener = SHL32(EXTEND32(compute_rms16(exc, NB_SUBFRAME_SIZE)), SIG_SHIFT);

        /*FIXME: Should use DIV32_16 and make sure result fits in 16 bits */
#ifdef FIXED_POINT
        {
         spx_word32_t f = PDIV32(ener,PSHR32(ol_gain,SIG_SHIFT));
         if (f<=32767)
            fine_gain = f;
         else
            fine_gain = 32767;
      }
#else
        fine_gain = PDIV32_16(ener, PSHR32(ol_gain, SIG_SHIFT));
#endif
        /* Calculate gain correction for the sub-frame (if any) */
        if (SUBMODE(have_subframe_gain)) {
            int qe;
            if (SUBMODE(have_subframe_gain) == 3) {
                qe = scal_quant(fine_gain, exc_gain_quant_scal3_bound, 8);
                speex_bits_pack(bits, qe, 3);
                ener = MULT16_32_Q14(exc_gain_quant_scal3[qe], ol_gain);
            } else {
                qe = scal_quant(fine_gain, exc_gain_quant_scal1_bound, 2);
                speex_bits_pack(bits, qe, 1);
                ener = MULT16_32_Q14(exc_gain_quant_scal1[qe], ol_gain);
            }
        } else {
            ener = ol_gain;
        }

        /*printf ("%f %f\n", ener, ol_gain);*/

        /* Normalize innovation */
        signal_div(target, target, ener, NB_SUBFRAME_SIZE);

        /* Quantize innovation */
                speex_assert (SUBMODE(innovation_quant));
        {
            /* Codebook search */
            SUBMODE(innovation_quant)(target, interp_qlpc, bw_lpc1, bw_lpc2,
                                      SUBMODE(innovation_params), NB_ORDER, NB_SUBFRAME_SIZE,
                                      innov, syn_resp, bits, stack, st->complexity, SUBMODE(double_codebook));

            /* De-normalize innovation and update excitation */
            signal_mul(innov, innov, ener, NB_SUBFRAME_SIZE);

            /* In some (rare) modes, we do a second search (more bits) to reduce noise even more */
            if (SUBMODE(double_codebook)) {
                char *tmp_stack = stack;
                VARDECL(spx_sig_t *innov2);
                ALLOC(innov2, NB_SUBFRAME_SIZE, spx_sig_t);
                SPEEX_MEMSET(innov2, 0, NB_SUBFRAME_SIZE);
                for (i = 0; i < NB_SUBFRAME_SIZE; i++)
                    target[i] = MULT16_16_P13(QCONST16(2.2f, 13), target[i]);
                SUBMODE(innovation_quant)(target, interp_qlpc, bw_lpc1, bw_lpc2,
                                          SUBMODE(innovation_params), NB_ORDER, NB_SUBFRAME_SIZE,
                                          innov2, syn_resp, bits, stack, st->complexity, 0);
                signal_mul(innov2, innov2, MULT16_32_Q15(QCONST16(0.454545f, 15), ener), NB_SUBFRAME_SIZE);
                for (i = 0; i < NB_SUBFRAME_SIZE; i++)
                    innov[i] = ADD32(innov[i], innov2[i]);
                stack = tmp_stack;
            }
            for (i = 0; i < NB_SUBFRAME_SIZE; i++)
                exc[i] = EXTRACT16(SATURATE32(PSHR32(ADD32(SHL32(exc32[i], 1), innov[i]), SIG_SHIFT), 32767));
            if (st->innov_rms_save)
                st->innov_rms_save[sub] = compute_rms(innov, NB_SUBFRAME_SIZE);
        }

        /* Final signal synthesis from excitation */
        iir_mem16(exc, interp_qlpc, sw, NB_SUBFRAME_SIZE, NB_ORDER, st->mem_sp, stack);

        /* Compute weighted signal again, from synthesized speech (not sure it's the right thing) */
        if (st->complexity != 0)
            filter10(sw, bw_lpc1, bw_lpc2, sw, NB_SUBFRAME_SIZE, st->mem_sw, stack);

    }

    /* Store the LSPs for interpolation in the next frame */
    if (st->submodeID >= 1) {
        for (i = 0; i < NB_ORDER; i++)
            st->old_lsp[i] = lsp[i];
        for (i = 0; i < NB_ORDER; i++)
            st->old_qlsp[i] = qlsp[i];
    }

#ifdef VORBIS_PSYCHO
    if (st->submodeID>=1)
      SPEEX_COPY(st->old_curve, st->curve, 128);
#endif

    if (st->submodeID == 1) {
#ifndef DISABLE_VBR
        if (st->dtx_count)
            speex_bits_pack(bits, 15, 4);
        else
#endif
            speex_bits_pack(bits, 0, 4);
    }

    /* The next frame will not be the first (Duh!) */
    st->first = 0;
    SPEEX_COPY(st->winBuf, in + 2 * NB_FRAME_SIZE - NB_WINDOW_SIZE, NB_WINDOW_SIZE - NB_FRAME_SIZE);

    if (SUBMODE(innovation_quant) == noise_codebook_quant || st->submodeID == 0)
        st->bounded_pitch = 1;
    else
        st->bounded_pitch = 0;

    return 1;
}
