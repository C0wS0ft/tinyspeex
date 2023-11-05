#pragma once

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "tinyspeexbits.h"
#include "modes.h"

typedef float spx_mem_t;
typedef float spx_coef_t;
typedef float spx_lsp_t;
typedef float spx_sig_t;
typedef float spx_word16_t;
typedef float spx_word32_t;
typedef uint32_t spx_uint32_t;
typedef int32_t spx_int32_t;
typedef int16_t spx_int16_t;

#define HIGHPASS_NARROWBAND 0
#define HIGHPASS_WIDEBAND 2
#define HIGHPASS_INPUT 0
#define HIGHPASS_OUTPUT 1

#if __GNUC__ <= 3
/* GCC-3.4 and older did not use hardware loops and thus did not have
 * register constraints for declaring clobbers.
 */
# define BFIN_HWLOOP0_REGS
# define BFIN_HWLOOP1_REGS
#else
# define BFIN_HWLOOP0_REGS , "LB0", "LT0", "LC0"
# define BFIN_HWLOOP1_REGS , "LB1", "LT1", "LC1"
#endif

#define ABS(x) ((x) < 0 ? (-(x)) : (x))      /**< Absolute integer value. */

static const float exc_gain_quant_scal3_bound[7] = {0.112338f, 0.236980f, 0.369316f, 0.492054f, 0.637471f, 0.828874f, 1.132784f};
static const float exc_gain_quant_scal3[8] = {0.061130f, 0.163546f, 0.310413f, 0.428220f, 0.555887f, 0.719055f, 0.938694f, 1.326874f};
static const float exc_gain_quant_scal1_bound[1] = {0.87798f};
static const float exc_gain_quant_scal1[2] = {0.70469f, 1.05127f};

#define LSP_MARGIN .002f
#define LSP_DELTA1 .2f
#define LSP_DELTA2 .05f

#define MIN_ENERGY 6000
#define NOISE_POW .3

#define NB_SUBMODES 16
#define NB_SUBMODE_BITS 4

#define NB_ORDER 10
#define NB_FRAME_SIZE 160
#define NB_SUBFRAME_SIZE 40
#define NB_NB_SUBFRAMES 4
#define NB_PITCH_START 17
#define NB_PITCH_END 144
#define NB_ENC_STACK (8000*sizeof(spx_sig_t))

#define NB_WINDOW_SIZE (NB_FRAME_SIZE+NB_SUBFRAME_SIZE)
#define NB_EXCBUF (NB_FRAME_SIZE+NB_PITCH_END+2)
#define NB_DEC_BUFFER (NB_FRAME_SIZE+2*NB_PITCH_END+NB_SUBFRAME_SIZE+12)

#define filter10(x, num, den, y, N, mem, stack) filter_mem16(x, num, den, y, N, 10, mem, stack)
#define spx_cos cos
#define spx_sqrt sqrt
#define speex_assert assert
#define sqr(x) ((x)*(x))

#define VBR_MEMORY_SIZE 5

#define SPEEX_COPY(dst, src, n) (memcpy((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#define SPEEX_MEMSET(dst, c, n) (memset((dst), (c), (n)*sizeof(*(dst))))
#define SPEEX_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#define SPEEX_MEMSET(dst, c, n) (memset((dst), (c), (n)*sizeof(*(dst))))

typedef float spx_mem_t;
typedef float spx_coef_t;
typedef float spx_lsp_t;
typedef float spx_sig_t;
typedef float spx_word16_t;
typedef float spx_word32_t;

#define Q15ONE 1.0f
#define LPC_SCALING  1.f
#define SIG_SCALING  1.f
#define LSP_SCALING  1.f
#define GAMMA_SCALING 1.f
#define GAIN_SCALING 1.f
#define GAIN_SCALING_1 1.f

#define VERY_SMALL 1e-15f
#define VERY_LARGE32 1e15f
#define VERY_LARGE16 1e15f
#define Q15_ONE ((spx_word16_t)1.f)

#define QCONST16(x, bits) (x)
#define QCONST32(x, bits) (x)

#define NEG16(x) (-(x))
#define NEG32(x) (-(x))
#define EXTRACT16(x) (x)
#define EXTEND32(x) (x)
#define SHR16(a, shift) (a)
#define SHL16(a, shift) (a)
#define SHR32(a, shift) (a)
#define SHL32(a, shift) (a)
#define PSHR16(a, shift) (a)
#define PSHR32(a, shift) (a)
#define VSHR32(a, shift) (a)
#define SATURATE16(x, a) (x)
#define SATURATE32(x, a) (x)

#define PSHR(a, shift)       (a)
#define SHR(a, shift)       (a)
#define SHL(a, shift)       (a)
#define SATURATE(x, a) (x)

#define ADD16(a, b) ((a)+(b))
#define SUB16(a, b) ((a)-(b))
#define ADD32(a, b) ((a)+(b))
#define SUB32(a, b) ((a)-(b))
#define MULT16_16_16(a, b)     ((a)*(b))
#define MULT16_16(a, b)     ((spx_word32_t)(a)*(spx_word32_t)(b))
#define MAC16_16(c, a, b)     ((c)+(spx_word32_t)(a)*(spx_word32_t)(b))

#define MULT16_32_Q11(a, b)     ((a)*(b))
#define MULT16_32_Q13(a, b)     ((a)*(b))
#define MULT16_32_Q14(a, b)     ((a)*(b))
#define MULT16_32_Q15(a, b)     ((a)*(b))
#define MULT16_32_P15(a, b)     ((a)*(b))

#define MAC16_32_Q11(c, a, b)     ((c)+(a)*(b))
#define MAC16_32_Q15(c, a, b)     ((c)+(a)*(b))

#define MAC16_16_Q11(c, a, b)     ((c)+(a)*(b))
#define MAC16_16_Q13(c, a, b)     ((c)+(a)*(b))
#define MAC16_16_P13(c, a, b)     ((c)+(a)*(b))
#define MULT16_16_Q11_32(a, b)     ((a)*(b))
#define MULT16_16_Q13(a, b)     ((a)*(b))
#define MULT16_16_Q14(a, b)     ((a)*(b))
#define MULT16_16_Q15(a, b)     ((a)*(b))
#define MULT16_16_P15(a, b)     ((a)*(b))
#define MULT16_16_P13(a, b)     ((a)*(b))
#define MULT16_16_P14(a, b)     ((a)*(b))

#define DIV32_16(a, b)     (((spx_word32_t)(a))/(spx_word16_t)(b))
#define PDIV32_16(a, b)     (((spx_word32_t)(a))/(spx_word16_t)(b))
#define DIV32(a, b)     (((spx_word32_t)(a))/(spx_word32_t)(b))
#define PDIV32(a, b)     (((spx_word32_t)(a))/(spx_word32_t)(b))

#define FREQ_SCALE 1.
#define ANGLE2X(a) (spx_cos(a))
#define X2ANGLE(x) (acos(x))

#ifdef FIXED_POINT
#define SIGN_CHANGE(a,b) ((((a)^(b))&0x80000000)||(b==0))
#else
#define SIGN_CHANGE(a, b) (((a)*(b))<0.0)
#endif

#define SUBMODE(x) st->submodes[st->submodeID]->x
#define VARDECL(var)
#define ALLOC(var, size, type) type var[size]

static inline void *speex_alloc_scratch(int size) {
    return calloc(size, 1);
}

static inline void speex_free_scratch(void *ptr) {
    free(ptr);
}

/** Quantizes LSPs */
typedef void (*lsp_quant_func)(spx_lsp_t *, spx_lsp_t *, int, SpeexBits *);

/** Decodes quantized LSPs */
typedef void (*lsp_unquant_func)(spx_lsp_t *, int, SpeexBits *);


/** Long-term predictor quantization */
typedef int (*ltp_quant_func)(spx_word16_t *, spx_word16_t *, spx_coef_t *, spx_coef_t *,
                              spx_coef_t *, spx_sig_t *, const void *, int, int, spx_word16_t,
                              int, int, SpeexBits *, char *, spx_word16_t *, spx_word16_t *, int, int, int, spx_word32_t *);

/** Long-term un-quantize */
typedef void (*ltp_unquant_func)(spx_word16_t *, spx_word32_t *, int, int, spx_word16_t, const void *, int, int *,
                                 spx_word16_t *, SpeexBits *, char *, int, int, spx_word16_t, int);


/** Innovation quantization function */
typedef void (*innovation_quant_func)(spx_word16_t *, spx_coef_t *, spx_coef_t *, spx_coef_t *, const void *, int, int,
                                      spx_sig_t *, spx_word16_t *, SpeexBits *, char *, int, int);

/** Innovation unquantization function */
typedef void (*innovation_unquant_func)(spx_sig_t *, const void *, int, SpeexBits *, char *, spx_uint32_t *);

typedef struct SpeexSubmode {
    int lbr_pitch;          /**< Set to -1 for "normal" modes, otherwise encode pitch using a global pitch and allowing a +- lbr_pitch variation (for low not-rates)*/
    int forced_pitch_gain;  /**< Use the same (forced) pitch gain for all sub-frames */
    int have_subframe_gain; /**< Number of bits to use as sub-frame innovation gain */
    int double_codebook;    /**< Apply innovation quantization twice for higher quality (and higher bit-rate)*/
    /*LSP functions*/
    lsp_quant_func lsp_quant; /**< LSP quantization function */
    lsp_unquant_func lsp_unquant; /**< LSP unquantization function */

    /*Long-term predictor functions*/
    ltp_quant_func ltp_quant; /**< Long-term predictor (pitch) quantizer */
    ltp_unquant_func ltp_unquant; /**< Long-term predictor (pitch) un-quantizer */
    const void *ltp_params; /**< Pitch parameters (options) */

    /*Quantization of innovation*/
    innovation_quant_func innovation_quant; /**< Innovation quantization */
    innovation_unquant_func innovation_unquant; /**< Innovation un-quantization */
    const void *innovation_params; /**< Innovation quantization parameters*/

    spx_word16_t comb_gain;  /**< Gain of enhancer comb filter */

    int bits_per_frame; /**< Number of bits per frame after encoding*/
} SpeexSubmode;

typedef struct SpeexNBMode {
    int frameSize;      /**< Size of frames used for encoding */
    int subframeSize;   /**< Size of sub-frames used for encoding */
    int lpcSize;        /**< Order of LPC filter */
    int pitchStart;     /**< Smallest pitch value allowed */
    int pitchEnd;       /**< Largest pitch value allowed */

    spx_word16_t gamma1;    /**< Perceptual filter parameter #1 */
    spx_word16_t gamma2;    /**< Perceptual filter parameter #2 */
    spx_word16_t lpc_floor;      /**< Noise floor for LPC analysis */

    const SpeexSubmode *submodes[NB_SUBMODES]; /**< Sub-mode data for the mode */
    int defaultSubmode; /**< Default sub-mode to use when encoding */
    int quality_map[11]; /**< Mode corresponding to each quality setting */
} SpeexNBMode;


typedef struct VBRState {
    float average_energy;
    float last_energy;
    float last_log_energy[VBR_MEMORY_SIZE];
    float accum_sum;
    float last_pitch_coef;
    float soft_pitch;
    float last_quality;
    float noise_level;
    float noise_accum;
    float noise_accum_count;
    int consec_noise;
} VBRState;

/**Structure representing the full state of the narrowband encoder*/
typedef struct EncState {
    const SpeexMode *mode;        /**< Mode corresponding to the state */
    int first;                 /**< Is this the first frame? */

    spx_word32_t cumul_gain;      /**< Product of previously used pitch gains (Q10) */
    int bounded_pitch;         /**< Next frame should not rely on previous frames for pitch */
    int ol_pitch;              /**< Open-loop pitch */
    int ol_voiced;             /**< Open-loop voiced/non-voiced decision */
    int pitch[NB_NB_SUBFRAMES];

#ifdef VORBIS_PSYCHO
    VorbisPsy *psy;
   float *psy_window;
   float *curve;
   float *old_curve;
#endif

    spx_word16_t gamma1;         /**< Perceptual filter: A(z/gamma1) */
    spx_word16_t gamma2;         /**< Perceptual filter: A(z/gamma2) */
    spx_word16_t lpc_floor;      /**< Noise floor multiplier for A[0] in LPC analysis*/
    char *stack;                 /**< Pseudo-stack allocation for temporary memory */
    spx_word16_t winBuf[NB_WINDOW_SIZE - NB_FRAME_SIZE];         /**< Input buffer (original signal) */
    spx_word16_t excBuf[NB_EXCBUF];         /**< Excitation buffer */
    spx_word16_t *exc;            /**< Start of excitation frame */
    spx_word16_t swBuf[NB_EXCBUF];          /**< Weighted signal buffer */
    spx_word16_t *sw;             /**< Start of weighted signal frame */
    const spx_word16_t *window;   /**< Temporary (Hanning) window */
    const spx_word16_t *lagWindow;      /**< Window applied to auto-correlation */
    spx_lsp_t old_lsp[NB_ORDER];           /**< LSPs for previous frame */
    spx_lsp_t old_qlsp[NB_ORDER];          /**< Quantized LSPs for previous frame */
    spx_mem_t mem_sp[NB_ORDER];            /**< Filter memory for signal synthesis */
    spx_mem_t mem_sw[NB_ORDER];            /**< Filter memory for perceptually-weighted signal */
    spx_mem_t mem_sw_whole[NB_ORDER];      /**< Filter memory for perceptually-weighted signal (whole frame)*/
    spx_mem_t mem_exc[NB_ORDER];           /**< Filter memory for excitation (whole frame) */
    spx_mem_t mem_exc2[NB_ORDER];          /**< Filter memory for excitation (whole frame) */
    spx_mem_t mem_hp[2];          /**< High-pass filter memory */
    spx_word32_t pi_gain[NB_NB_SUBFRAMES];        /**< Gain of LPC filter at theta=pi (fe/2) */
    spx_word16_t *innov_rms_save; /**< If non-NULL, innovation RMS is copied here */

#ifndef DISABLE_VBR
    VBRState vbr;                /**< State of the VBR data */
    float vbr_quality;           /**< Quality setting for VBR encoding */
    float relative_quality;      /**< Relative quality that will be needed by VBR */
    spx_int32_t vbr_enabled;      /**< 1 for enabling VBR, 0 otherwise */
    spx_int32_t vbr_max;          /**< Max bit-rate allowed in VBR mode */
    int vad_enabled;           /**< 1 for enabling VAD, 0 otherwise */
    int dtx_enabled;           /**< 1 for enabling DTX, 0 otherwise */
    int dtx_count;             /**< Number of consecutive DTX frames */
    spx_int32_t abr_enabled;      /**< ABR setting (in bps), 0 if off */
    float abr_drift;
    float abr_drift2;
    float abr_count;
#endif /* #ifndef DISABLE_VBR */

    int complexity;            /**< Complexity setting (0-10 from least complex to most complex) */
    spx_int32_t sampling_rate;
    int plc_tuning;
    int encode_submode;
    const SpeexSubmode *const *submodes; /**< Sub-mode data */
    int submodeID;             /**< Activated sub-mode */
    int submodeSelect;         /**< Mode chosen by the user (may differ from submodeID if VAD is on) */
    int isWideband;            /**< Is this used as part of the embedded wideband codec */
    int highpass_enabled;        /**< Is the input filter enabled */
} EncState;

void *nb_encoder_init(const SpeexMode *m);

void nb_encoder_destroy(void *state);

int nb_encoder_ctl(void *state, int request, void *ptr);

int nb_encode(void *state, void *vin, SpeexBits *bits);

void iir_mem16(const spx_word16_t *x, const spx_coef_t *den, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack);

float inner_prod(const float *a, const float *b, int len);

static void pitch_xcorr(const float *_x, const float *_y, float *corr, int len, int nb_pitch, char *stack);

void open_loop_nbest_pitch(spx_word16_t *sw, int start, int end, int len, int *pitch, spx_word16_t *gain, int N, char *stack);

void filter_mem16(const spx_word16_t *x, const spx_coef_t *num, const spx_coef_t *den, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack);

int scal_quant(spx_word16_t in, const spx_word16_t *boundary, int entries);

void fir_mem16(const spx_word16_t *x, const spx_coef_t *num, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack);

void
residue_percep_zero16(const spx_word16_t *xx, const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord,
                      char *stack);

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
);