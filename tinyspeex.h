#pragma once

#include <stdlib.h>
#include "tinyspeexbits.h"
#include "tinyspeexencoder.h"
#include "modes.h"

#define MAX_IN_SAMPLES 640

#define SPEEX_INBAND_STEREO              9

#define M_PI        3.14159265358979323846
#define MAX32(a, b) ((a) > (b) ? (a) : (b))

#define LSP_LINEAR(i) (.25*(i)+.25)
#define LSP_LINEAR_HIGH(i) (.3125*(i)+.75)
#define LSP_SCALE 256.
#define LSP_DIV_256(x) (0.0039062*(x))
#define LSP_DIV_512(x) (0.0019531*(x))
#define LSP_DIV_1024(x) (0.00097656*(x))
#define LSP_PI M_PI

#define NB_CDBK_SIZE 64
#define NB_CDBK_SIZE_LOW1 64
#define NB_CDBK_SIZE_LOW2 64
#define NB_CDBK_SIZE_HIGH1 64
#define NB_CDBK_SIZE_HIGH2 64

#define filter10(x, num, den, y, N, mem, stack) filter_mem16(x, num, den, y, N, 10, mem, stack)
#define gain_3tap_to_1tap(g) (ABS(g[1]) + (g[0]>0 ? g[0] : -.5*g[0]) + (g[2]>0 ? g[2] : -.5*g[2]))

typedef struct {
    const signed char *gain_cdbk;
    int gain_bits;
    int pitch_bits;
} ltp_params;

typedef struct split_cb_params {
    int subvect_size;
    int nb_subvect;
    const signed char *shape_cb;
    int shape_bits;
    int have_sign;
} split_cb_params;

#define SPEEX_VERSION "1.2.1"
#define SPEEX_HEADER_STRING_LENGTH 8
#define SPEEX_HEADER_VERSION_LENGTH 20

/** Speex header info for file-based formats */
typedef struct SpeexHeader {
    char speex_string[SPEEX_HEADER_STRING_LENGTH];   /**< Identifies a Speex bit-stream, always set to "Speex   " */
    char speex_version[SPEEX_HEADER_VERSION_LENGTH]; /**< Speex version */
    spx_int32_t speex_version_id;       /**< Version for Speex (for checking compatibility) */
    spx_int32_t header_size;            /**< Total size of the header ( sizeof(SpeexHeader) ) */
    spx_int32_t rate;                   /**< Sampling rate used */
    spx_int32_t mode;                   /**< Mode used (0 for narrowband, 1 for wideband) */
    spx_int32_t mode_bitstream_version; /**< Version ID of the bit-stream */
    spx_int32_t nb_channels;            /**< Number of channels encoded */
    spx_int32_t bitrate;                /**< Bit-rate used */
    spx_int32_t frame_size;             /**< Size of frames */
    spx_int32_t vbr;                    /**< 1 for a VBR encoding, 0 otherwise */
    spx_int32_t frames_per_packet;      /**< Number of frames stored per Ogg packet */
    spx_int32_t extra_headers;          /**< Number of additional headers after the comments */
    spx_int32_t reserved1;              /**< Reserved for future use, must be zero */
    spx_int32_t reserved2;              /**< Reserved for future use, must be zero */
} SpeexHeader;

const struct SpeexMode *speex_lib_get_mode(int);
void speex_init_header(SpeexHeader *header, int rate, int nb_channels, const SpeexMode *m);
char *speex_header_to_packet(SpeexHeader *header, int *size);
void speex_encode_stereo(float *data, int frame_size, SpeexBits *bits);
