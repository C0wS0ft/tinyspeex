#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define MAX_CHARS_PER_FRAME 2000
#define BYTES_PER_CHAR 1
#define BITS_PER_CHAR 8
#define LOG2_BITS_PER_CHAR 3
#define HTOLS(A) (A)

static inline void *speex_alloc(int size) {
    return calloc(size, 1);
}

static inline void speex_free(void *ptr) {
    free(ptr);
}

static inline void *speex_realloc(void *ptr, int size) {
    return realloc(ptr, size);
}

typedef struct SpeexBits {
    char *chars;   /**< "raw" data */
    int nbBits;  /**< Total number of bits stored in the stream*/
    int charPtr; /**< Position of the byte "cursor" */
    int bitPtr;  /**< Position of the bit "cursor" within the current char */
    int owner;   /**< Does the struct "own" the "raw" buffer (member "chars") */
    int overflow;/**< Set to one if we try to read past the valid data */
    int buf_size;/**< Allocated size for buffer */
    int reserved1; /**< Reserved for future use */
    void *reserved2; /**< Reserved for future use */
} SpeexBits;

void speex_bits_pack(SpeexBits *bits, int data, int nbBits);
unsigned int speex_bits_unpack_unsigned(SpeexBits *bits, int nbBits);

typedef void *(*encoder_init_func)(const struct SpeexMode *mode);

typedef void (*encoder_destroy_func)(void *st);

typedef int (*encode_func)(void *state, void *in, SpeexBits *bits);

typedef int (*encoder_ctl_func)(void *state, int request, void *ptr);

typedef struct SpeexMode {
    /** Pointer to the low-level mode data */
    const void *mode;

    /** The name of the mode (you should not rely on this to identify the mode)*/
    const char *modeName;

    /**ID of the mode*/
    int modeID;

    /**Version number of the bitstream (incremented every time we break
     bitstream compatibility*/
    int bitstream_version;

    /** Pointer to encoder initialization function */
    encoder_init_func enc_init;

    /** Pointer to encoder destruction function */
    encoder_destroy_func enc_destroy;

    /** Pointer to frame encoding function */
    encode_func enc;

    /** ioctl-like requests for encoder */
    encoder_ctl_func enc_ctl;
} SpeexMode;

void speex_bits_init(SpeexBits *bits);
void speex_bits_reset(SpeexBits *bits);
void speex_bits_destroy(SpeexBits *bits);
void speex_bits_pack(SpeexBits *bits, int data, int nbBits);
void speex_bits_insert_terminator(SpeexBits *bits);
int speex_bits_write(SpeexBits *bits, char *chars, int max_nbytes);
unsigned int speex_bits_unpack_unsigned(SpeexBits *bits, int nbBits);