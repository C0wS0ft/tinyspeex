#include "tinyspeexbits.h"

void speex_bits_reset(SpeexBits *bits) {
    /* We only need to clear the first byte now */
    bits->chars[0] = 0;
    bits->nbBits = 0;
    bits->charPtr = 0;
    bits->bitPtr = 0;
    bits->overflow = 0;
}

void speex_bits_init(SpeexBits *bits) {
    bits->chars = (char *) speex_alloc(MAX_CHARS_PER_FRAME);
    if (!bits->chars)
        return;

    bits->buf_size = MAX_CHARS_PER_FRAME;

    bits->owner = 1;

    speex_bits_reset(bits);
}


void speex_bits_destroy(SpeexBits *bits) {
    if (bits->owner)
        speex_free(bits->chars);
}

void speex_bits_pack(SpeexBits *bits, int data, int nbBits) {
    unsigned int d = data;

    if (bits->charPtr + ((nbBits + bits->bitPtr) >> LOG2_BITS_PER_CHAR) >= bits->buf_size) {
        //speex_notify("Buffer too small to pack bits");
        if (bits->owner) {
            int new_nchars = ((bits->buf_size + 5) * 3) >> 1;
            char *tmp = (char *) speex_realloc(bits->chars, new_nchars);
            if (tmp) {
                bits->buf_size = new_nchars;
                bits->chars = tmp;
            } else {
                //speex_warning("Could not resize input buffer: not packing");
                return;
            }
        } else {
            //speex_warning("Do not own input buffer: not packing");
            return;
        }
    }

    while (nbBits) {
        int bit;
        bit = (d >> (nbBits - 1)) & 1;
        bits->chars[bits->charPtr] |= bit << (BITS_PER_CHAR - 1 - bits->bitPtr);
        bits->bitPtr++;

        if (bits->bitPtr == BITS_PER_CHAR) {
            bits->bitPtr = 0;
            bits->charPtr++;
            bits->chars[bits->charPtr] = 0;
        }
        bits->nbBits++;
        nbBits--;
    }
}

void speex_bits_insert_terminator(SpeexBits *bits) {
    if (bits->bitPtr)
        speex_bits_pack(bits, 0, 1);
    while (bits->bitPtr)
        speex_bits_pack(bits, 1, 1);
}

int speex_bits_write(SpeexBits *bits, char *chars, int max_nbytes) {
    int i;
    int max_nchars = max_nbytes;
    int charPtr, bitPtr, nbBits;

    /* Insert terminator, but save the data so we can put it back after */
    bitPtr = bits->bitPtr;
    charPtr = bits->charPtr;
    nbBits = bits->nbBits;
    speex_bits_insert_terminator(bits);
    bits->bitPtr = bitPtr;
    bits->charPtr = charPtr;
    bits->nbBits = nbBits;

    if (max_nchars > ((bits->nbBits + BITS_PER_CHAR - 1) >> LOG2_BITS_PER_CHAR))
        max_nchars = ((bits->nbBits + BITS_PER_CHAR - 1) >> LOG2_BITS_PER_CHAR);

    for (i = 0; i < max_nchars; i++)
        chars[i] = HTOLS(bits->chars[i]);

    return max_nchars * BYTES_PER_CHAR;
}

unsigned int speex_bits_unpack_unsigned(SpeexBits *bits, int nbBits)
{
    unsigned int d=0;
    if ((bits->charPtr<<LOG2_BITS_PER_CHAR)+bits->bitPtr+nbBits>bits->nbBits)
        bits->overflow=1;
    if (bits->overflow)
        return 0;
    while(nbBits)
    {
        d<<=1;
        d |= (bits->chars[bits->charPtr]>>(BITS_PER_CHAR-1 - bits->bitPtr))&1;
        bits->bitPtr++;
        if (bits->bitPtr==BITS_PER_CHAR)
        {
            bits->bitPtr=0;
            bits->charPtr++;
        }
        nbBits--;
    }
    return d;
}

