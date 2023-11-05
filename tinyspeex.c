#include <stdio.h>
#include "tinyspeex.h"
#include "tinyspeexencoder.h"
#include "submodes.c"

/* Default mode for narrowband */
static const SpeexNBMode nb_mode = {
        NB_FRAME_SIZE,    /*frameSize*/
        NB_SUBFRAME_SIZE, /*subframeSize*/
        NB_ORDER,         /*lpcSize*/
        NB_PITCH_START,               /*pitchStart*/
        NB_PITCH_END,              /*pitchEnd*/
        QCONST16(0.92, 15),  /* gamma1 */
        QCONST16(0.6, 15),   /* gamma2 */
        QCONST16(.0002, 15), /*lpc_floor*/
        //{NULL, &nb_submode1, &nb_submode2, &nb_submode3, &nb_submode4, &nb_submode5, &nb_submode6, &nb_submode7, &nb_submode8, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
        {NULL, &nb_submode3, &nb_submode3, &nb_submode3, &nb_submode3, &nb_submode3, &nb_submode3, &nb_submode3, &nb_submode3, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
        5, // TODO CHECK was 5
        {1, 8, 2, 3, 3, 4, 4, 5, 5, 6, 7}
};

const SpeexMode speex_nb_mode = {
        &nb_mode,
        "narrowband",
        0,
        4,
        nb_encoder_init,
        nb_encoder_destroy,
        nb_encode,
        nb_encoder_ctl,
};

const struct SpeexMode *speex_lib_get_mode(int) {
    return &speex_nb_mode;
}

void *speex_encoder_init(const SpeexMode *mode) {
    printf("%s\n", __FUNCTION__);
    return mode->enc_init(mode);
}

int speex_encoder_ctl(void *state, int request, void *ptr) {
    printf("%s\n", __FUNCTION__);
    return (*((SpeexMode **) state))->enc_ctl(state, request, ptr);
}

void speex_encoder_destroy(void *state) {
    printf("%s\n", __FUNCTION__);
    (*((SpeexMode **) state))->enc_destroy(state);
}

int speex_encode(void *state, float *in, SpeexBits *bits) {
    printf("%s\n", __FUNCTION__);
    return (*((SpeexMode **) state))->enc(state, in, bits);
}

int speex_encode_int(void *state, int16_t *in, SpeexBits *bits) {
    int i;
    spx_int32_t N;
    float float_in[MAX_IN_SAMPLES];
    speex_encoder_ctl(state, SPEEX_GET_FRAME_SIZE, &N);

    printf("FRAME SIZE [%d]\n", N);

    for (i = 0; i < N; i++)
        float_in[i] = in[i];

    return (*((SpeexMode **) state))->enc(state, float_in, bits);
}

void speex_init_header(SpeexHeader *header, int rate, int nb_channels, const SpeexMode *m) {
    int i;
    const char *h = "Speex   ";

    printf("%s\n", __FUNCTION__);
    /*
    strncpy(header->speex_string, "Speex   ", 8);
    strncpy(header->speex_version, SPEEX_VERSION, SPEEX_HEADER_VERSION_LENGTH-1);
    header->speex_version[SPEEX_HEADER_VERSION_LENGTH-1]=0;
    */
    for (i = 0; i < 8; i++)
        header->speex_string[i] = h[i];

    for (i = 0; i < SPEEX_HEADER_VERSION_LENGTH - 1 && SPEEX_VERSION[i]; i++)
        header->speex_version[i] = SPEEX_VERSION[i];

    for (; i < SPEEX_HEADER_VERSION_LENGTH; i++)
        header->speex_version[i] = 0;

    header->speex_version_id = 1;
    header->header_size = sizeof(SpeexHeader);

    header->rate = rate;
    header->mode = m->modeID;
    header->mode_bitstream_version = m->bitstream_version;

    //if (m->modeID<0)
    //  speex_warning("This mode is meant to be used alone");

    header->nb_channels = nb_channels;
    header->bitrate = -1;
    header->frame_size = NB_FRAME_SIZE;
    //speex_mode_query(m, SPEEX_MODE_FRAME_SIZE, &header->frame_size);

    header->vbr = 0;

    header->frames_per_packet = 0;
    header->extra_headers = 0;
    header->reserved1 = 0;
    header->reserved2 = 0;
}

char *speex_header_to_packet(SpeexHeader *header, int *size) {
    SpeexHeader *le_header;
    le_header = (SpeexHeader *) speex_alloc(sizeof(SpeexHeader));

    SPEEX_COPY(le_header, header, 1);

    *size = sizeof(SpeexHeader);

    return (char *) le_header;
}

int speex_get_frame_size() {
    return NB_FRAME_SIZE;
}

static const float e_ratio_quant_bounds[3] = {0.2825f, 0.356f, 0.4485f};

void speex_encode_stereo(float *data, int frame_size, SpeexBits *bits) {
    int i, tmp;
    float e_left = 0, e_right = 0, e_tot = 0;
    float balance, e_ratio;
    for (i = 0; i < frame_size; i++) {
        e_left += ((float) data[2 * i]) * data[2 * i];
        e_right += ((float) data[2 * i + 1]) * data[2 * i + 1];
        data[i] = .5 * (((float) data[2 * i]) + data[2 * i + 1]);
        e_tot += ((float) data[i]) * data[i];
    }
    balance = (e_left + 1) / (e_right + 1);
    e_ratio = e_tot / (1 + e_left + e_right);

    /*Quantization*/
    speex_bits_pack(bits, 14, 5);
    speex_bits_pack(bits, SPEEX_INBAND_STEREO, 4);

    balance = 4 * log(balance);

    /*Pack sign*/
    if (balance > 0)
        speex_bits_pack(bits, 0, 1);
    else
        speex_bits_pack(bits, 1, 1);
    balance = floor(.5 + fabs(balance));
    if (balance > 30)
        balance = 31;

    speex_bits_pack(bits, (int) balance, 5);

    /* FIXME: this is a hack */
    tmp = scal_quant(e_ratio * Q15_ONE, e_ratio_quant_bounds, 4);
    speex_bits_pack(bits, tmp, 2);
}
