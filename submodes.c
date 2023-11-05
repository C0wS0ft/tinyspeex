#include "tinyspeexencoder.h"
#include "tinyspeex.h"

//#include "gain_table.c"
#include "gain_table_lbr.c"
#include "lsp_tables_nb.c"

//#include "exc_10_16_table.c"
#include "exc_10_32_table.c"
//#include "exc_20_32_table.c"
//#include "exc_5_256_table.c"
//#include "exc_5_64_table.c"
//#include "exc_8_128_table.c"

static void compute_quant_weights(spx_lsp_t *qlsp, spx_word16_t *quant_weight, int order)
{
    int i;
    spx_word16_t tmp1, tmp2;
    for (i=0;i<order;i++)
    {
        if (i==0)
            tmp1 = qlsp[i];
        else
            tmp1 = qlsp[i]-qlsp[i-1];
        if (i==order-1)
            tmp2 = LSP_PI-qlsp[i];
        else
            tmp2 = qlsp[i+1]-qlsp[i];
        if (tmp2<tmp1)
            tmp1 = tmp2;
#ifdef FIXED_POINT
        quant_weight[i] = DIV32_16(81920,ADD16(300,tmp1));
#else
        quant_weight[i] = 10/(.04+tmp1);
#endif
    }
}

static int lsp_quant(spx_word16_t *x, const signed char *cdbk, int nbVec, int nbDim)
{
    int i,j;
    spx_word32_t dist;
    spx_word16_t tmp;
    spx_word32_t best_dist=VERY_LARGE32;
    int best_id=0;
    const signed char *ptr=cdbk;
    for (i=0;i<nbVec;i++)
    {
        dist=0;
        for (j=0;j<nbDim;j++)
        {
            tmp=SUB16(x[j],SHL16((spx_word16_t)*ptr++,5));
            dist=MAC16_16(dist,tmp,tmp);
        }
        if (dist<best_dist)
        {
            best_dist=dist;
            best_id=i;
        }
    }

    for (j=0;j<nbDim;j++)
        x[j] = SUB16(x[j],SHL16((spx_word16_t)cdbk[best_id*nbDim+j],5));

    return best_id;
}

static int lsp_weight_quant(spx_word16_t *x, spx_word16_t *weight, const signed char *cdbk, int nbVec, int nbDim)
{
    int i,j;
    spx_word32_t dist;
    spx_word16_t tmp;
    spx_word32_t best_dist=VERY_LARGE32;
    int best_id=0;
    const signed char *ptr=cdbk;
    for (i=0;i<nbVec;i++)
    {
        dist=0;
        for (j=0;j<nbDim;j++)
        {
            tmp=SUB16(x[j],SHL16((spx_word16_t)*ptr++,5));
            dist=MAC16_32_Q15(dist,weight[j],MULT16_16(tmp,tmp));
        }
        if (dist<best_dist)
        {
            best_dist=dist;
            best_id=i;
        }
    }

    for (j=0;j<nbDim;j++)
        x[j] = SUB16(x[j],SHL16((spx_word16_t)cdbk[best_id*nbDim+j],5));
    return best_id;
}

void lsp_quant_lbr(spx_lsp_t *lsp, spx_lsp_t *qlsp, int order, SpeexBits *bits)
{
    int i;
    int id;
    spx_word16_t quant_weight[10];

    for (i=0;i<order;i++)
        qlsp[i]=lsp[i];

    compute_quant_weights(qlsp, quant_weight, order);

    for (i=0;i<order;i++)
        qlsp[i]=SUB16(qlsp[i],LSP_LINEAR(i));
#ifndef FIXED_POINT
    for (i=0;i<order;i++)
        qlsp[i]=qlsp[i]*LSP_SCALE;
#endif
    id = lsp_quant(qlsp, cdbk_nb, NB_CDBK_SIZE, order);
    speex_bits_pack(bits, id, 6);

    for (i=0;i<order;i++)
        qlsp[i]*=2;

    id = lsp_weight_quant(qlsp, quant_weight, cdbk_nb_low1, NB_CDBK_SIZE_LOW1, 5);
    speex_bits_pack(bits, id, 6);

    id = lsp_weight_quant(qlsp+5, quant_weight+5, cdbk_nb_high1, NB_CDBK_SIZE_HIGH1, 5);
    speex_bits_pack(bits, id, 6);

#ifdef FIXED_POINT
    for (i=0;i<order;i++)
      qlsp[i] = PSHR16(qlsp[i],1);
#else
    for (i=0;i<order;i++)
        qlsp[i] = qlsp[i]*0.0019531;
#endif

    for (i=0;i<order;i++)
        qlsp[i]=lsp[i]-qlsp[i];
}

void lsp_unquant_lbr(spx_lsp_t *lsp, int order, SpeexBits *bits)
{
    int i, id;
    for (i=0;i<order;i++)
        lsp[i]=LSP_LINEAR(i);


    id=speex_bits_unpack_unsigned(bits, 6);
    for (i=0;i<10;i++)
        lsp[i] += LSP_DIV_256(cdbk_nb[id*10+i]);

    id=speex_bits_unpack_unsigned(bits, 6);
    for (i=0;i<5;i++)
        lsp[i] += LSP_DIV_512(cdbk_nb_low1[id*5+i]);

    id=speex_bits_unpack_unsigned(bits, 6);
    for (i=0;i<5;i++)
        lsp[i+5] += LSP_DIV_512(cdbk_nb_high1[id*5+i]);

}

void syn_percep_zero16(const spx_word16_t *xx, const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord, char *stack)
{
    int i;
    VARDECL(spx_mem_t *mem);
    ALLOC(mem, ord, spx_mem_t);
    for (i=0;i<ord;i++)
        mem[i]=0;
    iir_mem16(xx, ak, y, N, ord, mem, stack);
    for (i=0;i<ord;i++)
        mem[i]=0;
    filter_mem16(y, awk1, awk2, y, N, ord, mem, stack);
}

static inline spx_word16_t speex_rand(spx_word16_t std, spx_uint32_t *seed)
{
    const unsigned int jflone = 0x3f800000;
    const unsigned int jflmsk = 0x007fffff;
    union {int i; float f;} ran;
    *seed = 1664525 * *seed + 1013904223;
    ran.i = jflone | (jflmsk & *seed);
    ran.f -= 1.5;
    return 3.4642*std*ran.f;
}

static void compute_weighted_codebook(const signed char *shape_cb, const spx_word16_t *r, spx_word16_t *resp, spx_word16_t *resp2, spx_word32_t *E, int shape_cb_size, int subvect_size, char *stack)
{
    int i, j, k;
    VARDECL(spx_word16_t *shape);
    ALLOC(shape, subvect_size, spx_word16_t);
    for (i=0;i<shape_cb_size;i++)
    {
        spx_word16_t *res;

        res = resp+i*subvect_size;
        for (k=0;k<subvect_size;k++)
            shape[k] = (spx_word16_t)shape_cb[i*subvect_size+k];
        E[i]=0;

        /* Compute codeword response using convolution with impulse response */
        for(j=0;j<subvect_size;j++)
        {
            spx_word32_t resj=0;
            spx_word16_t res16;
            for (k=0;k<=j;k++)
                resj = MAC16_16(resj,shape[k],r[j-k]);
#ifdef FIXED_POINT
            res16 = EXTRACT16(SHR32(resj, 13));
#else
            res16 = 0.03125f*resj;
#endif
            /* Compute codeword energy */
            E[i]=MAC16_16(E[i],res16,res16);
            res[j] = res16;
            /*printf ("%d\n", (int)res[j]);*/
        }
    }
}

void vq_nbest(spx_word16_t *in, const spx_word16_t *codebook, int len, int entries, spx_word32_t *E, int N, int *nbest, spx_word32_t *best_dist, char *stack)
{
    int i,j,k,used;
    used = 0;
    for (i=0;i<entries;i++)
    {
        spx_word32_t dist=0;
        for (j=0;j<len;j++)
            dist = MAC16_16(dist,in[j],*codebook++);
#ifdef FIXED_POINT
        dist=SUB32(SHR32(E[i],1),dist);
#else
        dist=.5f*E[i]-dist;
#endif
        if (i<N || dist<best_dist[N-1])
        {
            for (k=N-1; (k >= 1) && (k > used || dist < best_dist[k-1]); k--)
            {
                best_dist[k]=best_dist[k-1];
                nbest[k] = nbest[k-1];
            }
            best_dist[k]=dist;
            nbest[k]=i;
            used++;
        }
    }
}

void vq_nbest_sign(spx_word16_t *in, const spx_word16_t *codebook, int len, int entries, spx_word32_t *E, int N, int *nbest, spx_word32_t *best_dist, char *stack)
{
    int i,j,k, sign, used;
    used=0;
    for (i=0;i<entries;i++)
    {
        spx_word32_t dist=0;
        for (j=0;j<len;j++)
            dist = MAC16_16(dist,in[j],*codebook++);
        if (dist>0)
        {
            sign=0;
            dist=-dist;
        } else
        {
            sign=1;
        }
#ifdef FIXED_POINT
        dist = ADD32(dist,SHR32(E[i],1));
#else
        dist = ADD32(dist,.5f*E[i]);
#endif
        if (i<N || dist<best_dist[N-1])
        {
            for (k=N-1; (k >= 1) && (k > used || dist < best_dist[k-1]); k--)
            {
                best_dist[k]=best_dist[k-1];
                nbest[k] = nbest[k-1];
            }
            best_dist[k]=dist;
            nbest[k]=i;
            used++;
            if (sign)
                nbest[k]+=entries;
        }
    }
}

static inline void target_update(spx_word16_t *t, spx_word16_t g, spx_word16_t *r, int len)
{
    int n;
    for (n=0;n<len;n++)
        t[n] = SUB16(t[n],PSHR32(MULT16_16(g,r[n]),13));
}

static void split_cb_search_shape_sign_N1(
        spx_word16_t target[],			/* target vector */
        spx_coef_t ak[],			/* LPCs for this subframe */
        spx_coef_t awk1[],			/* Weighted LPCs for this subframe */
        spx_coef_t awk2[],			/* Weighted LPCs for this subframe */
        const void *par,                      /* Codebook/search parameters*/
        int   p,                        /* number of LPC coeffs */
        int   nsf,                      /* number of samples in subframe */
        spx_sig_t *exc,
        spx_word16_t *r,
        SpeexBits *bits,
        char *stack,
        int   update_target
)
{
    int i,j,m,q;
    VARDECL(spx_word16_t *resp);
#ifdef _USE_SSE
    VARDECL(__m128 *resp2);
   VARDECL(__m128 *E);
#else
    spx_word16_t *resp2;
    VARDECL(spx_word32_t *E);
#endif
    VARDECL(spx_word16_t *t);
    VARDECL(spx_sig_t *e);
    const signed char *shape_cb;
    int shape_cb_size, subvect_size, nb_subvect;
    const split_cb_params *params;
    int best_index;
    spx_word32_t best_dist;
    int have_sign;

    params = (const split_cb_params *) par;
    subvect_size = params->subvect_size;
    nb_subvect = params->nb_subvect;
    shape_cb_size = 1<<params->shape_bits;
    shape_cb = params->shape_cb;
    have_sign = params->have_sign;
    ALLOC(resp, shape_cb_size*subvect_size, spx_word16_t);
#ifdef _USE_SSE
    ALLOC(resp2, (shape_cb_size*subvect_size)>>2, __m128);
   ALLOC(E, shape_cb_size>>2, __m128);
#else
    resp2 = resp;
    ALLOC(E, shape_cb_size, spx_word32_t);
#endif
    ALLOC(t, nsf, spx_word16_t);
    ALLOC(e, nsf, spx_sig_t);

    /* FIXME: Do we still need to copy the target? */
    SPEEX_COPY(t, target, nsf);

    compute_weighted_codebook(shape_cb, r, resp, resp2, E, shape_cb_size, subvect_size, stack);

    for (i=0;i<nb_subvect;i++)
    {
        spx_word16_t *x=t+subvect_size*i;
        /*Find new n-best based on previous n-best j*/
#ifndef DISABLE_WIDEBAND
        if (have_sign)
            vq_nbest_sign(x, resp2, subvect_size, shape_cb_size, E, 1, &best_index, &best_dist, stack);
        else
#endif /* DISABLE_WIDEBAND */
            vq_nbest(x, resp2, subvect_size, shape_cb_size, E, 1, &best_index, &best_dist, stack);

        speex_bits_pack(bits,best_index,params->shape_bits+have_sign);

        {
            int rind;
            spx_word16_t *res;
            spx_word16_t sign=1;
            rind = best_index;
            if (rind>=shape_cb_size)
            {
                sign=-1;
                rind-=shape_cb_size;
            }
            res = resp+rind*subvect_size;
            if (sign>0)
                for (m=0;m<subvect_size;m++)
                    t[subvect_size*i+m] = SUB16(t[subvect_size*i+m], res[m]);
            else
                for (m=0;m<subvect_size;m++)
                    t[subvect_size*i+m] = ADD16(t[subvect_size*i+m], res[m]);

#ifdef FIXED_POINT
            if (sign==1)
         {
            for (j=0;j<subvect_size;j++)
               e[subvect_size*i+j]=SHL32(EXTEND32(shape_cb[rind*subvect_size+j]),SIG_SHIFT-5);
         } else {
            for (j=0;j<subvect_size;j++)
               e[subvect_size*i+j]=NEG32(SHL32(EXTEND32(shape_cb[rind*subvect_size+j]),SIG_SHIFT-5));
         }
#else
            for (j=0;j<subvect_size;j++)
                e[subvect_size*i+j]=sign*0.03125*shape_cb[rind*subvect_size+j];
#endif

        }

        for (m=0;m<subvect_size;m++)
        {
            spx_word16_t g;
            int rind;
            spx_word16_t sign=1;
            rind = best_index;
            if (rind>=shape_cb_size)
            {
                sign=-1;
                rind-=shape_cb_size;
            }

            q=subvect_size-m;
#ifdef FIXED_POINT
            g=sign*shape_cb[rind*subvect_size+m];
#else
            g=sign*0.03125*shape_cb[rind*subvect_size+m];
#endif
            target_update(t+subvect_size*(i+1), g, r+q, nsf-subvect_size*(i+1));
        }
    }

    /* Update excitation */
    /* FIXME: We could update the excitation directly above */
    for (j=0;j<nsf;j++)
        exc[j]=ADD32(exc[j],e[j]);

    /* Update target: only update target if necessary */
    if (update_target)
    {
        VARDECL(spx_word16_t *r2);
        ALLOC(r2, nsf, spx_word16_t);
        for (j=0;j<nsf;j++)
            r2[j] = EXTRACT16(PSHR32(e[j] ,6));
        syn_percep_zero16(r2, ak, awk1, awk2, r2, nsf,p, stack);
        for (j=0;j<nsf;j++)
            target[j]=SUB16(target[j],PSHR16(r2[j],2));
    }
}

void split_cb_search_shape_sign(
        spx_word16_t target[],			/* target vector */
        spx_coef_t ak[],			/* LPCs for this subframe */
        spx_coef_t awk1[],			/* Weighted LPCs for this subframe */
        spx_coef_t awk2[],			/* Weighted LPCs for this subframe */
        const void *par,                      /* Codebook/search parameters*/
        int   p,                        /* number of LPC coeffs */
        int   nsf,                      /* number of samples in subframe */
        spx_sig_t *exc,
        spx_word16_t *r,
        SpeexBits *bits,
        char *stack,
        int   complexity,
        int   update_target
)
{
    int i,j,k,m,n,q;
    VARDECL(spx_word16_t *resp);
#ifdef _USE_SSE
    VARDECL(__m128 *resp2);
   VARDECL(__m128 *E);
#else
    spx_word16_t *resp2;
    VARDECL(spx_word32_t *E);
#endif
    VARDECL(spx_word16_t *t);
    VARDECL(spx_sig_t *e);
    VARDECL(spx_word16_t *tmp);
    VARDECL(spx_word32_t *ndist);
    VARDECL(spx_word32_t *odist);
    VARDECL(int *itmp);
    VARDECL(spx_word16_t **ot2);
    VARDECL(spx_word16_t **nt2);
    spx_word16_t **ot, **nt;
    VARDECL(int **nind);
    VARDECL(int **oind);
    VARDECL(int *ind);
    const signed char *shape_cb;
    int shape_cb_size, subvect_size, nb_subvect;
    const split_cb_params *params;
    int N=2;
    VARDECL(int *best_index);
    VARDECL(spx_word32_t *best_dist);
    VARDECL(int *best_nind);
    VARDECL(int *best_ntarget);
    int have_sign;
    N=complexity;
    if (N>10)
        N=10;
    /* Complexity isn't as important for the codebooks as it is for the pitch */
    N=(2*N)/3;
    if (N<1)
        N=1;
    if (N==1)
    {
        split_cb_search_shape_sign_N1(target,ak,awk1,awk2,par,p,nsf,exc,r,bits,stack,update_target);
        return;
    }
    ALLOC(ot2, N, spx_word16_t*);
    ALLOC(nt2, N, spx_word16_t*);
    ALLOC(oind, N, int*);
    ALLOC(nind, N, int*);

    params = (const split_cb_params *) par;
    subvect_size = params->subvect_size;
    nb_subvect = params->nb_subvect;
    shape_cb_size = 1<<params->shape_bits;
    shape_cb = params->shape_cb;
    have_sign = params->have_sign;
    ALLOC(resp, shape_cb_size*subvect_size, spx_word16_t);
#ifdef _USE_SSE
    ALLOC(resp2, (shape_cb_size*subvect_size)>>2, __m128);
   ALLOC(E, shape_cb_size>>2, __m128);
#else
    resp2 = resp;
    ALLOC(E, shape_cb_size, spx_word32_t);
#endif
    ALLOC(t, nsf, spx_word16_t);
    ALLOC(e, nsf, spx_sig_t);
    ALLOC(ind, nb_subvect, int);

    ALLOC(tmp, 2*N*nsf, spx_word16_t);
    for (i=0;i<N;i++)
    {
        ot2[i]=tmp+2*i*nsf;
        nt2[i]=tmp+(2*i+1)*nsf;
    }
    ot=ot2;
    nt=nt2;
    ALLOC(best_index, N, int);
    ALLOC(best_dist, N, spx_word32_t);
    ALLOC(best_nind, N, int);
    ALLOC(best_ntarget, N, int);
    ALLOC(ndist, N, spx_word32_t);
    ALLOC(odist, N, spx_word32_t);

    ALLOC(itmp, 2*N*nb_subvect, int);
    for (i=0;i<N;i++)
    {
        nind[i]=itmp+2*i*nb_subvect;
        oind[i]=itmp+(2*i+1)*nb_subvect;
    }

    SPEEX_COPY(t, target, nsf);

    for (j=0;j<N;j++)
        SPEEX_COPY(&ot[j][0], t, nsf);

    /* Pre-compute codewords response and energy */
    compute_weighted_codebook(shape_cb, r, resp, resp2, E, shape_cb_size, subvect_size, stack);

    for (j=0;j<N;j++)
        odist[j]=0;

    /*For all subvectors*/
    for (i=0;i<nb_subvect;i++)
    {
        /*"erase" nbest list*/
        for (j=0;j<N;j++)
            ndist[j]=VERY_LARGE32;
        /* This is not strictly necessary, but it provides an additonal safety
           to prevent crashes in case something goes wrong in the previous
           steps (e.g. NaNs) */
        for (j=0;j<N;j++)
            best_nind[j] = best_ntarget[j] = 0;
        /*For all n-bests of previous subvector*/
        for (j=0;j<N;j++)
        {
            spx_word16_t *x=ot[j]+subvect_size*i;
            spx_word32_t tener = 0;
            for (m=0;m<subvect_size;m++)
                tener = MAC16_16(tener, x[m],x[m]);
#ifdef FIXED_POINT
            tener = SHR32(tener,1);
#else
            tener *= .5;
#endif
            /*Find new n-best based on previous n-best j*/
#ifndef DISABLE_WIDEBAND
            if (have_sign)
                vq_nbest_sign(x, resp2, subvect_size, shape_cb_size, E, N, best_index, best_dist, stack);
            else
#endif /* DISABLE_WIDEBAND */
                vq_nbest(x, resp2, subvect_size, shape_cb_size, E, N, best_index, best_dist, stack);

            /*For all new n-bests*/
            for (k=0;k<N;k++)
            {
                /* Compute total distance (including previous sub-vectors */
                spx_word32_t err = ADD32(ADD32(odist[j],best_dist[k]),tener);

                /*update n-best list*/
                if (err<ndist[N-1])
                {
                    for (m=0;m<N;m++)
                    {
                        if (err < ndist[m])
                        {
                            for (n=N-1;n>m;n--)
                            {
                                ndist[n] = ndist[n-1];
                                best_nind[n] = best_nind[n-1];
                                best_ntarget[n] = best_ntarget[n-1];
                            }
                            /* n is equal to m here, so they're interchangeable */
                            ndist[m] = err;
                            best_nind[n] = best_index[k];
                            best_ntarget[n] = j;
                            break;
                        }
                    }
                }
            }
            if (i==0)
                break;
        }
        for (j=0;j<N;j++)
        {
            /*previous target (we don't care what happened before*/
            for (m=(i+1)*subvect_size;m<nsf;m++)
                nt[j][m]=ot[best_ntarget[j]][m];

            /* New code: update the rest of the target only if it's worth it */
            for (m=0;m<subvect_size;m++)
            {
                spx_word16_t g;
                int rind;
                spx_word16_t sign=1;
                rind = best_nind[j];
                if (rind>=shape_cb_size)
                {
                    sign=-1;
                    rind-=shape_cb_size;
                }

                q=subvect_size-m;
#ifdef FIXED_POINT
                g=sign*shape_cb[rind*subvect_size+m];
#else
                g=sign*0.03125*shape_cb[rind*subvect_size+m];
#endif
                target_update(nt[j]+subvect_size*(i+1), g, r+q, nsf-subvect_size*(i+1));
            }

            for (q=0;q<nb_subvect;q++)
                nind[j][q]=oind[best_ntarget[j]][q];
            nind[j][i]=best_nind[j];
        }

        /*update old-new data*/
        /* just swap pointers instead of a long copy */
        {
            spx_word16_t **tmp2;
            tmp2=ot;
            ot=nt;
            nt=tmp2;
        }
        for (j=0;j<N;j++)
            for (m=0;m<nb_subvect;m++)
                oind[j][m]=nind[j][m];
        for (j=0;j<N;j++)
            odist[j]=ndist[j];
    }

    /*save indices*/
    for (i=0;i<nb_subvect;i++)
    {
        ind[i]=nind[0][i];
        speex_bits_pack(bits,ind[i],params->shape_bits+have_sign);
    }

    /* Put everything back together */
    for (i=0;i<nb_subvect;i++)
    {
        int rind;
        spx_word16_t sign=1;
        rind = ind[i];
        if (rind>=shape_cb_size)
        {
            sign=-1;
            rind-=shape_cb_size;
        }
#ifdef FIXED_POINT
        if (sign==1)
      {
         for (j=0;j<subvect_size;j++)
            e[subvect_size*i+j]=SHL32(EXTEND32(shape_cb[rind*subvect_size+j]),SIG_SHIFT-5);
      } else {
         for (j=0;j<subvect_size;j++)
            e[subvect_size*i+j]=NEG32(SHL32(EXTEND32(shape_cb[rind*subvect_size+j]),SIG_SHIFT-5));
      }
#else
        for (j=0;j<subvect_size;j++)
            e[subvect_size*i+j]=sign*0.03125*shape_cb[rind*subvect_size+j];
#endif
    }
    /* Update excitation */
    for (j=0;j<nsf;j++)
        exc[j]=ADD32(exc[j],e[j]);

    /* Update target: only update target if necessary */
    if (update_target)
    {
        VARDECL(spx_word16_t *r2);
        ALLOC(r2, nsf, spx_word16_t);
        for (j=0;j<nsf;j++)
            r2[j] = EXTRACT16(PSHR32(e[j] ,6));
        syn_percep_zero16(r2, ak, awk1, awk2, r2, nsf,p, stack);
        for (j=0;j<nsf;j++)
            target[j]=SUB16(target[j],PSHR16(r2[j],2));
    }
}

void split_cb_shape_sign_unquant(
        spx_sig_t *exc,
        const void *par,                      /* non-overlapping codebook */
        int   nsf,                      /* number of samples in subframe */
        SpeexBits *bits,
        char *stack,
        spx_uint32_t *seed
)
{
    int i,j;
    VARDECL(int *ind);
    VARDECL(int *signs);
    const signed char *shape_cb;
    int subvect_size, nb_subvect;
    const split_cb_params *params;
    int have_sign;

    params = (const split_cb_params *) par;
    subvect_size = params->subvect_size;
    nb_subvect = params->nb_subvect;

    shape_cb = params->shape_cb;
    have_sign = params->have_sign;

    ALLOC(ind, nb_subvect, int);
    ALLOC(signs, nb_subvect, int);

    /* Decode codewords and gains */
    for (i=0;i<nb_subvect;i++)
    {
        if (have_sign)
            signs[i] = speex_bits_unpack_unsigned(bits, 1);
        else
            signs[i] = 0;
        ind[i] = speex_bits_unpack_unsigned(bits, params->shape_bits);
    }
    /* Compute decoded excitation */
    for (i=0;i<nb_subvect;i++)
    {
        spx_word16_t s=1;
        if (signs[i])
            s=-1;
#ifdef FIXED_POINT
        if (s==1)
      {
         for (j=0;j<subvect_size;j++)
            exc[subvect_size*i+j]=SHL32(EXTEND32(shape_cb[ind[i]*subvect_size+j]),SIG_SHIFT-5);
      } else {
         for (j=0;j<subvect_size;j++)
            exc[subvect_size*i+j]=NEG32(SHL32(EXTEND32(shape_cb[ind[i]*subvect_size+j]),SIG_SHIFT-5));
      }
#else
        for (j=0;j<subvect_size;j++)
            exc[subvect_size*i+j]+=s*0.03125*shape_cb[ind[i]*subvect_size+j];
#endif
    }
}

/* Split-VQ innovation parameters for low bit-rate narrowband */
static const split_cb_params split_cb_nb_lbr = {
        10,              /*subvect_size*/
        4,               /*nb_subvect*/
        exc_10_32_table, /*shape_cb*/
        5,               /*shape_bits*/
        0,
};

/* Parameters for Long-Term Prediction (LTP)*/
static const ltp_params ltp_params_lbr = {
        gain_cdbk_lbr,
        5,
        7
};

static inline spx_word32_t compute_pitch_error(spx_word16_t *C, spx_word16_t *g, spx_word16_t pitch_control)
{
    spx_word32_t sum = 0;
    sum = ADD32(sum,MULT16_16(MULT16_16_16(g[0],pitch_control),C[0]));
    sum = ADD32(sum,MULT16_16(MULT16_16_16(g[1],pitch_control),C[1]));
    sum = ADD32(sum,MULT16_16(MULT16_16_16(g[2],pitch_control),C[2]));
    sum = SUB32(sum,MULT16_16(MULT16_16_16(g[0],g[1]),C[3]));
    sum = SUB32(sum,MULT16_16(MULT16_16_16(g[2],g[1]),C[4]));
    sum = SUB32(sum,MULT16_16(MULT16_16_16(g[2],g[0]),C[5]));
    sum = SUB32(sum,MULT16_16(MULT16_16_16(g[0],g[0]),C[6]));
    sum = SUB32(sum,MULT16_16(MULT16_16_16(g[1],g[1]),C[7]));
    sum = SUB32(sum,MULT16_16(MULT16_16_16(g[2],g[2]),C[8]));
    return sum;
}

static int pitch_gain_search_3tap_vq(
        const signed char *gain_cdbk,
        int                gain_cdbk_size,
        spx_word16_t      *C16,
        spx_word16_t       max_gain
)
{
    const signed char *ptr=gain_cdbk;
    int                best_cdbk=0;
    spx_word32_t       best_sum=-VERY_LARGE32;
    spx_word32_t       sum=0;
    spx_word16_t       g[3];
    spx_word16_t       pitch_control=64;
    spx_word16_t       gain_sum;
    int                i;

    for (i=0;i<gain_cdbk_size;i++) {

        ptr = gain_cdbk+4*i;
        g[0]=ADD16((spx_word16_t)ptr[0],32);
        g[1]=ADD16((spx_word16_t)ptr[1],32);
        g[2]=ADD16((spx_word16_t)ptr[2],32);
        gain_sum = (spx_word16_t)ptr[3];

        sum = compute_pitch_error(C16, g, pitch_control);

        if (sum>best_sum && gain_sum<=max_gain) {
            best_sum=sum;
            best_cdbk=i;
        }
    }

    return best_cdbk;
}

/** Finds the best quantized 3-tap pitch predictor by analysis by synthesis */
static spx_word32_t pitch_gain_search_3tap(
        const spx_word16_t target[],       /* Target vector */
        const spx_coef_t ak[],          /* LPCs for this subframe */
        const spx_coef_t awk1[],        /* Weighted LPCs #1 for this subframe */
        const spx_coef_t awk2[],        /* Weighted LPCs #2 for this subframe */
        spx_sig_t exc[],                /* Excitation */
        const signed char *gain_cdbk,
        int gain_cdbk_size,
        int   pitch,                    /* Pitch value */
        int   p,                        /* Number of LPC coeffs */
        int   nsf,                      /* Number of samples in subframe */
        SpeexBits *bits,
        char *stack,
        const spx_word16_t *exc2,
        const spx_word16_t *r,
        spx_word16_t *new_target,
        int  *cdbk_index,
        int plc_tuning,
        spx_word32_t cumul_gain,
        int scaledown
)
{
    int i,j;
    VARDECL(spx_word16_t *tmp1);
    VARDECL(spx_word16_t *e);
    spx_word16_t *x[3];
    spx_word32_t corr[3];
    spx_word32_t A[3][3];
    spx_word16_t gain[3];
    spx_word32_t err;
    spx_word16_t max_gain=128;
    int          best_cdbk=0;

    ALLOC(tmp1, 3*nsf, spx_word16_t);
    ALLOC(e, nsf, spx_word16_t);

    if (cumul_gain > 262144)
        max_gain = 31;

    x[0]=tmp1;
    x[1]=tmp1+nsf;
    x[2]=tmp1+2*nsf;

    for (j=0;j<nsf;j++)
        new_target[j] = target[j];

    {
        int bound;
        VARDECL(spx_mem_t *mm);
        int pp=pitch-1;
        ALLOC(mm, p, spx_mem_t);
        bound = nsf;
        if (nsf-pp>0)
            bound = pp;
        for (j=0;j<bound;j++)
            e[j]=exc2[j-pp];
        bound = nsf;
        if (nsf-pp-pitch>0)
            bound = pp+pitch;
        for (;j<bound;j++)
            e[j]=exc2[j-pp-pitch];
        for (;j<nsf;j++)
            e[j]=0;
#ifdef FIXED_POINT
        /* Scale target and excitation down if needed (avoiding overflow) */
      if (scaledown)
      {
         for (j=0;j<nsf;j++)
            e[j] = SHR16(e[j],1);
         for (j=0;j<nsf;j++)
            new_target[j] = SHR16(new_target[j],1);
      }
#endif
        for (j=0;j<p;j++)
            mm[j] = 0;
        iir_mem16(e, ak, e, nsf, p, mm, stack);
        for (j=0;j<p;j++)
            mm[j] = 0;
        filter10(e, awk1, awk2, e, nsf, mm, stack);
        for (j=0;j<nsf;j++)
            x[2][j] = e[j];
    }
    for (i=1;i>=0;i--)
    {
        spx_word16_t e0=exc2[-pitch-1+i];
#ifdef FIXED_POINT
        /* Scale excitation down if needed (avoiding overflow) */
      if (scaledown)
         e0 = SHR16(e0,1);
#endif
        x[i][0]=MULT16_16_Q14(r[0], e0);
        for (j=0;j<nsf-1;j++)
            x[i][j+1]=ADD32(x[i+1][j],MULT16_16_P14(r[j+1], e0));
    }

    for (i=0;i<3;i++)
        corr[i]=inner_prod(x[i],new_target,nsf);
    for (i=0;i<3;i++)
        for (j=0;j<=i;j++)
            A[i][j]=A[j][i]=inner_prod(x[i],x[j],nsf);

    {
        spx_word32_t C[9];
#ifdef FIXED_POINT
        spx_word16_t C16[9];
#else
        spx_word16_t *C16=C;
#endif
        C[0]=corr[2];
        C[1]=corr[1];
        C[2]=corr[0];
        C[3]=A[1][2];
        C[4]=A[0][1];
        C[5]=A[0][2];
        C[6]=A[2][2];
        C[7]=A[1][1];
        C[8]=A[0][0];

        /*plc_tuning *= 2;*/
        if (plc_tuning<2)
            plc_tuning=2;
        if (plc_tuning>30)
            plc_tuning=30;
#ifdef FIXED_POINT
        C[0] = SHL32(C[0],1);
      C[1] = SHL32(C[1],1);
      C[2] = SHL32(C[2],1);
      C[3] = SHL32(C[3],1);
      C[4] = SHL32(C[4],1);
      C[5] = SHL32(C[5],1);
      C[6] = MAC16_32_Q15(C[6],MULT16_16_16(plc_tuning,655),C[6]);
      C[7] = MAC16_32_Q15(C[7],MULT16_16_16(plc_tuning,655),C[7]);
      C[8] = MAC16_32_Q15(C[8],MULT16_16_16(plc_tuning,655),C[8]);
      normalize16(C, C16, 32767, 9);
#else
        C[6]*=.5*(1+.02*plc_tuning);
        C[7]*=.5*(1+.02*plc_tuning);
        C[8]*=.5*(1+.02*plc_tuning);
#endif

        best_cdbk = pitch_gain_search_3tap_vq(gain_cdbk, gain_cdbk_size, C16, max_gain);

#ifdef FIXED_POINT
        gain[0] = ADD16(32,(spx_word16_t)gain_cdbk[best_cdbk*4]);
      gain[1] = ADD16(32,(spx_word16_t)gain_cdbk[best_cdbk*4+1]);
      gain[2] = ADD16(32,(spx_word16_t)gain_cdbk[best_cdbk*4+2]);
      /*printf ("%d %d %d %d\n",gain[0],gain[1],gain[2], best_cdbk);*/
#else
        gain[0] = 0.015625*gain_cdbk[best_cdbk*4]  + .5;
        gain[1] = 0.015625*gain_cdbk[best_cdbk*4+1]+ .5;
        gain[2] = 0.015625*gain_cdbk[best_cdbk*4+2]+ .5;
#endif
        *cdbk_index=best_cdbk;
    }

    SPEEX_MEMSET(exc, 0, nsf);
    for (i=0;i<3;i++)
    {
        int j;
        int tmp1, tmp3;
        int pp=pitch+1-i;
        tmp1=nsf;
        if (tmp1>pp)
            tmp1=pp;
        for (j=0;j<tmp1;j++)
            exc[j]=MAC16_16(exc[j],SHL16(gain[2-i],7),exc2[j-pp]);
        tmp3=nsf;
        if (tmp3>pp+pitch)
            tmp3=pp+pitch;
        for (j=tmp1;j<tmp3;j++)
            exc[j]=MAC16_16(exc[j],SHL16(gain[2-i],7),exc2[j-pp-pitch]);
    }
    for (i=0;i<nsf;i++)
    {
        spx_word32_t tmp = ADD32(ADD32(MULT16_16(gain[0],x[2][i]),MULT16_16(gain[1],x[1][i])),
                                 MULT16_16(gain[2],x[0][i]));
        new_target[i] = SUB16(new_target[i], EXTRACT16(PSHR32(tmp,6)));
    }
    err = inner_prod(new_target, new_target, nsf);

    return err;
}

/** Finds the best quantized 3-tap pitch predictor by analysis by synthesis */
int pitch_search_3tap(
        spx_word16_t target[],                 /* Target vector */
        spx_word16_t *sw,
        spx_coef_t ak[],                     /* LPCs for this subframe */
        spx_coef_t awk1[],                   /* Weighted LPCs #1 for this subframe */
        spx_coef_t awk2[],                   /* Weighted LPCs #2 for this subframe */
        spx_sig_t exc[],                    /* Excitation */
        const void *par,
        int   start,                    /* Smallest pitch value allowed */
        int   end,                      /* Largest pitch value allowed */
        spx_word16_t pitch_coef,               /* Voicing (pitch) coefficient */
        int   p,                        /* Number of LPC coeffs */
        int   nsf,                      /* Number of samples in subframe */
        SpeexBits *bits,
        char *stack,
        spx_word16_t *exc2,
        spx_word16_t *r,
        int complexity,
        int cdbk_offset,
        int plc_tuning,
        spx_word32_t *cumul_gain
)
{
    int i;
    int cdbk_index, pitch=0, best_gain_index=0;
    VARDECL(spx_sig_t *best_exc);
    VARDECL(spx_word16_t *new_target);
    VARDECL(spx_word16_t *best_target);
    int best_pitch=0;
    spx_word32_t err, best_err=-1;
    int N;
    const ltp_params *params;
    const signed char *gain_cdbk;
    int   gain_cdbk_size;
    int scaledown=0;

    VARDECL(int *nbest);

    params = (const ltp_params*) par;
    gain_cdbk_size = 1<<params->gain_bits;
    gain_cdbk = params->gain_cdbk + 4*gain_cdbk_size*cdbk_offset;

    N=complexity;
    if (N>10)
        N=10;
    if (N<1)
        N=1;

    ALLOC(nbest, N, int);
    params = (const ltp_params*) par;

    if (end<start)
    {
        speex_bits_pack(bits, 0, params->pitch_bits);
        speex_bits_pack(bits, 0, params->gain_bits);
        SPEEX_MEMSET(exc, 0, nsf);
        return start;
    }

#ifdef FIXED_POINT
    /* Check if we need to scale everything down in the pitch search to avoid overflows */
   for (i=0;i<nsf;i++)
   {
      if (ABS16(target[i])>16383)
      {
         scaledown=1;
         break;
      }
   }
   for (i=-end;i<0;i++)
   {
      if (ABS16(exc2[i])>16383)
      {
         scaledown=1;
         break;
      }
   }
#endif
    if (N>end-start+1)
        N=end-start+1;
    if (end != start)
        open_loop_nbest_pitch(sw, start, end, nsf, nbest, NULL, N, stack);
    else
        nbest[0] = start;

    ALLOC(best_exc, nsf, spx_sig_t);
    ALLOC(new_target, nsf, spx_word16_t);
    ALLOC(best_target, nsf, spx_word16_t);

    for (i=0;i<N;i++)
    {
        pitch=nbest[i];
        SPEEX_MEMSET(exc, 0, nsf);
        err=pitch_gain_search_3tap(target, ak, awk1, awk2, exc, gain_cdbk, gain_cdbk_size, pitch, p, nsf,
                                   bits, stack, exc2, r, new_target, &cdbk_index, plc_tuning, *cumul_gain, scaledown);
        if (err<best_err || best_err<0)
        {
            SPEEX_COPY(best_exc, exc, nsf);
            SPEEX_COPY(best_target, new_target, nsf);
            best_err=err;
            best_pitch=pitch;
            best_gain_index=cdbk_index;
        }
    }
    /*printf ("pitch: %d %d\n", best_pitch, best_gain_index);*/
    speex_bits_pack(bits, best_pitch-start, params->pitch_bits);
    speex_bits_pack(bits, best_gain_index, params->gain_bits);
#ifdef FIXED_POINT
    *cumul_gain = MULT16_32_Q13(SHL16(params->gain_cdbk[4*best_gain_index+3],8), MAX32(1024,*cumul_gain));
#else
    *cumul_gain = 0.03125*MAX32(1024,*cumul_gain)*params->gain_cdbk[4*best_gain_index+3];
#endif
    /*printf ("%f\n", cumul_gain);*/
    /*printf ("encode pitch: %d %d\n", best_pitch, best_gain_index);*/
    SPEEX_COPY(exc, best_exc, nsf);
    SPEEX_COPY(target, best_target, nsf);
#ifdef FIXED_POINT
    /* Scale target back up if needed */
   if (scaledown)
   {
      for (i=0;i<nsf;i++)
         target[i]=SHL16(target[i],1);
   }
#endif
    return pitch;
}

void pitch_unquant_3tap(
        spx_word16_t exc[],             /* Input excitation */
        spx_word32_t exc_out[],         /* Output excitation */
        int   start,                    /* Smallest pitch value allowed */
        int   end,                      /* Largest pitch value allowed */
        spx_word16_t pitch_coef,        /* Voicing (pitch) coefficient */
        const void *par,
        int   nsf,                      /* Number of samples in subframe */
        int *pitch_val,
        spx_word16_t *gain_val,
        SpeexBits *bits,
        char *stack,
        int count_lost,
        int subframe_offset,
        spx_word16_t last_pitch_gain,
        int cdbk_offset
)
{
    int i;
    int pitch;
    int gain_index;
    spx_word16_t gain[3];
    const signed char *gain_cdbk;
    int gain_cdbk_size;
    const ltp_params *params;

    params = (const ltp_params*) par;
    gain_cdbk_size = 1<<params->gain_bits;
    gain_cdbk = params->gain_cdbk + 4*gain_cdbk_size*cdbk_offset;

    pitch = speex_bits_unpack_unsigned(bits, params->pitch_bits);
    pitch += start;
    gain_index = speex_bits_unpack_unsigned(bits, params->gain_bits);
    /*printf ("decode pitch: %d %d\n", pitch, gain_index);*/
#ifdef FIXED_POINT
    gain[0] = ADD16(32,(spx_word16_t)gain_cdbk[gain_index*4]);
   gain[1] = ADD16(32,(spx_word16_t)gain_cdbk[gain_index*4+1]);
   gain[2] = ADD16(32,(spx_word16_t)gain_cdbk[gain_index*4+2]);
#else
    gain[0] = 0.015625*gain_cdbk[gain_index*4]+.5;
    gain[1] = 0.015625*gain_cdbk[gain_index*4+1]+.5;
    gain[2] = 0.015625*gain_cdbk[gain_index*4+2]+.5;
#endif

    if (count_lost && pitch > subframe_offset)
    {
        spx_word16_t gain_sum;
        if (1) {
#ifdef FIXED_POINT
            spx_word16_t tmp = count_lost < 4 ? last_pitch_gain : SHR16(last_pitch_gain,1);
         if (tmp>62)
            tmp=62;
#else
            spx_word16_t tmp = count_lost < 4 ? last_pitch_gain : 0.5 * last_pitch_gain;
            if (tmp>.95)
                tmp=.95;
#endif
            gain_sum = gain_3tap_to_1tap(gain);

            if (gain_sum > tmp)
            {
                spx_word16_t fact = DIV32_16(SHL32(EXTEND32(tmp),14),gain_sum);
                for (i=0;i<3;i++)
                    gain[i]=MULT16_16_Q14(fact,gain[i]);
            }

        }

    }

    *pitch_val = pitch;
    gain_val[0]=gain[0];
    gain_val[1]=gain[1];
    gain_val[2]=gain[2];
    gain[0] = SHL16(gain[0],7);
    gain[1] = SHL16(gain[1],7);
    gain[2] = SHL16(gain[2],7);
    SPEEX_MEMSET(exc_out, 0, nsf);
    for (i=0;i<3;i++)
    {
        int j;
        int tmp1, tmp3;
        int pp=pitch+1-i;
        tmp1=nsf;
        if (tmp1>pp)
            tmp1=pp;
        for (j=0;j<tmp1;j++)
            exc_out[j]=MAC16_16(exc_out[j],gain[2-i],exc[j-pp]);
        tmp3=nsf;
        if (tmp3>pp+pitch)
            tmp3=pp+pitch;
        for (j=tmp1;j<tmp3;j++)
            exc_out[j]=MAC16_16(exc_out[j],gain[2-i],exc[j-pp-pitch]);
    }
    /*for (i=0;i<nsf;i++)
    exc[i]=PSHR32(exc32[i],13);*/
}

/* 8 kbps low bit-rate mode */
static const SpeexSubmode nb_submode3 = {
        -1,
        0,
        1,
        0,
        /*LSP quantization*/
        lsp_quant_lbr,
        lsp_unquant_lbr,
        /*Pitch quantization*/
        pitch_search_3tap,
        pitch_unquant_3tap,
        &ltp_params_lbr,
        /*Innovation quantization*/
        split_cb_search_shape_sign,
        split_cb_shape_sign_unquant,
        &split_cb_nb_lbr,
        QCONST16(.55,15),
        160
};
