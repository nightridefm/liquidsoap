#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/misc.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "image_pixel.h"
#include "image_rgb.h"
#include "image_yuv420.h"

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)

/* SSE2 alpha-blend helpers for caml_yuv420_add. Compiled with a
 * function-level target attribute so callers don't need a global
 * -msse2 flag; dispatch is gated at runtime via __builtin_cpu_supports.
 * Non-x86 builds skip this block entirely and use the scalar path.
 *
 * See the comment block above caml_yuv420_add for the math identities
 * and bit-exactness contract. */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#include <immintrin.h>
#define MM_HAVE_X86_SIMD 1

#if defined(__GNUC__) || defined(__clang__)
#define MM_TARGET_SSE2 __attribute__((target("sse2")))
#else
#define MM_TARGET_SSE2
#endif

/* (s*a + d*(255-a)) for 16 bytes, branchless. Sum overflows u16
 * (max 130050) so we widen to u32 and use the identity
 * x/255 == (x + (x>>8) + 1) >> 8 for u32 x <= ~1015806. */
MM_TARGET_SSE2 static inline __m128i mm_blend16_sse2(__m128i src, __m128i dst,
                                                     __m128i alpha) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i v255 = _mm_set1_epi16(255);
  const __m128i v1_32 = _mm_set1_epi32(1);

  /* Low 8 pixels */
  __m128i s_lo = _mm_unpacklo_epi8(src, zero);
  __m128i d_lo = _mm_unpacklo_epi8(dst, zero);
  __m128i a_lo = _mm_unpacklo_epi8(alpha, zero);
  __m128i ia_lo = _mm_sub_epi16(v255, a_lo);
  __m128i sa_lo = _mm_mullo_epi16(s_lo, a_lo);
  __m128i da_lo = _mm_mullo_epi16(d_lo, ia_lo);
  __m128i sum_ll = _mm_add_epi32(_mm_unpacklo_epi16(sa_lo, zero),
                                 _mm_unpacklo_epi16(da_lo, zero));
  __m128i sum_lh = _mm_add_epi32(_mm_unpackhi_epi16(sa_lo, zero),
                                 _mm_unpackhi_epi16(da_lo, zero));
  sum_ll = _mm_srli_epi32(
      _mm_add_epi32(_mm_add_epi32(sum_ll, _mm_srli_epi32(sum_ll, 8)), v1_32),
      8);
  sum_lh = _mm_srli_epi32(
      _mm_add_epi32(_mm_add_epi32(sum_lh, _mm_srli_epi32(sum_lh, 8)), v1_32),
      8);
  __m128i res_lo = _mm_packs_epi32(sum_ll, sum_lh);

  /* High 8 pixels */
  __m128i s_hi = _mm_unpackhi_epi8(src, zero);
  __m128i d_hi = _mm_unpackhi_epi8(dst, zero);
  __m128i a_hi = _mm_unpackhi_epi8(alpha, zero);
  __m128i ia_hi = _mm_sub_epi16(v255, a_hi);
  __m128i sa_hi = _mm_mullo_epi16(s_hi, a_hi);
  __m128i da_hi = _mm_mullo_epi16(d_hi, ia_hi);
  __m128i sum_hl = _mm_add_epi32(_mm_unpacklo_epi16(sa_hi, zero),
                                 _mm_unpacklo_epi16(da_hi, zero));
  __m128i sum_hh = _mm_add_epi32(_mm_unpackhi_epi16(sa_hi, zero),
                                 _mm_unpackhi_epi16(da_hi, zero));
  sum_hl = _mm_srli_epi32(
      _mm_add_epi32(_mm_add_epi32(sum_hl, _mm_srli_epi32(sum_hl, 8)), v1_32),
      8);
  sum_hh = _mm_srli_epi32(
      _mm_add_epi32(_mm_add_epi32(sum_hh, _mm_srli_epi32(sum_hh, 8)), v1_32),
      8);
  __m128i res_hi = _mm_packs_epi32(sum_hl, sum_hh);

  return _mm_packus_epi16(res_lo, res_hi);
}

/* Alpha-plane compose for 16 bytes:
 *   A_out = 0xff - ((0xff - a_src) * (0xff - A_dst)) / 0xff
 * Branchless; collapses to A_dst when a_src==0 and 0xff when a_src==0xff. */
MM_TARGET_SSE2 static inline __m128i mm_alpha_compose16_sse2(__m128i src_a,
                                                             __m128i dst_a) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i v255 = _mm_set1_epi16(255);
  const __m128i v255b = _mm_set1_epi8((char)0xff);
  const __m128i v1_32 = _mm_set1_epi32(1);

  __m128i isa_lo = _mm_sub_epi16(v255, _mm_unpacklo_epi8(src_a, zero));
  __m128i ida_lo = _mm_sub_epi16(v255, _mm_unpacklo_epi8(dst_a, zero));
  __m128i prod_lo = _mm_mullo_epi16(isa_lo, ida_lo);
  __m128i isa_hi = _mm_sub_epi16(v255, _mm_unpackhi_epi8(src_a, zero));
  __m128i ida_hi = _mm_sub_epi16(v255, _mm_unpackhi_epi8(dst_a, zero));
  __m128i prod_hi = _mm_mullo_epi16(isa_hi, ida_hi);

  __m128i pll = _mm_unpacklo_epi16(prod_lo, zero);
  __m128i plh = _mm_unpackhi_epi16(prod_lo, zero);
  __m128i phl = _mm_unpacklo_epi16(prod_hi, zero);
  __m128i phh = _mm_unpackhi_epi16(prod_hi, zero);
  pll = _mm_srli_epi32(
      _mm_add_epi32(_mm_add_epi32(pll, _mm_srli_epi32(pll, 8)), v1_32), 8);
  plh = _mm_srli_epi32(
      _mm_add_epi32(_mm_add_epi32(plh, _mm_srli_epi32(plh, 8)), v1_32), 8);
  phl = _mm_srli_epi32(
      _mm_add_epi32(_mm_add_epi32(phl, _mm_srli_epi32(phl, 8)), v1_32), 8);
  phh = _mm_srli_epi32(
      _mm_add_epi32(_mm_add_epi32(phh, _mm_srli_epi32(phh, 8)), v1_32), 8);
  __m128i div_lo = _mm_packs_epi32(pll, plh);
  __m128i div_hi = _mm_packs_epi32(phl, phh);
  __m128i div8 = _mm_packus_epi16(div_lo, div_hi);
  return _mm_sub_epi8(v255b, div8);
}

/* Dispatch flag: set on first call, sampled atomically thereafter.
 * 0 = uninitialised, 1 = scalar only, 2 = SSE2 ok. We never downgrade
 * once set, so plain int reads are safe (worst case: a few extra
 * cpuid calls during the race window). */
static int mm_simd_dispatch = 0;
static int mm_simd_pick(void) {
  int d = mm_simd_dispatch;
  if (d) return d;
#if defined(__GNUC__) || defined(__clang__)
  d = __builtin_cpu_supports("sse2") ? 2 : 1;
#else
  d = 1;
#endif
  mm_simd_dispatch = d;
  return d;
}
#else
#define MM_HAVE_X86_SIMD 0
static int mm_simd_pick(void) { return 1; }
#endif

CAMLprim value caml_yuv420_fill(value img, value p, value uv_height) {
  CAMLparam2(img, p);
  int y = Int_val(Field(p, 0));
  int u = Int_val(Field(p, 1));
  int v = Int_val(Field(p, 2));
  int height = YUV420_height(img);
  int y_stride = YUV420_y_stride(img);
  int uv_stride = YUV420_uv_stride(img);
  memset(YUV420_y(img), y, height * y_stride);
  memset(YUV420_u(img), u, Int_val(uv_height) * uv_stride);
  memset(YUV420_v(img), v, Int_val(uv_height) * uv_stride);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv420_to_int_image(value img) {
  CAMLparam1(img);
  CAMLlocal2(ans, tmp);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int i, j;
  int y, u, v, r, g, b, a;
  ans = caml_alloc_tuple(yuv.height);
  for (j = 0; j < yuv.height; j++) {
    tmp = caml_alloc_tuple(yuv.width);
    for (i = 0; i < yuv.width; i++) {
      y = Y(yuv, i, j);
      u = U(yuv, i, j);
      v = V(yuv, i, j);
      r = RofYUV(y, u, v);
      g = GofYUV(y, u, v);
      b = BofYUV(y, u, v);
      if (yuv.alpha) {
        a = A(yuv, i, j);
        r = r * a / 0xff;
        g = g * a / 0xff;
        b = b * a / 0xff;
      }
      Store_field(tmp, i, Val_int((r << 16) + (g << 8) + b));
    }
    Store_field(ans, j, tmp);
  }
  CAMLreturn(ans);
}

CAMLprim value caml_yuv420_of_rgb24_string(value img, value s) {
  CAMLparam2(img, s);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  // We don't copy so we cannot release the lock
  unsigned char *data = (unsigned char *)String_val(s);
  int i, j;

  for (j = 0; j < yuv.height; j++)
    for (i = 0; i < yuv.width; i++) {
      int r = data[3 * (j * yuv.width + i) + 0];
      int g = data[3 * (j * yuv.width + i) + 1];
      int b = data[3 * (j * yuv.width + i) + 2];
      Y(yuv, i, j) = YofRGB(r, g, b);
      if (i % 2 == 0 && j % 2 == 0) {
        U(yuv, i, j) = UofRGB(r, g, b);
        V(yuv, i, j) = VofRGB(r, g, b);
      }
    }

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv420_of_rgba32(value _rgb, value img) {
  CAMLparam2(_rgb, img);
  frame rgb;
  frame_of_value(_rgb, &rgb);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int i, j;

  caml_enter_blocking_section();
  for (j = 0; j < rgb.height; j++)
    for (i = 0; i < rgb.width; i++) {
      int r = Red(&rgb, i, j);
      int g = Green(&rgb, i, j);
      int b = Blue(&rgb, i, j);
      Y(yuv, i, j) = YofRGB(r, g, b);
      A(yuv, i, j) = Alpha(&rgb, i, j);
      if (i % 2 == 0 && j % 2 == 0) {
        U(yuv, i, j) = UofRGB(r, g, b);
        V(yuv, i, j) = VofRGB(r, g, b);
      }
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv420_to_rgba32(value img, value _rgb) {
  CAMLparam2(img, _rgb);
  frame rgb;
  frame_of_value(_rgb, &rgb);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int i, j;

  caml_enter_blocking_section();
  for (j = 0; j < yuv.height; j++)
    for (i = 0; i < yuv.width; i++) {
      int y = Y(yuv, i, j);
      int u = U(yuv, i, j);
      int v = V(yuv, i, j);
      Red(&rgb, i, j) = RofYUV(y, u, v);
      Green(&rgb, i, j) = GofYUV(y, u, v);
      Blue(&rgb, i, j) = BofYUV(y, u, v);
      Alpha(&rgb, i, j) = yuv.alpha ? A(yuv, i, j) : 0xff;
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv420_scale(value _src, value _dst) {
  CAMLparam2(_src, _dst);
  yuv420 src, dst;
  yuv420_of_value(&src, _src);
  yuv420_of_value(&dst, _dst);
  int i, j, is, js;

  assert(!src.alpha || dst.alpha);

  caml_enter_blocking_section();
  /*
  for (j = 0; j < dst.height; j++)
    for (i = 0; i < dst.width; i++) {
      is = i * src.width / dst.width;
      js = j * src.height / dst.height;
      Y(dst, i, j) = Y(src, is, js);
      U(dst, i, j) = U(src, is, js);
      V(dst, i, j) = V(src, is, js);
    }
  */
  for (j = 0; j < dst.height; j++)
    for (i = 0; i < dst.width; i++) {
      is = i * src.width / dst.width;
      js = j * src.height / dst.height;
      Y(dst, i, j) = Y(src, is, js);
    }
  for (j = 0; j < dst.height / 2; j++)
    for (i = 0; i < dst.width / 2; i++) {
      is = i * (src.width / 2) / (dst.width / 2);
      js = j * (src.height / 2) / (dst.height / 2);
      U2(dst, i, j) = U2(src, is, js);
      V2(dst, i, j) = V2(src, is, js);
    }
  if (src.alpha)
    for (j = 0; j < dst.height; j++)
      for (i = 0; i < dst.width; i++) {
        is = i * src.width / dst.width;
        js = j * src.height / dst.height;
        A(dst, i, j) = A(src, is, js);
      }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv420_scale_coef(value _src, value _dst, value xscale,
                                      value yscale) {
  CAMLparam4(_src, _dst, xscale, yscale);
  yuv420 src, dst;
  yuv420_of_value(&src, _src);
  yuv420_of_value(&dst, _dst);
  // x scaling (xn: numerator, xd: denominator)
  int xn = Int_val(Field(xscale, 0));
  int xd = Int_val(Field(xscale, 1));
  // y scaling
  int yn = Int_val(Field(yscale, 0));
  int yd = Int_val(Field(yscale, 1));
  // offsets
  int ox = (dst.width - src.width * xn / xd) / 2;
  int oy = (dst.height - src.height * yn / yd) / 2;
  int i, j;

  assert(ox >= 0 && oy >= 0);

  caml_enter_blocking_section();
  // TODO: blank
  /* if (ox != 0 || oy != 0) rgb_blank(&dst); */
  for (j = oy; j < dst.height - oy; j++)
    for (i = ox; i < dst.width - ox; i++) {
      int is = (i - ox) * xd / xn;
      int js = (j - oy) * yd / yn;
      Y(dst, i, j) = Y(src, is, js);
      if (i % 2 == 0 && j % 2 == 0) {
        U(dst, i, j) = U(src, is, js);
        V(dst, i, j) = V(src, is, js);
      }
      if (src.alpha)
        A(dst, i, j) = A(src, is, js);
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

/* Scalar core kept verbatim from the original implementation, with the
 * sole change that the body is now a static helper so the dispatcher
 * can pick it as the fallback. Used directly when:
 *   - src has no alpha (memcpy/memset is already optimal);
 *   - the host CPU lacks SSE2 (mm_simd_pick() returned 1);
 *   - the build target isn't x86 (MM_HAVE_X86_SIMD == 0). */
static void caml_yuv420_add_scalar(yuv420 src, int x, int y, yuv420 dst,
                                   int ia, int ib, int ja, int jb) {
  int i, j;
  if (src.alpha == NULL) {
    int il = ib - ia;
    for (j = ja; j < jb; j++) {
      int is = ia - x;
      int js = j - y;
      memcpy(dst.y + (j * dst.y_stride + ia), src.y + (js * src.y_stride + is),
             il);
    }
    for (j = ja; j < jb; j += 2) {
      int is = ia - x;
      int js = j - y;
      memcpy(dst.u + (j / 2 * dst.uv_stride + ia / 2),
             src.u + (js / 2 * src.uv_stride + is / 2), il / 2);
      memcpy(dst.v + (j / 2 * dst.uv_stride + ia / 2),
             src.v + (js / 2 * src.uv_stride + is / 2), il / 2);
    }
    if (dst.alpha)
      for (j = ja; j < jb; j++)
        memset(dst.alpha + (j * dst.y_stride + ia), 0xff, il);
    return;
  }

  for (j = ja; j < jb; j++)
    for (i = ia; i < ib; i++) {
      int is = i - x;
      int js = j - y;
      int a = A(src, is, js);

      if (a == 0) {
      } else if (a == 0xff) {
        Y(dst, i, j) = Y(src, is, js);
        U(dst, i, j) = U(src, is, js);
        V(dst, i, j) = V(src, is, js);
        if (dst.alpha)
          A(dst, i, j) = 0xff;
      } else {
        Y(dst, i, j) =
            (Y(src, is, js) * a + Y(dst, i, j) * (0xff - a)) / 0xff;
        if (dst.alpha)
          A(dst, i, j) = 0xff - ((0xff - a) * (0xff - A(dst, i, j))) / 0xff;
        if (i % 2 == 0 && j % 2 == 0) {
          U(dst, i, j) =
              (U(src, is, js) * a + U(dst, i, j) * (0xff - a)) / 0xff;
          V(dst, i, j) =
              (V(src, is, js) * a + V(dst, i, j) * (0xff - a)) / 0xff;
        }
      }
    }
}

#if MM_HAVE_X86_SIMD
/* SSE2 path: bit-exact equivalent of the scalar with-alpha branch.
 *
 * The blend formula (s*a + d*(255-a)) / 255 produces the same result
 * regardless of whether a is 0, 0xff, or in between - the scalar's
 * if/elif fastpaths are pure branch optimisations. Removing them is
 * what lets us SIMD: branchy per-pixel control flow can't vectorise.
 *
 * Y plane and dst alpha plane are blended 16 bytes at a time with
 * mm_blend16_sse2 / mm_alpha_compose16_sse2. U/V planes use a separate
 * chroma-resolution loop that faithfully reproduces scalar's
 * last-write-wins semantics: scalar writes chroma at any of the four
 * full-pixel visits to a chroma cell when alpha there is 0xff, plus
 * blends at visit V1 (even-even) when alpha is mid. The four visits
 * may read DIFFERENT source chroma cells when x or y is odd, so we
 * search visits V4>V3>V2>V1 for the first 0xff fastpath and copy its
 * src cell, falling back to V1's blend when no corner is opaque.
 *
 * (Bit-exactness with scalar is a precondition for landing this
 * upstream: tests/yuv420_simd_test.c diffs both implementations across
 * many shapes and offsets, including odd ones; CI rejects any byte
 * difference. We are not preserving the "U/V iterates at full
 * resolution and writes the same cell three times" quirk - we IS that
 * quirk by reading whichever single src cell scalar's
 * last-write-wins would have settled on.) */
MM_TARGET_SSE2 static void caml_yuv420_add_sse2(yuv420 src, int x, int y,
                                                yuv420 dst, int ia, int ib,
                                                int ja, int jb) {
  const __m128i v_zero = _mm_setzero_si128();
  const __m128i v_ff = _mm_set1_epi8((char)0xff);

  /* --- Y plane: 16-byte SSE2 blend, two block-level fastpaths,
   * scalar tail. Real overlay shapes (transparent margin + opaque
   * body + thin AA edge) hit one of the fastpaths in >95% of blocks. */
  for (int j = ja; j < jb; j++) {
    int is = ia - x;
    int js = j - y;
    unsigned char *dst_row = dst.y + (j * dst.y_stride + ia);
    unsigned char *src_row = src.y + (js * src.y_stride + is);
    unsigned char *a_row = src.alpha + (js * src.y_stride + is);
    int len = ib - ia;
    int k = 0;
    for (; k + 16 <= len; k += 16) {
      __m128i a = _mm_loadu_si128((const __m128i *)(a_row + k));
      /* All-zero alpha: dst unchanged, skip the load+blend+store. */
      if (_mm_movemask_epi8(_mm_cmpeq_epi8(a, v_zero)) == 0xFFFF) continue;
      /* All-max alpha: dst = src, no blend math. */
      if (_mm_movemask_epi8(_mm_cmpeq_epi8(a, v_ff)) == 0xFFFF) {
        _mm_storeu_si128((__m128i *)(dst_row + k),
                         _mm_loadu_si128((const __m128i *)(src_row + k)));
        continue;
      }
      __m128i s = _mm_loadu_si128((const __m128i *)(src_row + k));
      __m128i d = _mm_loadu_si128((const __m128i *)(dst_row + k));
      _mm_storeu_si128((__m128i *)(dst_row + k), mm_blend16_sse2(s, d, a));
    }
    for (; k < len; k++) {
      int a = a_row[k];
      if (a == 0) continue;
      if (a == 0xff) dst_row[k] = src_row[k];
      else dst_row[k] = (src_row[k] * a + dst_row[k] * (0xff - a)) / 0xff;
    }
  }

  /* --- dst alpha plane (optional): 16-byte SSE2 compose, same two
   * fastpaths. The all-max output is just v_ff, no src read needed. */
  if (dst.alpha) {
    for (int j = ja; j < jb; j++) {
      int is = ia - x;
      int js = j - y;
      unsigned char *dst_a_row = dst.alpha + (j * dst.y_stride + ia);
      unsigned char *src_a_row = src.alpha + (js * src.y_stride + is);
      int len = ib - ia;
      int k = 0;
      for (; k + 16 <= len; k += 16) {
        __m128i sa = _mm_loadu_si128((const __m128i *)(src_a_row + k));
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(sa, v_zero)) == 0xFFFF) continue;
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(sa, v_ff)) == 0xFFFF) {
          _mm_storeu_si128((__m128i *)(dst_a_row + k), v_ff);
          continue;
        }
        __m128i da = _mm_loadu_si128((const __m128i *)(dst_a_row + k));
        _mm_storeu_si128((__m128i *)(dst_a_row + k),
                         mm_alpha_compose16_sse2(sa, da));
      }
      for (; k < len; k++) {
        int a = src_a_row[k];
        if (a == 0) continue;
        if (a == 0xff) dst_a_row[k] = 0xff;
        else dst_a_row[k] = 0xff - ((0xff - a) * (0xff - dst_a_row[k])) / 0xff;
      }
    }
  }

  /* --- U/V planes: scalar at chroma resolution, last-write-wins.
   * Row-level fastpath skips chroma rows whose two source alpha rows
   * are entirely zero (~half the chroma rows for a centred overlay
   * with a transparent margin). One SSE2 OR-reduce per source alpha
   * row replaces the per-cell scan in those rows. */
  int ci_start = ia / 2;
  int ci_end = (ib + 1) / 2;
  int cj_start = ja / 2;
  int cj_end = (jb + 1) / 2;
  for (int cj = cj_start; cj < cj_end; cj++) {
    int j_full = 2 * cj;
    {
      /* OR-reduce alpha bytes in source rows js_v1 and js_v3 within
       * [ia, ib). If the reduce is zero AND any scalar-tail bytes are
       * also zero, no chroma cell on this row will write - skip. */
      int js_v1 = j_full - y;
      int js_v3 = (j_full + 1) - y;
      int row_v1_in = (j_full >= ja) && (j_full < jb);
      int row_v3_in = (j_full + 1 >= ja) && (j_full + 1 < jb);
      __m128i acc = _mm_setzero_si128();
      int len = ib - ia;
      int k = 0;
      if (row_v1_in) {
        unsigned char *p1 = src.alpha + (js_v1 * src.y_stride + (ia - x));
        for (; k + 16 <= len; k += 16)
          acc = _mm_or_si128(acc,
                             _mm_loadu_si128((const __m128i *)(p1 + k)));
      }
      if (row_v3_in) {
        unsigned char *p3 = src.alpha + (js_v3 * src.y_stride + (ia - x));
        int k2 = 0;
        for (; k2 + 16 <= len; k2 += 16)
          acc = _mm_or_si128(acc,
                             _mm_loadu_si128((const __m128i *)(p3 + k2)));
      }
      int any_nonzero = _mm_movemask_epi8(_mm_cmpeq_epi8(acc, v_zero)) !=
                        0xFFFF;
      if (!any_nonzero) {
        int tail_any = 0;
        if (row_v1_in) {
          unsigned char *p1 = src.alpha + (js_v1 * src.y_stride + (ia - x));
          for (int kk = k; kk < len; kk++)
            if (p1[kk]) {
              tail_any = 1;
              break;
            }
        }
        if (!tail_any && row_v3_in) {
          unsigned char *p3 = src.alpha + (js_v3 * src.y_stride + (ia - x));
          for (int kk = k; kk < len; kk++)
            if (p3[kk]) {
              tail_any = 1;
              break;
            }
        }
        if (!tail_any) continue;
      }
    }
    for (int ci = ci_start; ci < ci_end; ci++) {
      int i_full = 2 * ci;
      int v1_in =
          (i_full >= ia) && (j_full >= ja) && (i_full < ib) && (j_full < jb);
      int v2_in = (i_full + 1 >= ia) && (j_full >= ja) &&
                  (i_full + 1 < ib) && (j_full < jb);
      int v3_in = (i_full >= ia) && (j_full + 1 >= ja) && (i_full < ib) &&
                  (j_full + 1 < jb);
      int v4_in = (i_full + 1 >= ia) && (j_full + 1 >= ja) &&
                  (i_full + 1 < ib) && (j_full + 1 < jb);
      int v1_is = i_full - x, v1_js = j_full - y;
      int v2_is = v1_is + 1, v2_js = v1_js;
      int v3_is = v1_is, v3_js = v1_js + 1;
      int v4_is = v1_is + 1, v4_js = v1_js + 1;
      int a1 = v1_in ? src.alpha[v1_js * src.y_stride + v1_is] : 0;
      int a2 = v2_in ? src.alpha[v2_js * src.y_stride + v2_is] : 0;
      int a3 = v3_in ? src.alpha[v3_js * src.y_stride + v3_is] : 0;
      int a4 = v4_in ? src.alpha[v4_js * src.y_stride + v4_is] : 0;

      int chosen_is = -1, chosen_js = -1;
      if (a4 == 0xff) {
        chosen_is = v4_is;
        chosen_js = v4_js;
      } else if (a3 == 0xff) {
        chosen_is = v3_is;
        chosen_js = v3_js;
      } else if (a2 == 0xff) {
        chosen_is = v2_is;
        chosen_js = v2_js;
      } else if (a1 == 0xff) {
        chosen_is = v1_is;
        chosen_js = v1_js;
      }
      if (chosen_is >= 0) {
        int src_ci = chosen_is / 2;
        int src_cj = chosen_js / 2;
        dst.u[cj * dst.uv_stride + ci] =
            src.u[src_cj * src.uv_stride + src_ci];
        dst.v[cj * dst.uv_stride + ci] =
            src.v[src_cj * src.uv_stride + src_ci];
      } else if (a1 > 0) {
        int src_ci = v1_is / 2;
        int src_cj = v1_js / 2;
        int su = src.u[src_cj * src.uv_stride + src_ci];
        int sv = src.v[src_cj * src.uv_stride + src_ci];
        int du = dst.u[cj * dst.uv_stride + ci];
        int dv = dst.v[cj * dst.uv_stride + ci];
        dst.u[cj * dst.uv_stride + ci] =
            (su * a1 + du * (0xff - a1)) / 0xff;
        dst.v[cj * dst.uv_stride + ci] =
            (sv * a1 + dv * (0xff - a1)) / 0xff;
      }
    }
  }
}
#endif

CAMLprim value caml_yuv420_add(value _src, value _x, value _y, value _dst) {
  CAMLparam4(_src, _x, _y, _dst);
  int x = Int_val(_x);
  int y = Int_val(_y);
  yuv420 src, dst;
  yuv420_of_value(&src, _src);
  yuv420_of_value(&dst, _dst);

  int ia = max(x, 0);
  int ib = min(x + src.width, dst.width);
  int ja = max(y, 0);
  int jb = min(y + src.height, dst.height);

  if (!(ia < ib))
    CAMLreturn(Val_unit);

  caml_enter_blocking_section();
#if MM_HAVE_X86_SIMD
  /* The SSE2 path only wins on the with-alpha branch; the no-alpha
   * branch is already memcpy/memset and SIMD can't improve on that. */
  if (src.alpha != NULL && mm_simd_pick() == 2)
    caml_yuv420_add_sse2(src, x, y, dst, ia, ib, ja, jb);
  else
    caml_yuv420_add_scalar(src, x, y, dst, ia, ib, ja, jb);
#else
  caml_yuv420_add_scalar(src, x, y, dst, ia, ib, ja, jb);
#endif
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv420_get_pixel_rgba(value img, value _i, value _j) {
  CAMLparam3(img, _i, _j);
  CAMLlocal1(ans);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int i = Int_val(_i);
  int j = Int_val(_j);

  int y = Y(yuv, i, j);
  int u = U(yuv, i, j);
  int v = V(yuv, i, j);
  int a = yuv.alpha ? A(yuv, i, j) : 0xff;
  int r = RofYUV(y, u, v);
  int g = GofYUV(y, u, v);
  int b = BofYUV(y, u, v);

  ans = caml_alloc_tuple(4);
  Store_field(ans, 0, Val_int(r));
  Store_field(ans, 1, Val_int(g));
  Store_field(ans, 2, Val_int(b));
  Store_field(ans, 3, Val_int(a));
  CAMLreturn(ans);
}

CAMLprim value caml_yuv420_set_pixel_rgba(value img, value _i, value _j,
                                          value c) {
  CAMLparam4(img, _i, _j, c);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int i = Int_val(_i);
  int j = Int_val(_j);
  int r = Int_val(Field(c, 0));
  int g = Int_val(Field(c, 1));
  int b = Int_val(Field(c, 2));
  int a = Int_val(Field(c, 3));

  Y(yuv, i, j) = YofRGB(r, g, b);
  U(yuv, i, j) = UofRGB(r, g, b);
  V(yuv, i, j) = VofRGB(r, g, b);
  if (yuv.alpha)
    A(yuv, i, j) = a;

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_randomize(value img) {
  CAMLparam1(img);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int i, j;

  caml_enter_blocking_section();
  for (j = 0; j < yuv.height; j++)
    for (i = 0; i < yuv.width; i++) {
      Y(yuv, i, j) = rand();
      U(yuv, i, j) = rand();
      V(yuv, i, j) = rand();
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_greyscale(value img) {
  CAMLparam1(img);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int i, j;

  caml_enter_blocking_section();
  for (j = 0; j < yuv.height; j += 2)
    for (i = 0; i < yuv.width; i += 2) {
      U(yuv, i, j) = 0x7f;
      V(yuv, i, j) = 0x7f;
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

#define PIXEL_PRECISON 0x10000
CAMLprim value caml_yuv_scale_alpha(value img, value _c) {
  CAMLparam2(img, _c);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int c = Double_val(_c) * PIXEL_PRECISON;
  int amax = CLIP(0xff * c / PIXEL_PRECISON);
  int i, j;

  caml_enter_blocking_section();
  for (j = 0; j < yuv.height; j++)
    for (i = 0; i < yuv.width; i++) {
      int a = A(yuv, i, j);
      if (a != 0) {
        if (a == 0xff)
          A(yuv, i, j) = amax;
        else
          A(yuv, i, j) = CLIP(a * c / PIXEL_PRECISON);
      }
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_disk_alpha(value img, value _x, value _y, value _r) {
  CAMLparam4(img, _x, _y, _r);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int x = Int_val(_x);
  int y = Int_val(_y);
  int radius = Int_val(_r);
  radius = radius * radius;
  int i, j;

  caml_enter_blocking_section();
  for (j = 0; j < yuv.height; j++)
    for (i = 0; i < yuv.width; i++) {
      int r = (i - x) * (i - x) + (j - y) * (j - y);
      if (r > radius)
        A(yuv, i, j) = 0;
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

#define crop(x, m) (x > m ? m : (x < 0 ? 0 : x))

CAMLprim value caml_yuv_box_alpha_native(value img, value _x, value _y,
                                         value _w, value _h, value _a) {
  CAMLparam1(img);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int x = crop(Int_val(_x), yuv.width);
  int y = crop(Int_val(_y), yuv.height);
  int w = crop(Int_val(_w), yuv.width);
  int h = max(Int_val(_h), yuv.height);
  int a = CLIP(Double_val(_a) * PIXEL_PRECISON);
  int i, j;

  caml_enter_blocking_section();
  for (j = y; j < h; j++)
    for (i = x; i < w; i++)
      A(yuv, i, j) = a;
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_box_alpha_bytecode(value *argv, int argn) {
  return caml_yuv_box_alpha_native(argv[0], argv[1], argv[2], argv[3], argv[4],
                                   argv[5]);
}

CAMLprim value caml_yuv_alpha_of_color(value img, value _y, value _u, value _v,
                                       value _d) {
  CAMLparam5(img, _y, _u, _v, _d);
  yuv420 yuv;
  yuv420_of_value(&yuv, img);
  int y = Int_val(_y);
  int u = Int_val(_u);
  int v = Int_val(_v);
  int d = Int_val(_d);
  int i, j;
  int yy, uu, vv;

  d = d * d * 3;

  caml_enter_blocking_section();
  for (j = 0; j < yuv.height; j++)
    for (i = 0; i < yuv.width; i++) {
      yy = Y(yuv, i, j) - y;
      uu = U(yuv, i, j) - u;
      vv = V(yuv, i, j) - v;
      if (yy * yy + uu * uu + vv * vv <= d)
        A(yuv, i, j) = 0;
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_rotate(value _src, value _ox, value _oy, value _angle,
                               value _dst) {
  CAMLparam5(_src, _ox, _oy, _angle, _dst);
  yuv420 src, dst;
  yuv420_of_value(&src, _src);
  yuv420_of_value(&dst, _dst);
  int ox = Int_val(_ox);
  int oy = Int_val(_oy);
  double a = Double_val(_angle);
  double sina = sin(a);
  double cosa = cos(a);
  int i, j, i2, j2;

  caml_enter_blocking_section();
  for (j = 0; j < dst.height; j++)
    for (i = 0; i < dst.width; i++) {
      i2 = (i - ox) * cosa + (j - oy) * sina + ox;
      j2 = -(i - ox) * sina + (j - oy) * cosa + oy;
      if (0 <= i2 && i2 < src.width && 0 <= j2 && j2 < src.height) {
        Y(dst, i, j) = Y(src, i2, j2);
        U(dst, i, j) = U(src, i2, j2);
        V(dst, i, j) = V(src, i2, j2);
        A(dst, i, j) = src.alpha ? A(src, i2, j2) : 0xff;
      } else
        A(dst, i, j) = 0;
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_gradient_uv(value _img, value uv, value duvx,
                                    value duvy) {
  CAMLparam4(_img, uv, duvx, duvy);
  yuv420 img;
  yuv420_of_value(&img, _img);
  int u = Int_val(Field(uv, 0));
  int v = Int_val(Field(uv, 1));
  int ux = Int_val(Field(duvx, 0)) - u;
  int vx = Int_val(Field(duvx, 1)) - v;
  int uy = Int_val(Field(duvy, 0)) - u;
  int vy = Int_val(Field(duvy, 1)) - v;

  int i, j;

  caml_enter_blocking_section();
  for (j = 0; j < img.height; j++)
    for (i = 0; i < img.width; i++) {
      Y(img, i, j) = 0xff;
      U(img, i, j) = u + ux * i / img.width + uy * j / img.height;
      V(img, i, j) = v + vx * i / img.width + vy * j / img.height;
      if (img.alpha)
        A(img, i, j) = 0xff;
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_invert(value _img) {
  CAMLparam1(_img);
  yuv420 img;
  yuv420_of_value(&img, _img);

  int i, j;

  caml_enter_blocking_section();
  for (j = 0; j < img.height; j++)
    for (i = 0; i < img.width; i++) {
      Y(img, i, j) = 0xff - Y(img, i, j);
      U(img, i, j) = 0xff - U(img, i, j);
      V(img, i, j) = 0xff - V(img, i, j);
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_sepia(value _img) {
  CAMLparam1(_img);
  yuv420 img;
  yuv420_of_value(&img, _img);

  int i, j;
  int y, u, v;
  int r, g, b;
  int c;

  caml_enter_blocking_section();
  for (j = 0; j < img.height; j++)
    for (i = 0; i < img.width; i++) {
      // TODO: avoid converting back and forth
      y = Y(img, i, j);
      u = U(img, i, j);
      v = V(img, i, j);
      r = RofYUV(y, u, v);
      g = GofYUV(y, u, v);
      b = BofYUV(y, u, v);
      c = (r + g + b) / 3;
      r = c;
      g = c * 201 / 0xff;
      b = c * 201 / 0xff;
      Y(img, i, j) = YofRGB(r, g, b);
      U(img, i, j) = UofRGB(r, g, b);
      V(img, i, j) = VofRGB(r, g, b);
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_lomo(value _img) {
  CAMLparam1(_img);
  yuv420 img;
  yuv420_of_value(&img, _img);

  int i, j;

  caml_enter_blocking_section();
  for (j = 0; j < img.height; j++)
    for (i = 0; i < img.width; i++) {
      U(img, i, j) = V(img, i, j);
      V(img, i, j) = U(img, i, j);
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_is_opaque(value _img) {
  CAMLparam1(_img);
  yuv420 img;
  yuv420_of_value(&img, _img);

  int i, j;
  int ans = 1;

  caml_enter_blocking_section();
  for (j = 0; j < img.height; j++) {
    if (!ans)
      break;
    for (i = 0; i < img.width; i++)
      if (A(img, i, j) != 0xff) {
        ans = 0;
        break;
      }
  }
  caml_leave_blocking_section();

  CAMLreturn(Val_bool(ans));
}

CAMLprim value caml_yuv_alpha_of_sameness(value _ref, value _img, value _d) {
  CAMLparam3(_ref, _img, _d);
  yuv420 ref, img;
  yuv420_of_value(&ref, _ref);
  yuv420_of_value(&img, _img);
  int d = Int_val(_d);

  int i, j;
  int y, u, v;

  d = d * d * 3;

  caml_enter_blocking_section();
  for (j = 0; j < img.height; j++)
    for (i = 0; i < img.width; i++) {
      {
        y = Y(img, i, j) - Y(ref, i, j);
        u = U(img, i, j) - U(ref, i, j);
        v = V(img, i, j) - V(ref, i, j);
        if (y * y + u * u + v * v <= d)
          A(img, i, j) = 0;
      }
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_alpha_of_diff(value _ref, value _img, value _level,
                                      value _speed) {
  CAMLparam4(_ref, _img, _level, _speed);
  yuv420 ref, img;
  yuv420_of_value(&ref, _ref);
  yuv420_of_value(&img, _img);
  int level = Int_val(_level);
  int speed = Int_val(_speed);
  int i, j;
  int y, u, v;
  int d;

  if (speed < 1)
    speed = 1;
  level = level * level * 3;

  caml_enter_blocking_section();
  for (j = 0; j < img.height; j++)
    for (i = 0; i < img.width; i++) {
      {
        y = Y(img, i, j) - Y(ref, i, j);
        u = U(img, i, j) - U(ref, i, j);
        v = V(img, i, j) - V(ref, i, j);
        d = y * y + u * u + v * v;
        if (d <= level)
          A(img, i, j) =
              A(img, i, j) * (speed * level - (level - d)) / (speed * level);
        else
          d = level - min(level, d - level);
        A(img, i, j) = 0xff - (0xff - A(img, i, j)) *
                                  (speed * level - (level - d)) /
                                  (speed * level);
      }
    }
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

CAMLprim value caml_yuv_hmirror(value _img) {
  CAMLparam1(_img);
  yuv420 img;
  yuv420_of_value(&img, _img);
  int w = img.width;
  int i, j;
  int y, u, v;

  caml_enter_blocking_section();
  for (j = 0; j < img.height; j++) {
    for (i = 0; i < w / 2; i++) {
      y = Y(img, i, j);
      Y(img, i, j) = Y(img, w - 1 - i, j);
      Y(img, w - 1 - i, j) = y;
    }
  }
  for (j = 0; j < img.height / 2; j++)
    for (i = 0; i < w / 4; i++) {
      u = U2(img, i, j);
      U2(img, i, j) = U2(img, w / 2 - 1 - i, j);
      U2(img, w / 2 - 1 - i, j) = u;
      v = V2(img, i, j);
      V2(img, i, j) = V2(img, w / 2 - 1 - i, j);
      V2(img, w / 2 - 1 - i, j) = v;
    }

  if (img.alpha)
    for (j = 0; j < img.height; j++)
      for (i = 0; i < w / 2; i++)
        A(img, i, j) = A(img, w - i, j);
  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}
