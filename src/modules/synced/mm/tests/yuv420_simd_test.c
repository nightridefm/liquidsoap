/* Bit-exact + perf harness for the SSE2 path in caml_yuv420_add.
 *
 * Pulls in src/image_yuv420.c directly (after stubbing CAML macros) so
 * the test always exercises the live code, never a stale copy. Calls
 * caml_yuv420_add_scalar() and caml_yuv420_add_sse2() side-by-side on
 * synthetic inputs and byte-compares Y/U/V/alpha output planes.
 *
 * Build / run (from the tests/ directory):
 *
 *   make -f Makefile yuv420_simd_test
 *   ./yuv420_simd_test
 *
 * Exit 0 = all correctness cases pass + perf numbers printed. Exit 1 =
 * a byte difference between scalar and SIMD on any case; the offending
 * coordinates and values are printed. The perf numbers are advisory
 * (used to track regressions during local optimisation work); CI only
 * gates on the correctness exit code.
 *
 * Why not a `dune runtest` entry: this is a C test, not OCaml. dune's
 * runtest is wired for OCaml executables; building a C target through
 * dune to invoke our scalar/SSE2 helpers would require duplicating
 * the foreign_stubs library boilerplate. A plain Makefile target keeps
 * the dependency on the OCaml toolchain at zero - any host with `cc`
 * and a C99 compiler can run it. */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* CAML macro stubs - the only parts of image_yuv420.c we exercise are
 * the static helpers (caml_yuv420_add_scalar / caml_yuv420_add_sse2);
 * the CAMLprim entry points are not called by this test. Stubs just
 * have to be syntactically valid so the file compiles. */
#define value long
#define CAMLprim
#define CAMLparam1(a)
#define CAMLparam2(a, b)
#define CAMLparam3(a, b, c)
#define CAMLparam4(a, b, c, d)
#define CAMLparam5(a, b, c, d, e)
#define CAMLlocal1(a) value a = 0
#define CAMLlocal2(a, b) value a = 0, b = 0
#define CAMLreturn(a) return (a)
#define Val_unit 0L
#define Val_int(x) ((long)(x))
#define Val_bool(x) ((long)(x))
#define Int_val(x) ((int)(x))
#define Double_val(x) ((double)(x))
#define Field(b, i) (((long *)(b))[i])
#define Is_block(x) 1
#define Store_field(b, i, v) (((long *)(b))[i] = (v))
#define caml_alloc_tuple(n) ((value)0)
#define caml_enter_blocking_section()
#define caml_leave_blocking_section()
#define Caml_ba_data_val(x) ((void *)(x))
#define String_val(x) ((char *)(x))

/* Stub CAML header files normally provided by the OCaml install. */
#define CAML_ALLOC_H
#define CAML_BIGARRAY_H
#define CAML_MEMORY_H
#define CAML_MISC_H
#define CAML_MLVALUES_H
#define CAML_THREADS_H

/* Pull the real source in. It #include's image_pixel.h / image_rgb.h /
 * image_yuv420.h transitively, which give us the `yuv420` struct and
 * Y/U/V/A macros plus our static helpers caml_yuv420_add_scalar and
 * caml_yuv420_add_sse2. The .h files lack include guards so we mustn't
 * #include them separately here. */
#include "../src/image_yuv420.c"

/* ============================================================================
 * Harness
 * ========================================================================= */

static void alloc_yuv420(yuv420 *yuv, int w, int h, int has_alpha) {
	yuv->width = w;
	yuv->height = h;
	yuv->y_stride = w;
	yuv->uv_stride = w / 2;
	yuv->y = malloc(h * w);
	yuv->u = malloc((h / 2) * (w / 2));
	yuv->v = malloc((h / 2) * (w / 2));
	yuv->alpha = has_alpha ? malloc(h * w) : NULL;
}

static void free_yuv420(yuv420 *yuv) {
	free(yuv->y);
	free(yuv->u);
	free(yuv->v);
	if (yuv->alpha) free(yuv->alpha);
}

static void rand_fill(unsigned char *p, size_t n) {
	for (size_t i = 0; i < n; i++) p[i] = rand() & 0xff;
}

static void copy_yuv(yuv420 *to, const yuv420 *from) {
	memcpy(to->y, from->y, from->height * from->y_stride);
	memcpy(to->u, from->u, (from->height / 2) * from->uv_stride);
	memcpy(to->v, from->v, (from->height / 2) * from->uv_stride);
	if (from->alpha && to->alpha)
		memcpy(to->alpha, from->alpha, from->height * from->y_stride);
}

static int plane_diff(const char *name, const unsigned char *a,
                      const unsigned char *b, int stride, int rows, int cols) {
	int diffs = 0;
	for (int j = 0; j < rows; j++)
		for (int i = 0; i < cols; i++)
			if (a[j * stride + i] != b[j * stride + i]) {
				if (diffs < 8)
					printf("  %s[%d,%d]: scalar=%d simd=%d\n", name, i, j,
					       a[j * stride + i], b[j * stride + i]);
				diffs++;
			}
	if (diffs) printf("  %s: %d diff bytes\n", name, diffs);
	return diffs;
}

/* Run scalar + SSE2 with the same inputs, return total byte differences. */
static int run_case(const char *label, int dst_w, int dst_h, int src_w,
                    int src_h, int x, int y_off, int src_has_alpha,
                    int dst_has_alpha, unsigned seed) {
	yuv420 src, dst_scalar, dst_simd, dst_init;
	alloc_yuv420(&src, src_w, src_h, src_has_alpha);
	alloc_yuv420(&dst_scalar, dst_w, dst_h, dst_has_alpha);
	alloc_yuv420(&dst_simd, dst_w, dst_h, dst_has_alpha);
	alloc_yuv420(&dst_init, dst_w, dst_h, dst_has_alpha);

	srand(seed);
	rand_fill(src.y, src_h * src.y_stride);
	rand_fill(src.u, (src_h / 2) * src.uv_stride);
	rand_fill(src.v, (src_h / 2) * src.uv_stride);
	if (src.alpha) rand_fill(src.alpha, src_h * src.y_stride);
	rand_fill(dst_init.y, dst_h * dst_init.y_stride);
	rand_fill(dst_init.u, (dst_h / 2) * dst_init.uv_stride);
	rand_fill(dst_init.v, (dst_h / 2) * dst_init.uv_stride);
	if (dst_init.alpha) rand_fill(dst_init.alpha, dst_h * dst_init.y_stride);

	copy_yuv(&dst_scalar, &dst_init);
	copy_yuv(&dst_simd, &dst_init);

	int ia = (x > 0) ? x : 0;
	int ib = (x + src_w < dst_w) ? x + src_w : dst_w;
	int ja = (y_off > 0) ? y_off : 0;
	int jb = (y_off + src_h < dst_h) ? y_off + src_h : dst_h;

	caml_yuv420_add_scalar(src, x, y_off, dst_scalar, ia, ib, ja, jb);
#if MM_HAVE_X86_SIMD
	if (src.alpha != NULL)
		caml_yuv420_add_sse2(src, x, y_off, dst_simd, ia, ib, ja, jb);
	else
		caml_yuv420_add_scalar(src, x, y_off, dst_simd, ia, ib, ja, jb);
#else
	caml_yuv420_add_scalar(src, x, y_off, dst_simd, ia, ib, ja, jb);
#endif

	printf("CASE %s: dst=%dx%d src=%dx%d offset=(%d,%d) src_alpha=%d dst_alpha=%d\n",
	       label, dst_w, dst_h, src_w, src_h, x, y_off, src_has_alpha,
	       dst_has_alpha);

	int diffs = 0;
	diffs += plane_diff("Y", dst_scalar.y, dst_simd.y, dst_scalar.y_stride,
	                    dst_h, dst_w);
	diffs += plane_diff("U", dst_scalar.u, dst_simd.u, dst_scalar.uv_stride,
	                    dst_h / 2, dst_w / 2);
	diffs += plane_diff("V", dst_scalar.v, dst_simd.v, dst_scalar.uv_stride,
	                    dst_h / 2, dst_w / 2);
	if (dst_scalar.alpha && dst_simd.alpha)
		diffs += plane_diff("A", dst_scalar.alpha, dst_simd.alpha,
		                    dst_scalar.y_stride, dst_h, dst_w);

	if (diffs == 0) printf("  PASS\n");

	free_yuv420(&src);
	free_yuv420(&dst_scalar);
	free_yuv420(&dst_simd);
	free_yuv420(&dst_init);
	return diffs;
}

/* Per-pass bench helper. Re-copies dst_init before each iter so the
 * input state is consistent across implementations. */
typedef void (*add_fn)(yuv420, int, int, yuv420, int, int, int, int);
static double bench(add_fn fn, yuv420 src, int x, int y_off, yuv420 dst_init,
                    int iters) {
	yuv420 dst;
	alloc_yuv420(&dst, dst_init.width, dst_init.height,
	             dst_init.alpha != NULL);
	int ia = (x > 0) ? x : 0;
	int ib = (x + src.width < dst.width) ? x + src.width : dst.width;
	int ja = (y_off > 0) ? y_off : 0;
	int jb = (y_off + src.height < dst.height) ? y_off + src.height
	                                            : dst.height;
	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for (int i = 0; i < iters; i++) {
		copy_yuv(&dst, &dst_init);
		fn(src, x, y_off, dst, ia, ib, ja, jb);
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);
	free_yuv420(&dst);
	return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
}

int main(void) {
	int total = 0;

	/* Aligned full overlay: src and dst same size, no offset. */
	total += run_case("full_overlay_alpha_dst_alpha", 1920, 1080, 1920, 1080,
	                  0, 0, 1, 1, 42);
	total += run_case("full_overlay_alpha_no_dst_alpha", 1920, 1080, 1920,
	                  1080, 0, 0, 1, 0, 42);
	total += run_case("full_overlay_no_alpha_no_dst_alpha", 1920, 1080, 1920,
	                  1080, 0, 0, 0, 0, 42);
	total += run_case("full_overlay_no_alpha_dst_alpha", 1920, 1080, 1920,
	                  1080, 0, 0, 0, 1, 42);

	/* Positive offset, src fits inside dst. */
	total += run_case("offset_pos", 1920, 1080, 640, 360, 100, 200, 1, 1, 7);
	/* Negative offset (src clipped on top-left). */
	total += run_case("offset_neg", 1920, 1080, 640, 360, -50, -100, 1, 1, 7);
	/* Off-right (src clipped right/bottom). */
	total += run_case("offset_offright", 1920, 1080, 640, 360, 1500, 800, 1, 1,
	                  7);
	/* Tiny src - exercises the scalar tail rather than the SIMD body. */
	total += run_case("tiny_src", 1920, 1080, 17, 11, 100, 100, 1, 1, 7);
	/* Odd width - exercises the (len % 16) tail. */
	total += run_case("odd_width", 1920, 1080, 333, 333, 200, 200, 1, 1, 7);
	/* 16-pixel-aligned offset - all SIMD body, no tail. */
	total += run_case("aligned16", 1920, 1080, 256, 256, 320, 240, 1, 1, 7);
	/* Odd offset - U/V last-write-wins logic gets exercised because
	 * the 4 chroma-cell visits read 4 different src chroma cells. */
	total += run_case("odd_offset", 1920, 1080, 640, 360, 1, 1, 1, 1, 11);
	/* Very narrow band (Y tail dominates over body). */
	total += run_case("narrow", 1920, 1080, 23, 1080, 100, 0, 1, 1, 13);

	if (total) {
		printf("\nFAILED: %d total diff bytes\n", total);
		return 1;
	}
	printf("\nALL CORRECTNESS TESTS PASS\n\n");

	/* Perf section. Numbers are advisory; not used to gate CI. */
	yuv420 src, dst_init;
	alloc_yuv420(&src, 1920, 1080, 1);
	alloc_yuv420(&dst_init, 1920, 1080, 1);
	srand(99);
	rand_fill(src.y, 1080 * 1920);
	rand_fill(src.u, 540 * 960);
	rand_fill(src.v, 540 * 960);
	rand_fill(src.alpha, 1080 * 1920);
	rand_fill(dst_init.y, 1080 * 1920);
	rand_fill(dst_init.u, 540 * 960);
	rand_fill(dst_init.v, 540 * 960);
	rand_fill(dst_init.alpha, 1080 * 1920);
	int iters = 200;

	add_fn fn_scalar = caml_yuv420_add_scalar;
#if MM_HAVE_X86_SIMD
	add_fn fn_simd = caml_yuv420_add_sse2;
#else
	add_fn fn_simd = caml_yuv420_add_scalar;
#endif

	double t_scalar = bench(fn_scalar, src, 0, 0, dst_init, iters);
	double t_simd = bench(fn_simd, src, 0, 0, dst_init, iters);
	printf("Perf @ 1920x1080 random alpha, %d iters:\n", iters);
	printf("  scalar: %.3f s  (%.3f ms/frame)\n", t_scalar,
	       t_scalar * 1000.0 / iters);
	printf("  simd:   %.3f s  (%.3f ms/frame)\n", t_simd,
	       t_simd * 1000.0 / iters);
	printf("  speedup: %.2fx\n", t_scalar / t_simd);

	/* Spatial-clusters realistic alpha: centred ellipse with thin AA
	 * ring. Mirrors a typical logo/cover overlay's mask. */
	{
		int cx = 1920 / 2, cy = 1080 / 2;
		int rx = 400, ry = 250;
		int aa_thickness = 6;
		for (int j = 0; j < 1080; j++) {
			for (int i = 0; i < 1920; i++) {
				double dx = (double)(i - cx) / rx;
				double dy = (double)(j - cy) / ry;
				double r2 = dx * dx + dy * dy;
				if (r2 < 1.0 - (double)aa_thickness / rx)
					src.alpha[j * 1920 + i] = 0xff;
				else if (r2 > 1.0 + (double)aa_thickness / rx)
					src.alpha[j * 1920 + i] = 0;
				else {
					double aa = 0.5 - (r2 - 1.0) * rx / (2.0 * aa_thickness);
					if (aa < 0) aa = 0;
					if (aa > 1) aa = 1;
					src.alpha[j * 1920 + i] = (unsigned char)(aa * 255);
				}
			}
		}
		t_scalar = bench(fn_scalar, src, 0, 0, dst_init, iters);
		t_simd = bench(fn_simd, src, 0, 0, dst_init, iters);
		printf("Perf @ 1920x1080 spatial-clusters alpha (centre ellipse + AA), %d iters:\n",
		       iters);
		printf("  scalar: %.3f s  (%.3f ms/frame)\n", t_scalar,
		       t_scalar * 1000.0 / iters);
		printf("  simd:   %.3f s  (%.3f ms/frame)\n", t_simd,
		       t_simd * 1000.0 / iters);
		printf("  speedup: %.2fx\n", t_scalar / t_simd);
	}

	/* Pathological: alpha all 0. SIMD's all-zero block fastpath
	 * collapses this to a pure alpha-stream scan. */
	memset(src.alpha, 0, (size_t)1080 * 1920);
	t_scalar = bench(fn_scalar, src, 0, 0, dst_init, iters);
	t_simd = bench(fn_simd, src, 0, 0, dst_init, iters);
	printf("Perf @ 1920x1080 all-transparent alpha, %d iters:\n", iters);
	printf("  scalar: %.3f s  (%.3f ms/frame)\n", t_scalar,
	       t_scalar * 1000.0 / iters);
	printf("  simd:   %.3f s  (%.3f ms/frame)\n", t_simd,
	       t_simd * 1000.0 / iters);
	printf("  speedup: %.2fx\n", t_scalar / t_simd);

	/* Pathological: alpha all 0xff. */
	memset(src.alpha, 0xff, (size_t)1080 * 1920);
	t_scalar = bench(fn_scalar, src, 0, 0, dst_init, iters);
	t_simd = bench(fn_simd, src, 0, 0, dst_init, iters);
	printf("Perf @ 1920x1080 all-opaque alpha, %d iters:\n", iters);
	printf("  scalar: %.3f s  (%.3f ms/frame)\n", t_scalar,
	       t_scalar * 1000.0 / iters);
	printf("  simd:   %.3f s  (%.3f ms/frame)\n", t_simd,
	       t_simd * 1000.0 / iters);
	printf("  speedup: %.2fx\n", t_scalar / t_simd);

	free_yuv420(&src);
	free_yuv420(&dst_init);
	return 0;
}
