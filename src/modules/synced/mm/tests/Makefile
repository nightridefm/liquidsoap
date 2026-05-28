all:

test:
	dune exec ./bug22.exe

# C-level bit-exact + perf harness for the SSE2 path in
# src/image_yuv420.c (caml_yuv420_add). Standalone, no OCaml deps - any
# host with `cc` and a C99 compiler can run it. The test #include's the
# real source file (after stubbing CAML macros) so it always exercises
# the live algorithm, never a stale copy. _caml_stubs/ provides empty
# caml/*.h headers so the include-chain resolves without an opam
# install. Exit 0 on pass; exit 1 on any byte-level divergence between
# scalar and SIMD on any test case.
yuv420_simd_test: yuv420_simd_test.c ../src/image_yuv420.c ../src/image_yuv420.h ../src/image_pixel.h ../src/image_rgb.h
	cc -O3 -Wall -Wno-unused-function -Wno-array-bounds -I_caml_stubs -o $@ yuv420_simd_test.c -lm

test-simd: yuv420_simd_test
	./yuv420_simd_test

clean:
	rm -f yuv420_simd_test
