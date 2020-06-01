# SODA Host Data Layout

The host must feed data with proper layout to the kernel.
This article discusses the data layout expected by the SODA kernel.

## Single-Bank Single-Tile Data Layout

With a single bank and a single tile,
  input/output data do not need to the reordered.
However, due to the architectural implementation,
  the host must feed some additional void inputs to drive the data out.
The number of additional data elements is called the *stencil distance*.
The output will be prepended the same number of void data elements,
  followed by the same number of output data elements as the input.
Note that currently the halo region of the output is invalid and needs to be
  handled by the host.

As an example, for a 3x3 blur filter on a 100x100 image without tiling,
  the stenil distance is 202.
The input array or stream to the kernel should contain 100x100+202=10202 pixels,
  and the first 10000 pixels are the input image.
The output array or stream will contain 10202 pixels as well,
  but the last 10000 pixels are the output image.
The first 202 pixels of output will be non-deterministic as they depend on
  unintialized values in BRAMs.
The following shows the input/output data layout in SODA indices (column-major).

Input data layout:

```plaintext
 /----------- 100 pixels -----------\
/                                    \
| (0,  0) | (1,  0) | ... | (99,  0) |
| (0,  1) | (1,  1) | ... | (99,  1) |
...
| (0, 99) | (1, 99) | ... | (99, 99) |
|   void  |   void  | ... |   void   |
|   void  |   void  |
```

Output data layout:

```plaintext
 /---------------------- 100 pixels ----------------------\
/                                                          \
|   void   |   void   |                     ... |   void   |
|   void   |   void   |                     ... |   void   |
|   void   |   void   | (0,  0) | (1,  0) | ... | (97,  0) |
| (98,  0) | (99,  0) | (0,  1) | (1,  1) | ... | (97,  1) |
| (98,  1) | (99,  1) | (0,  2) | (1,  2) | ... | (97,  2) |
...
| (98, 98) | (99, 98) | (0, 99) | (1, 99) | ... | (97, 99) |
| (98, 99) | (99, 99) |
```

Besides, the input and output need to be aligned to the burst width.
This means if the total length of input/output is not a multiple of burst width,
  void data elements must be padded to the end.

The stencil distance can be retrieved in the kernel code as a comment.

## Multi-Bank Single-Tile Data Layout

With multiple banks,
  the host must scatter the inputs to all banks and/or gather the outputs from
  all banks.
The input/output data are cyclic-partitioned to each bank.
Using the same example in the
  [previous section](#single-bank-single-tile-data-layout),
  the input/output data layout with 2 banks are as follows.
Note that the burst width padding is not included.

Input data layout, bank 0:

```plaintext
 /------------ 50 pixels -----------\
/                                    \
| (0,  0) | (2,  0) | ... | (98,  0) |
| (0,  1) | (2,  1) | ... | (98,  1) |
...
| (0, 99) | (2, 99) | ... | (98, 99) |
|   void  |   void  | ... |   void   |
|   void  |
```

Input data layout, bank 1:

```plaintext
 /------------ 50 pixels -----------\
/                                    \
| (1,  0) | (3,  0) | ... | (99,  0) |
| (1,  1) | (3,  1) | ... | (99,  1) |
...
| (1, 99) | (3, 99) | ... | (99, 99) |
|   void  |   void  | ... |   void   |
|   void  |
```

Output data layout, bank 0:

```plaintext
 /------------------ 50 pixels ----------------\
/                                               \
|   void   |                     ... |   void   |
|   void   |                     ... |   void   |
|   void   | (0,  0) | (2,  0) | ... | (96,  0) |
| (98,  0) | (0,  1) | (2,  1) | ... | (96,  1) |
| (98,  1) | (0,  2) | (2,  2) | ... | (96,  2) |
...
| (98, 98) | (0, 99) | (2, 99) | ... | (96, 99) |
| (98, 99) |
```

Output data layout, bank 1:

```plaintext
 /------------------ 50 pixels ----------------\
/                                               \
|   void   |                     ... |   void   |
|   void   |                     ... |   void   |
|   void   | (1,  0) | (3,  0) | ... | (97,  0) |
| (99,  0) | (1,  1) | (3,  1) | ... | (97,  1) |
| (99,  1) | (1,  2) | (3,  2) | ... | (97,  2) |
.
| (99, 98) | (1, 99) | (3, 99) | ... | (97, 99) |
| (99, 99) |
```

## Single-Bank Multi-Tile Data Layout

If host input size is larger than the kernel input size,
  the host must tile the input before sending it to the kernel.
Since the kernel does not handle halo,
  the host must replicate the input halo region.
Tiled input must be padded with void data elements to make sure all tiles have
  the same size.
Note that an output pixel may appear in multiple tiles,
  but only one tile contains the valid output in the non-halo region.
Using the same example in the
  [previous section](#single-bank-single-tile-data-layout),
  the input/output data layout for 150x150 images are as follows.

Input data layout, 150x150 input on (100, *) kernel:

```plaintext
 /------------------------------ 100 pixels ------------------------------\
/                                                                          \
| ( 0,   0) | ( 1,   0) |                                  ... | (99,   0) |
| ( 0,   1) | ( 1,   1) |                                  ... | (99,   1) |
...
| ( 0, 149) | ( 1, 149) | ...                                  | (99, 149) |
| (98,   0) | (99,   0) | ... | (149,   0) | void | void | ... |    void   |
| (98,   1) | (99,   1) | ... | (149,   1) | void | void | ... |    void   |
...
| (98, 149) | (99, 149) | ... | (149, 149) | void | void | ... |    void   |
|    void   |    void   |                                  ... |    void   |
|    void   |    void   |
```

Output data layout, 150x150 input on (100, *) kernel:

```plaintext
 /--------------------------------- 100 pixels ---------------------------------\
/                                                                                \
|    void   |    void   |                                       ... |    void    |
|    void   |    void   |                                       ... |    void    |
|    void   |    void   | ( 0,   0) |                           ... | ( 97,   0) |
| (98,   0) | (99,   0) | ( 0,   1) |                           ... | ( 97,   1) |
| (98,   1) | (99,   1) | ( 0,   2) |                           ... | ( 97,   2) |
...
| (98, 148) | (99, 148) | ( 0, 149) |                           ... | ( 97, 149) |
| (98, 149) | (99, 149) | (98,   0) | ... | (149,   0) | void | ... |    void    |
|    void   |    void   | (98,   1) | ... | (149,   1) | void | ... |    void    |
...
|    void   |    void   | (98, 148) | ... | (149, 149) | void | ... |    void    |
|    void   |    void   |
```

## Multi-Bank Multi-Tile Data Layout

Using the same example in the
  [previous section](#single-bank-single-tile-data-layout),
  the input/output data layout for 150x150 images are as follows.

Input data layout, 150x150 input on (100, *) kernel, bank 0:

```plaintext
 /------------------------------- 50 pixels -------------------------------\
/                                                                           \
| ( 0,   0) | (  2,   0) |                                  ... | (98,   0) |
| ( 0,   1) | (  2,   1) |                                  ... | (98,   1) |
...
| ( 0, 149) | (  2, 149) | ...                                  | (98, 149) |
| (98,   0) | (100,   0) | ... | (148,   0) | void | void | ... |    void   |
| (98,   1) | (100,   1) | ... | (148,   1) | void | void | ... |    void   |
...
| (98, 149) | (100, 149) | ... | (148, 149) | void | void | ... |    void   |
|    void   |    void    |                                  ... |    void   |
|    void   |
```

Input data layout, 150x150 input on (100, *) kernel, bank 1:

```plaintext
 /------------------------------- 50 pixels -------------------------------\
/                                                                           \
| ( 1,   0) | (  3,   0) |                                  ... | (99,   0) |
| ( 1,   1) | (  3,   1) |                                  ... | (99,   1) |
...
| ( 1, 149) | (  3, 149) | ...                                  | (99, 149) |
| (99,   0) | (101,   0) | ... | (149,   0) | void | void | ... |    void   |
| (99,   1) | (101,   1) | ... | (149,   1) | void | void | ... |    void   |
...
| (99, 149) | (101, 149) | ... | (149, 149) | void | void | ... |    void   |
|    void   |    void    |                                  ... |    void   |
|    void   |
```

Output data layout, 150x150 input on (100, *) kernel, bank 0:

```plaintext
 /---------------------------------- 50 pixels ----------------------------------\
/                                                                                 \
|    void   |    void   |                                        ... |    void    |
|    void   |    void   |                                        ... |    void    |
|    void   | ( 0,   0) | (  2,   0) |                           ... | ( 96,   0) |
| (98,   0) | ( 0,   1) | (  2,   1) |                           ... | ( 96,   1) |
| (98,   1) | ( 0,   2) | (  2,   2) |                           ... | ( 96,   2) |
...
| (98, 148) | ( 0, 149) | (  2, 149) |                           ... | ( 96, 149) |
| (98, 149) | (98,   0) | (100,   0) | ... | (148,   0) | void | ... |    void    |
|    void   | (98,   1) | (100,   1) | ... | (148,   1) | void | ... |    void    |
...
|    void   | (98, 148) | (100, 148) | ... | (148, 149) | void | ... |    void    |
|    void   |
```

Output data layout, 150x150 input on (100, *) kernel, bank 1:

```plaintext
 /---------------------------------- 50 pixels ----------------------------------\
/                                                                                 \
|    void   |    void   |                                        ... |    void    |
|    void   |    void   |                                        ... |    void    |
|    void   | ( 1,   0) | (  3,   0) |                           ... | ( 97,   0) |
| (99,   0) | ( 1,   1) | (  3,   1) |                           ... | ( 97,   1) |
| (99,   1) | ( 1,   2) | (  3,   2) |                           ... | ( 97,   2) |
...
| (99, 148) | ( 1, 149) | (  3, 149) |                           ... | ( 97, 149) |
| (99, 149) | (99,   0) | (101,   0) | ... | (149,   0) | void | ... |    void    |
|    void   | (99,   1) | (101,   1) | ... | (149,   1) | void | ... |    void    |
...
|    void   | (99, 148) | (101, 148) | ... | (149, 149) | void | ... |    void    |
|    void   |
```
