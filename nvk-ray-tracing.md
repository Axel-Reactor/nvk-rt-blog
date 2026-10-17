---
title: "Reverse engineering NVIDIA ray tracing for NVK"
author: Axel Gneiting
date: 2026-10-17
---

# Reverse engineering NVIDIA ray tracing for NVK

Our renderer uses ray tracing heavily, and the features it relies on are new
ones: ray queries everywhere, cluster acceleration structures for streamed
meshes, and large dynamic TLASes rebuilt every frame. Working that close to
the edge, we kept running into bugs in NVIDIA's proprietary driver. Some were
shader compiler bugs that miscompiled our traversal loops, and others were
driver bugs we could only work around by guessing. The proprietary stack gave
us no way to find out what was actually happening.

We don't like guessing. When a frame is wrong, we want to step through every
layer between our Rust code and the SASS the GPU runs: the Vulkan driver, the
shader compiler, the acceleration structure builder, the command stream, and
the bytes in the BVH. We are power users, and for us that means owning the
whole stack so we can debug and fix any part of it.

NVK, the open source Vulkan driver for NVIDIA GPUs in Mesa, already gave us
most of that, but it had no ray tracing. The data formats and instructions
behind NVIDIA's RT cores are undocumented. This post describes how we
reverse engineered them and what we built on top: ray queries, ray tracing
pipelines, a GPU BVH builder, `VK_NV_cluster_acceleration_structure`, and a
way to run NVK on NVIDIA's proprietary kernel module.

## Decoding the BLAS format with an LLM

A bottom-level acceleration structure (BLAS) is a blob of bytes that the
driver writes and the RT core reads. NVIDIA doesn't document its layout, but
the proprietary Vulkan driver will build one from any geometry you hand it,
and nothing stops you from reading the result back.

We built a small tool, `blas-inspector`, for exactly that. It is a headless
Vulkan program that:

1. zero-fills a device-local buffer,
2. builds a BLAS into it from a few triangles,
3. copies the buffer into host-visible memory with `vkCmdCopyBuffer`.

The copy matters. Allocating the BLAS in host-visible memory might change how
the driver lays it out, so we read back the real device-local bytes instead.
Zeroing first means anything the driver didn't write reads as zero, so diffs
between two builds stay clean.

On top of that we added a GUI with a geometry editor on one side and a hex
view on the other. You drag a vertex and watch which bytes change. Bytes that
look like plausible `f32` values get highlighted. Later the tool grew a
decoded view that draws the hierarchy back out of the bytes, so you can see
our decoder's idea of the structure next to the input.

Most of the decoding was done with an LLM, Codex. The loop went like this:
put a tiny, controlled piece of geometry into a real build, dump the bytes,
and have the model form a hypothesis about the layout. Then write a script
that nudges one coordinate, rebuilds, and checks which bits moved, and repeat.
This kind of work suits an LLM well: it is tedious and depends on hypotheses,
the model can generate and run probes far faster than a person can, and the
hardware gives you ground truth on every attempt.

The workflow was fast. The design spec for the inspector was written on the
morning of July 20th, and the GUI was working by lunchtime. By that evening
we had decoded the vertex compression and written a fully synthetic BLAS
that the hardware traced correctly.

The method also produced wrong theories. For several hours the vertex
coordinates looked "bit-coupled through a shared exponent", nonlinearly
entangled and possibly not losslessly recoverable. The real answer was
simpler: each vertex stores only the suffix of its raw float bits that differs
from an anchor vertex, after dropping shared trailing zero bits. That one
finding explained everything we had blamed on a shared exponent or entangled
sign bits.

Every decoded field was checked on the hardware. For example, we copied
captured bytes into fresh acceleration structure storage without running a
build, patched a primitive index from 0 to 42, traced a ray, and checked that
the hardware reported 42.

### What the format looks like

The structure on Blackwell (SM120) is a **12-wide BVH**. Ada (SM89) turned out
to use the same format.

- **Nodes** are 128 bytes: a 32-byte header plus twelve 8-byte child slots.
  - The header holds:
    - a type byte,
    - an FP32 origin,
    - three 8-bit per-axis scale exponents,
    - signed deltas to the parent and to the first child.
  - Internal children are stored next to each other, so child *i* is
    `parent + delta + i`.
  - Every node knows its parent, so traversal can backtrack without a stack.
    There is no traversal stack buffer anywhere.
- **Child boxes** are quantized to one byte per plane, against a step of
  `2^(e-135)`, and rounded outward.
  - Byte 6 of a slot is `0xff` for an internal child, or the instance mask for
    a TLAS leaf.
  - Byte 7 is a leaf descriptor: a non-opaque bit, a triangle end index, and a
    "block advance" that walks a cursor across the leaf's triangle storage.
- **Triangles** live in 128-byte *triblocks* of up to 16 vertices.
  - The first vertex is stored as raw FP32.
  - The other vertices are a variable-width bitstream of differing suffixes,
    which grows upward from the start of the block.
  - Topology and per-triangle metadata grow downward from the end. Triangle 0
    is implicitly `(0,1,2)`, and every other triangle is a 14-bit record of
    three 4-bit local indices plus two flag bits.
  - The last word is a codec descriptor that gives the bit widths.
  - The compression is lossless, and dense meshes come out at around
    15–24 bytes per triangle.
- **Primitive and geometry indices** have several encodings:
  - a direct encoding, in which a selector chooses how many bits go to the
    primitive and how many to the geometry;
  - an indirect encoding through a side table, which is needed past 65,536
    geometries and for updatable builds.
- **The root** has a separate 32-byte record with full-precision bounds. The
  planes are padded, negated and "sign-folded", so a BLAS whose minimum is 0
  stores it as `0x4b3fffff`. Read as a float, that is 12582911.0, which
  confused us for a while.
- **TLAS instances** live in 0x480-byte pages of 16 instances. Each page holds
  parallel arrays of descriptors, BLAS addresses, transforms and SBT/custom
  index pairs, plus a 64-byte leaf per instance that stores the inverse
  transform.

We also checked the format at its limits. We built BLASes and TLASes with
16,777,215 primitives, placed a child node exactly 4 GiB away from its parent,
and swept AABB metadata addresses across 128 GiB of sparse memory. That last
sweep disproved an earlier "38-bit address" theory, which turned out to be
caused by two bugs in our own code. All of this is written up in a LaTeX
specification that is detailed enough to write an encoder from.

## Reverse engineering the RT core instructions

Knowing the data format was only half of the problem. We still needed to know
how a shader asks the RT core to traverse it.

We started by compiling a set of small GLSL ray query probes, 84 in the end,
through NVIDIA's offline toolchain and disassembling the result. That got us
the instruction bytes, but `nvdisasm` prints the RT instructions as raw
128-bit hex with no mnemonics. The names we use (`TTUOPEN`, `TTUST`, `TTUGO`,
`TTULD`, `TTUMACROFUSE`) are our own.

The offline compiler's output didn't fully match what the driver really
submits, so we went to the source. We wrote an `LD_PRELOAD` library that scans
the proprietary driver's mmapped `/dev/nvidia*` regions for compute launch
methods, follows the QMDs to the shader code heaps, and pulls out the exact
code the GPU runs. This also killed one of our theories. NVK shaders using the
new instructions were faulting with illegal instruction errors, and we had
assumed some RT enable bit was missing from the launch state. The captures
showed that QMDs with and without ray queries were byte-for-byte identical.
The bug was in our instruction encoding.

A traversal step turned out to be a fixed transaction:

```
TTUOPEN
TTUMACROFUSE n
TTUST   q[0x00]    ; control words (cull mask, flags, opacity)
TTUST   q[0x04]    ; origin, tmin
TTUST   q[0x05]    ; direction, tmax
TTUST   q[0x08]    ; more control
TTUST   q[0x10]    ; continuation state ...
TTUST   q[0x18]    ; ... or the root pointer,
TTUST   q[0x19]    ; ... selected per lane by predicates
TTUST   q[0x20]
TTUGO
TTULD   q[0x38], q[0x3c], q[0x3d]   ; continuation state for the next step
TTULD   q[0x30], q[0x31], q[0x32]   ; hit: barycentrics, t, triangle ref, SBT, instance
TTULD.CLOSE
```

The shader writes the ray and either the root or the saved continuation state
into numbered slots, and starts traversal. The hardware then runs until it has
something to report: a non-opaque candidate, a procedural AABB, or "done".
Opaque hits are committed inside the RT core. The returned continuation state
goes back into the next transaction, so `rayQueryProceedEXT` turns into a
loop over transactions, and everything else in the ray query API is software
that decodes the returned payload.

Two details took a while to find:

- A nibble we had labeled `mode=N` was actually a **predicate selector**. An
  exhaustive sweep of its values confirmed this.
- The transaction must stay **atomic through the compiler**. Our first version
  emitted it as separate instructions. Dead code elimination removed a
  predicated-off `TTUST`, and register allocation reshuffled operands. Each
  instruction was still valid on its own, but the hardware received a corrupt
  cull mask. NAK now models the whole sequence as one virtual
  `OpTtuOpBundle` instruction with 22 sources and 12 destinations, and only
  expands it into real instructions after register allocation and scheduling.

## First light: ray queries on hand-built BLASes

The first end-to-end test didn't use a driver-side builder at all. We got ray
query shaders compiling in NAK, built BLASes **on the CPU** with the encoder
from the reverse engineering work, uploaded them to plain buffers, and wrote
their device addresses into a TLAS. On July 23rd, three days after the first
byte dump, NVK rendered a glTF scene with ray queries.

This approach let us test the BVH format and the instruction encoding
separately. When an image was wrong, we could rebuild the same BLAS with the
proprietary driver and compare. Later, to separate a builder bug from a shader
bug in mixed-opacity scenes, we did the reverse: the proprietary driver traced
our NVK-built bytes and got identical results in all ten test cases, which
pinned the bug on the shader side.

## A GPU BVH builder

CPU-built BLASes are fine for a test, but a real driver has to build
acceleration structures on the GPU, every frame, for millions of triangles.

The NVK builder plugs into Mesa's common `vk_bvh` framework:

1. The common LBVH/PLOC builders produce a binary intermediate tree.
2. We **collapse** that tree top-down into 12-wide nodes, following an
   SAH-optimal collapse plan that the common builders now emit.
3. Each node is encoded by one subgroup, and triblocks are packed in
   registers.
4. Triangles that share an edge are paired before packing, and very large
   triangles are split at coarse Morton planes to tighten their boxes.

All builds in a `vkCmdBuildAccelerationStructuresKHR` call are batched into a
single dispatch per stage over a job table. The GPU can write that table
itself, which also gives us indirect builds. Small builds skip the full
pipeline and are built inside a single workgroup.

## Ray tracing pipelines and passing the CTS

Once ray queries worked, pipelines followed. The first version used a
continuation-style dispatcher. The current one makes each RT stage a native
function: raygen, closest-hit, miss, any-hit and intersection shaders call
each other with `CALL.ABS`/`RET.ABS` and a fixed register ABI, with arguments
starting at R10. Payloads of up to 48 bytes are passed entirely in registers.
On an RTX 4090 that took primary rays in our cluster LOD test from 188 µs to
122 µs. The traversal shader is built on the same TTU transaction as ray
queries. We also implemented indirect trace rays,
`VK_KHR_ray_tracing_maintenance1` and `VK_KHR_ray_tracing_position_fetch`.

The result is that NVK now passes the full Vulkan ray tracing CTS. For ray
queries on an RTX 5090, a full `dEQP-VK.ray_query.*` run on CTS 1.4.7.0 gives
**22,580 pass, 0 fail**. The 26,779 cases reported as NotSupported cover
features we don't expose: host-side acceleration structure builds and opacity
micromaps.

<!-- TODO: add dEQP-VK.ray_tracing_pipeline.* totals -->

Getting there also meant a lot of compiler work, because RT shaders are hard
on a compiler. Register allocation and spilling across calls, spilling to
shared memory, uniform register propagation, and modelling the TTU's
input/output allocation phases all came out of profiling traversal-heavy
shaders.

## NVK on the proprietary kernel driver

NVK normally runs on nouveau, the open source kernel driver. Many machines,
including our customers', run NVIDIA's kernel module instead, and asking
people to change kernel drivers to use our renderer isn't realistic.

So we added a second backend. When nouveau isn't available, NVK talks to the
NVIDIA kernel module directly through its Resource Manager (RM) ioctl
interface. It allocates a client, device and subdevice, a GPU virtual address
space, a channel group and a GPFIFO channel with usermode doorbells, and
synchronizes using RM semaphore surfaces. `nvidia-drm` handles dma-buf and
sync_file interop.

We matched the proprietary driver's memory behavior where it mattered for
performance:

- **2 MB pages** for large VRAM allocations. This took our water FFT from
  1.09 ms to 0.80 ms. The proprietary driver takes 1.05 ms.
- **Compression** for images in VRAM.
- **Asynchronous copy engines** for transfers, chosen so they don't share the
  graphics runlist.

The RM ABI isn't stable, but in practice it changes slowly. The backend
currently supports **every kernel module branch from 580 to 615**. Across that
range the only ABI difference we have needed to handle is one new field in
the channel allocation parameters in 610.

This helps our customers directly. We ship NVK with our application, so we no
longer have to tell anyone to update their NVIDIA driver to pick up a fix: as
long as the installed kernel module is in that range, they get our userspace
driver with our fixes.

## Cluster acceleration structures

Our renderer streams meshes at cluster granularity, so we also implemented
`VK_NV_cluster_acceleration_structure`. A cluster acceleration structure
(CLAS) has three parts:

- a small header,
- a single leaf-only 12-wide node,
- its triblocks.

A CLAS contains no absolute addresses, so moving one in memory is a plain
copy. That is what makes streaming cheap. Each cluster is built by a single
subgroup, which Morton-sorts up to 256 triangles and runs a small dynamic
program to choose the slot ranges with the lowest SAH cost. Cluster BLASes
then reference cluster triblocks by address. All cluster operations size
their dispatches on the GPU, so the CPU never has to read back counts.

## What's next

<!-- TODO: upstreaming plans, links to the BVH spec, blas-inspector, and the Mesa branch -->

Ray tracing is no longer a black box for us. When something breaks, we can
follow it from the Rust code that records the command buffer down to the bits
of a triblock, and fix it at whichever layer it's in.
