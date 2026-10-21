---
title: "Reverse engineering NVIDIA ray tracing for NVK"
author: Axel Gneiting
date: 2026-10-21
---

# Reverse engineering NVIDIA ray tracing for NVK

Our renderer is built on ray tracing, and on the newest parts of it. It uses
ray queries in almost every pass, cluster acceleration structures for the
meshes we stream in, and large TLASes that get rebuilt every frame.

Very few people use these features as heavily as we do, and that means we
find the bugs first.

We kept hitting shader compiler bugs and other driver bugs. Each time we
followed the same routine. We shrank the repro, filed a bug, and tried random
changes to the shader until the problem went away. Then we told customers
which driver version to install.

We didn't like that routine.

We write our engine in Rust. When a frame is wrong, we want to follow it all
the way down: from the Rust code that records the command buffer, through the
Vulkan driver and the shader compiler, to the SASS the GPU runs and the bytes
of the BVH it traverses. We're power users, and we want to control the entire
stack.

NVK, the open source Vulkan driver for NVIDIA GPUs in Mesa, already gave us
most of that. It was missing ray tracing, which is the part we needed most.
So we added it.

## Why is this hard?

Ray tracing on NVIDIA hardware involves two things that NVIDIA doesn't
document:

1. **The acceleration structure format.** A BLAS is a blob of bytes that the
   driver writes and the RT core reads. To build one ourselves, we need to know
   every bit of it.
2. **The instructions.** A shader has to tell the RT core "traverse this BVH
   with this ray, and give me the next thing you hit." Those instructions
   don't appear in any public ISA document, and `nvdisasm` doesn't even have
   mnemonics for them.

Neither the Vulkan spec nor the CUDA docs cover them, and nobody upstream has
reverse engineered them before. All we had was the proprietary driver, which
does all of this correctly and lets us inspect what it does.

## Feed it triangles, read the bytes

The proprietary Vulkan driver builds a BLAS from whatever geometry you pass
it, and nothing stops you from reading the result back afterwards.

So the first tool we wrote was `blas-inspector`. It's a headless Vulkan
program that zero-fills a device-local buffer, builds a BLAS into it from a
few triangles, and copies the result to host-visible memory. Zeroing first
means any byte the driver didn't write reads as zero. We copy instead of
allocating the BLAS in host-visible memory because we didn't want to change
how the driver lays it out.

On top of that we built a small GUI: a geometry editor on the left and a hex
view on the right. When you drag a vertex, the bytes that changed light up,
and anything that looks like a plausible `f32` gets highlighted.

<!-- TODO: screenshot of blas-inspector hex view -->

Then we pointed Codex at it.

Reverse engineering a binary format means proposing a hypothesis, building a
probe for it, and checking the result, over and over. A human gets slow and
careless doing that. An LLM doesn't get bored. It can come up with
"coordinate X is stored at offset 0x24 with a shared exponent", write a script
that moves one vertex by 0.002 and rebuilds, diff the output, and reject the
hypothesis in less time than it takes you to read the hex dump. The hardware
gives you ground truth on every iteration, so wrong guesses don't survive
long.

It moved fast. On July 20th we wrote the design spec in the morning and had
the GUI running by lunch. That evening the hardware was tracing a BLAS we had
written entirely by hand.

### False starts

The fast iteration also produced confident wrong answers.

For most of that first afternoon, the vertex coordinates looked
"bit-coupled through a shared exponent". A commit message from that day says
they were "not losslessly recoverable". That turned out to be wrong. Each
vertex stores only the bits of its float that differ from an anchor vertex,
after dropping the trailing zeros they share. Once we saw that, everything
we'd blamed on a shared exponent or entangled sign bits made sense.

Later, the bounds of the root node came back as `0x4b3fffff` for a mesh whose
minimum was 0. Read as a float that's 12582911.0, which matches nothing in the
scene. It turned out the planes are padded, negated and "sign-folded" before
they're stored.

The best one was the sandbox. When the agent ran captures in a sandbox that
couldn't see the NVIDIA GPU, Vulkan silently fell back to another device. That
device produced a completely different and perfectly self-consistent BVH
layout, with 56-byte binary nodes and 48-byte triangles. Nothing in the
output said it came from the wrong GPU.

So the rule became that the hardware decides whether a field is decoded. In
one test we copied captured bytes into fresh acceleration structure storage
without running a build, changed a primitive index from 0 to 42, and traced a
ray. The hardware reported 42.

### What we found

At a high level, Blackwell (and, as it turned out, Ada) uses a **12-wide
BVH**:

| Piece | Size | What's in it |
|---|---|---|
| Internal node | 128 B | an FP32 origin, a per-axis scale, parent and child offsets, and 12 child boxes quantized to 1 byte per plane |
| Triblock | 128 B | up to 16 vertices, losslessly compressed against an anchor vertex, with triangle topology and IDs packed from the other end |
| TLAS page | 0x480 B | 16 instances stored as parallel arrays of descriptors, BLAS pointers, transforms and SBT offsets |

A few things surprised us:

- **There's no traversal stack.** Every node stores an offset to its parent,
  so the RT core can backtrack without one.
- **Triangle compression is lossless.** Dense meshes come out at roughly 15
  to 24 bytes per triangle with no loss of precision.
- **IDs are packed adaptively.** Depending on how many geometries a BLAS has,
  a 32-bit word is split differently between the primitive index and the
  geometry index. Past 65,536 geometries, the IDs move to a side table.

We also tested the format at its limits. We built BLASes with 16,777,215
primitives, put a child node exactly 4 GiB away from its parent, and swept
AABB metadata across 128 GiB of sparse memory. That last test disproved our
"addresses are 38 bits" theory. The 38-bit limit had come from two bugs in
our own code.

Everything is written up in a LaTeX spec that's detailed enough to write an
encoder from. (Ours is written from it.)

## What does the shader actually do?

A BVH the hardware can read is no use until a shader can ask the hardware to
traverse it.

We started by compiling 84 small GLSL ray query probes through NVIDIA's
offline toolchain and disassembling them. That gave us instruction bytes, but
as raw 128-bit hex with no names. The mnemonics we use (`TTUOPEN`, `TTUST`,
`TTUGO`, `TTULD`) are our own.

The offline compiler's output didn't quite match what the driver actually
runs, so we went to the source. We wrote an `LD_PRELOAD` library that scans
the proprietary driver's mmapped `/dev/nvidia*` regions for compute launches,
follows them to the shader code heaps, and extracts the exact code the GPU is
running.

A traversal step turned out to be a small fixed transaction:

```
TTUOPEN
TTUST  q[0x04]   ; origin, tmin
TTUST  q[0x05]   ; direction, tmax
TTUST  q[0x20]   ; BVH root (or saved state from the last step)
...
TTUGO            ; go
TTULD  ...       ; state to resume from
TTULD  ...       ; what we hit: t, barycentrics, triangle, instance
```

You hand the RT core a ray and either a root pointer or the state left over
from the previous step, and it runs until it has something to report. Opaque
hits are handled entirely in hardware. Your shader only hears about
non-opaque candidates, procedural boxes, and "done".

`rayQueryProceedEXT` becomes a loop over these transactions. Everything else
in the ray query API is software that decodes what came back.

### Another false start

NVK's first shaders using these instructions died with illegal instruction
faults.

The theory was that we were missing some "RT enable" bit in the launch state,
so the hardware didn't know to decode these opcodes. That was a reasonable
guess, and it was wrong. The captures showed that the proprietary driver's
launch descriptors with and without ray queries were byte-for-byte identical.
The problem was in our encoding.

The next one was harder to find. Every instruction in the transaction was
correct, every value going into it was correct, and the hardware still saw a
corrupt cull mask. The cause was the compiler. Dead code elimination had
removed a predicated-off store, and register allocation had moved the
remaining operands around. The hardware expects the sequence to be exactly
what it is, so now our compiler, NAK, keeps the whole transaction as one
opaque instruction until after register allocation and scheduling, and only
then expands it.

We also had a field labeled `mode=N` in our notes. An exhaustive sweep showed
it was a predicate selector.

## First light

The first end-to-end test didn't involve a driver-side BVH builder at all.

We got ray query shaders compiling in NAK, built BLASes **on the CPU** with
the encoder from the reverse engineering work, uploaded them into ordinary
buffers, and put their device addresses into a TLAS. The driver had no idea
those buffers were acceleration structures.

On July 23rd, three days after the first byte dump, NVK rendered a glTF scene
with ray queries.

<!-- TODO: first-light screenshot -->

Building the BLASes outside the driver meant we could test the format and the
instruction encoding separately. When a frame was wrong, we could let the
proprietary driver build the same BLAS and see whether the image changed. We
also ran it the other way round: the proprietary driver traced BLASes NVK had
built, matched our results in all ten mixed-opacity cases, and so the bug
had to be in our shader code.

## A real builder

Hand-built BLASes are fine for a demo. A driver has to build them on the GPU,
every frame, for millions of triangles.

The NVK builder plugs into Mesa's common BVH framework, which RADV and others
also use. The common code produces a binary tree. We collapse it into 12-wide
nodes following an SAH-optimal plan, encode each node with one subgroup, and
pack the triblocks in registers. Before packing, triangles that share an edge
are paired up, and very large triangles are split so their boxes are tighter.

All builds in a single `vkCmdBuildAccelerationStructuresKHR` call are batched
into one dispatch per stage. Small builds skip most of that and are built
inside a single workgroup.

This is where the CPU encoder earned its keep. It became the reference that
every GPU-built BLAS was diffed against.

## Pipelines, and passing the CTS

Once ray queries worked, full ray tracing pipelines came next.

The first version used continuations: a dispatcher loop in the raygen shader
jumping between stages. The current version is simpler and faster. Every RT
stage is a native function that is called with real `CALL` and `RET`
instructions and a fixed register ABI. Payloads of up to 48 bytes stay in
registers for the whole trace. On an RTX 4090, that took primary rays in our
cluster LOD test from **188 µs to 122 µs**.

Along the way we added indirect trace rays, `VK_KHR_ray_tracing_maintenance1`,
and `VK_KHR_ray_tracing_position_fetch`.

Then we ran the CTS. On an RTX 5090, a full `dEQP-VK.ray_query.*` run:

```
Pass:          22,580
Fail:               0
NotSupported:  26,779   (host-side builds, opacity micromaps)
```

<!-- TODO: dEQP-VK.ray_tracing_pipeline.* totals -->

NVK now passes the full Vulkan ray tracing CTS.

Some CTS passes don't mean much. `object_behind_bounding_boxes` happily
reports `Pass` when both the reference image and the result are empty,
because two empty images compare equal. We learned to look at the images and
not just the verdict.

RT shaders also put a lot of pressure on the compiler. Many of the NAK
changes on this branch came from profiling traversal-heavy shaders: spilling
around calls, spilling to shared memory instead of local memory, and better
use of uniform registers. Shared-memory spills alone took one of our test
scenes from 6.2 ms to 5.7 ms.

## Running on NVIDIA's kernel driver

NVK normally runs on nouveau, the open source kernel driver.

Our customers don't. They run NVIDIA's kernel module, and "please switch your
kernel driver" isn't something we can ask them to do.

So NVK now has a second backend. If nouveau isn't there, NVK talks directly to
NVIDIA's kernel module through its Resource Manager ioctls. It allocates its
own GPU address space and channels, submits work through usermode doorbells,
and synchronizes using RM semaphores. The proprietary userspace driver isn't
involved at all.

We copied what the proprietary driver does wherever it affected performance:

- **2 MB pages** for large allocations took our water FFT from 1.09 ms to
  0.80 ms. The proprietary driver takes 1.05 ms.
- **Compression** for images in VRAM. We tried compressing buffers too, but
  it made acceleration structures slower, so we turned that off.
- **Async copy engines** for transfers, choosing ones that don't share the
  graphics queue.

The RM interface isn't a stable ABI, but in practice it changes slowly. Right
now we support every kernel module branch from **580 to 615**. Across that
whole range we've needed exactly one ABI special case, a new field in channel
allocation that arrived in 610.

This turned out to be one of the most useful parts of the project. We ship the
driver with our application, so a driver bug fix ships to customers as part
of our next release. We no longer ask anyone to update their NVIDIA driver.

## Clusters

We stream meshes in clusters, so we also implemented
`VK_NV_cluster_acceleration_structure`.

A cluster acceleration structure is a small header, a single 12-wide leaf
node, and its triblocks. It contains no absolute pointers, so moving a
cluster to a new place in memory is a plain `memcpy`, and that's what makes
streaming cheap.

Each cluster is built by one subgroup. The subgroup Morton-sorts up to 256
triangles and runs a small dynamic program to decide how to split them across
the 12 slots. Cluster BLASes then point at cluster triblocks directly. Every
cluster operation sizes its own dispatch on the GPU, so the CPU never waits on
a readback.

## Where this leaves us

| | |
|---|---|
| First byte dump | July 20 |
| First ray query frame on NVK | July 23 |
| `blas-inspector` commits | 127 |
| Mesa commits on the branch | 139 |
| GPUs | Ada (SM89), Blackwell (SM120) |
| Kernel module branches | 580–615 |
| Ray query CTS failures | 0 |

<!-- TODO: links to the BVH spec, blas-inspector, and the Mesa branch; upstreaming plans -->

The next time a ray tracing bug costs us a day, it'll be our bug, in code we
can read, and we can fix it and ship the fix in our next release.
