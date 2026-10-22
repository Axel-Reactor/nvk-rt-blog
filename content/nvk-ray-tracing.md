+++
title = "Reverse engineering NVIDIA's ray tracing hardware"
date = 2026-10-21
authors = ["Axel Gneiting"]
+++

NVIDIA makes the best ray tracing GPUs, and we have no plans to switch. We
do, however, keep running into problems with NVIDIA's driver. Most of them
are our own fault, in the sense that our renderer leans on features almost
nobody else uses yet: descriptor heaps, cluster acceleration structures,
ray queries in nearly every pass, and heavy use of subgroup operations in
divergent code. That is exactly where the shader compiler bugs and the
driver bugs live, and when you can't see what the compiler did to your
shader, debugging means changing the code until the problem goes away and
hoping it stays gone.

Getting a fix was its own problem. NVIDIA's Vulkan driver only works with its
matching kernel driver, so every fix meant asking our customers to update
their entire NVIDIA driver. Some of them did.

We are power users, and we want to be able to debug everything. Our engine is
written in Rust, and when a frame comes out wrong we want to follow it all
the way down: through the Vulkan driver, the shader compiler and the
acceleration structure builder, to the SASS the GPU actually runs. The only
way to get that is to own the whole stack.

The open source option is NVK, the Vulkan driver for NVIDIA GPUs in Mesa. It
was already good enough for most of what we do, with one gap: no ray tracing.
So we added it. Today NVK has ray queries, ray tracing pipelines, a GPU BVH
builder and `VK_NV_cluster_acceleration_structure` on both Ada and Blackwell,
and it passes the Vulkan ray tracing CTS. It also runs on NVIDIA's own kernel
driver, which turned out to matter just as much as the ray tracing.

None of this is documented by NVIDIA. The rest of this post is about how we
figured it out.

## The acceleration structure format

We had always wondered what makes NVIDIA's acceleration structures so good.
They trace fast, and they're small. When Arseny Kapoulkine
[measured acceleration structure sizes](https://zeux.io/2025/03/31/measuring-acceleration-structures/)
across vendors, NVIDIA's BLASes came out roughly 2.5 times smaller than
AMD's and Intel's for the same scene. AMD and Intel document their formats
well enough that he could explain where their bytes go. For NVIDIA he
couldn't: "pretty much nothing is known about NVidia, but they clearly have a
significant competitive advantage here so maybe they have something to hide."

To trace rays without NVIDIA's driver, we had to find out what that
something was. Fortunately the proprietary driver will happily build an
acceleration structure for any geometry you hand it, and nothing stops you
from reading the bytes back afterwards.

So on the morning of July 20th I wrote a spec for a small tool that does
exactly that. It builds a BLAS from a handful of triangles, copies the result
back to the CPU, and shows the bytes next to the geometry. Move a vertex and
you can see which bytes change. By 11:15 it existed.

Then I pointed Codex at it.

Figuring out a binary format is a loop: guess what a field means, build some
geometry that should change it, look at the bytes, and throw the guess away
when it's wrong. Codex runs that loop much faster than a person and never
gets tired of it. The hardware is the judge. A theory only counts once we've
written our own bytes based on it and the RT core has traced them correctly.

By 20:33 that evening, the GPU was tracing a BLAS written entirely by hand.
The tool's repository ended up with 127 commits, nearly all of them probes.

### False starts

It was fast, and for a good part of the day it was confidently wrong.

At 12:42 the commit log says the second and third vertex of each triangle
are "stored in a nonlinear shared-exponent compressed block and are not
losslessly recoverable." At 13:23 it has found that "v1x = negated full f32,
bit-coupled pack." Both commits are precise, well argued, and wrong. The real
answer came in the afternoon: each vertex only stores the low bits of its
float that differ from a reference vertex, after dropping the trailing zero
bits they all share. Once it saw that, the notes say, it "explains all
behavior previously attributed to a shared exponent or entangled sign bits."

The best one happened inside the sandbox. Some of the probe runs happened in
a sandbox that couldn't see the NVIDIA GPU, and Vulkan silently fell back to
a different device. That device produced its own perfectly self-consistent
BVH layout, with 56-byte binary nodes and 48-byte triangles, and the decoder
dutifully started decoding it. The harness now refuses any device whose
vendor ID isn't `0x10de`.

### What NVIDIA is hiding

Mostly very careful compression. The BVH is 12 wide, so there are few nodes,
and each node stores its twelve child boxes quantized down to one byte per
plane. Triangles are packed into 128-byte blocks, and inside a block each
vertex only stores the low bits that differ from the reference vertex. The
compression is lossless, so you get back exactly the floats you put in. On
dense meshes this comes out at around 15 to 25 bytes per triangle, in line
with what Arseny measured on Blackwell.

Then we tried to break it. We built BLASes and TLASes with 16,777,215
primitives. We placed a child node exactly 4 GiB away from its parent
to see whether the hardware would follow a 32-bit link that far (it did). We
swept procedural-geometry metadata across 128 GiB of sparse memory, which
killed a theory that addresses were 38 bits wide. The 38-bit limit turned
out to be two bugs in our own code.

All of it is written down in a spec detailed enough to build an encoder
from.

## The ray tracing instructions

Knowing the format only gets you halfway. A shader still has to tell the RT
core to traverse it, and the instructions for that aren't documented either.
NVIDIA's own disassembler prints them as raw 128-bit hex without mnemonics.

We started by compiling 84 small ray query shaders with NVIDIA's tools to
get a first look at the code. That doesn't tell you what the driver really
submits, though, so we captured that directly: a small `LD_PRELOAD` library
scans the command streams the proprietary driver writes into its mmapped
buffers, follows them to the shader code, and dumps it.

What came out is a fixed 17-instruction transaction. For a while the
instructions had made-up names in our notes. The real ones turned up in an
unexpected place: NVIDIA's `nvlink` binary carries its opcode tables with
the mnemonics ROT13-encoded, and someone had already
[decoded them](https://gh.evko.io/crucible-notes/nvlink/reference/rot13-passes.html).
`GGHBCRA` is `TTUOPEN`, `GGHFG` is `TTUST`, and so on. So the names below are
NVIDIA's own:

```
TTUOPEN
TTUMACROFUSE 15
TTUST  q[0x00], control
TTUST  q[0x04], origin, tmin
TTUST  q[0x05], direction, tmax
TTUST  q[0x08], control
TTUST  q[0x10], continuation      ; state from the last step ...
TTUST  q[0x18], continuation
TTUST  q[0x19], continuation
TTUST  q[0x20], root              ; ... or the BVH root
TTUGO
TTULD  q[0x38]                    ; where to resume
TTULD  q[0x3c]
TTULD  q[0x3d]
TTULD  q[0x32]                    ; what we hit: instance, SBT offset
TTULD  q[0x31]                    ; triangle block address
TTULD.CLOSE q[0x30]               ; t, barycentrics, triangle
```

The shader writes the ray and either the BVH root or the state left over
from the previous step into numbered slots, starts the unit, and reads back
either a hit candidate that needs a decision from the shader, or a signal
that traversal is done. Opaque hits never reach the shader at all; the
hardware handles them on its own. `rayQueryProceedEXT` becomes a loop around
that sequence, and the rest of the ray query API is ordinary shader code
that decodes the result.

### More false starts

The first NVK shaders that used these instructions died with illegal
instruction faults. The working theory, from the notes on July 23rd, was
that the fault comes from "per-channel/SM state (RT decode enable in
QMD/channel methods) the proprietary driver programs." Some enable bit we
hadn't set. That's a sensible theory, and it survived until we captured the
proprietary driver's launch descriptors with and without ray queries:

> The ray-query and no-ray-query renderer QMDs are identical.

The bug was in our encoding.

The next one was harder to see. Every instruction in the transaction was
correct, and every value going into it was correct. The RT core still got
garbage. The compiler had done what compilers do: it removed a store it
considered dead and hoisted another one, so that the transaction, as the
notes put it, "received a corrupt cull-mask/control pair even though every
individual instruction and computed value was valid." NAK, NVK's shader
compiler, now treats the whole transaction as a single instruction until
after register allocation and scheduling, and only then expands it.

There was also a field we'd labeled `mode=N` for a few days. An exhaustive
sweep showed it to be a predicate selector. The notes now describe the old
labels as "misnomers", which is putting it politely.

## From the first ray to the CTS

The first version had no BVH builder in the driver at all. We got ray query
shaders compiling, built the BLASes on the CPU with our own encoder,
uploaded them into normal buffers and wrote their device addresses into a
TLAS. The driver never needed to know they were acceleration structures.

That was enough to render a ray traced scene with NVK on July 23rd, three
days after we started looking at the format. Look closely and you can still
see the artifacts: a few triangles are missing, and some edges have
speckles where the shader decoded a hit wrong. But it's a shaderball, lit by
ray queries, running on an open source driver.

![The first ray traced image out of NVK on July 23rd: a shaderball lit with ray queries, traced against a BLAS built on the CPU by our own encoder, rendering artifacts included](/images/nvk-first-light-shaderball.png)

It also kept the problem small. Whenever an image was wrong, we could have
the proprietary driver build the same BLAS and see whether that fixed it,
which told us immediately whether the bug was in our BVH or in the shader.
It worked the other way round too: when mixed-opacity scenes came out wrong,
the proprietary driver traced NVK-built BLASes and agreed with our expected
results in all ten test cases, which put the bug firmly on the shader side.

After that we wrote a real BVH builder that runs on the GPU. It sits on top
of Mesa's common BVH code, which builds a binary tree; the NVK part collapses
that into 12-wide nodes and encodes the compressed triangle blocks. Once the
driver could build acceleration structures itself, existing Vulkan software
started working unmodified. Sascha Willems' ray query sample was the first:

![Sascha Willems' ray traced shadows sample running on NVK on an RTX 5090 laptop GPU](/images/nvk-sascha-willems-ray-queries.png)

Ray tracing pipelines followed, with each shader stage compiled as a real
function that the other stages call directly. Our mesh streaming works in
clusters, so we also implemented `VK_NV_cluster_acceleration_structure`.

Ray tracing pipelines are what most games use, which meant we could test
with the classic, Quake II RTX, path traced on an open source driver:

![Quake II RTX running on NVK with ray tracing pipelines](/images/nvk-quake2-rtx.png)

Then we ran the Vulkan CTS until there was nothing left to fix. The full ray
query suite on an RTX 5090:

```
dEQP-VK.ray_query.*       CTS 1.4.7.0

Pass           22,580
Fail                0
NotSupported   26,779   (host-side builds, opacity micromaps)
```

<!-- TODO: dEQP-VK.ray_tracing_pipeline.* totals -->

NVK now passes the full ray tracing CTS.

Passing the CTS is one thing. The real test was our own renderer, Hydrogen,
which uses every one of the extensions this post is about at once. Here it
is running on NVK for the first time, on an RTX 4090:

![Zeromatter's Hydrogen renderer running on NVK on an RTX 4090](/images/nvk-hydrogen-first-run.png)

## Running on NVIDIA's kernel driver

Fixing bugs ourselves only helps if the fixes reach our customers without
anyone having to change their setup. NVK normally runs on nouveau, the open
source kernel driver, and for us that was never an option. Nouveau doesn't
properly support memory compression, which costs a lot of performance, and
it can't run CUDA, which we and our customers need.

So we made NVK run on top of NVIDIA's proprietary kernel module instead. NVK
talks to the kernel module directly and uses none of NVIDIA's userspace
driver. The kernel interface isn't officially stable, but it changes slowly.
We currently support every kernel driver release from 580 to 615, and
across that whole range we've needed exactly one special case: a single new
field in channel allocation that arrived in 610.

A nice side effect is that both drivers can run on the same machine at the
same time, on the same kernel module. Here is the same Vulkan sample running
on NVIDIA's driver on the left and on NVK on the right. Performance was far
from final at that point; the first version didn't yet use large pages or
memory compression, and most of the compiler work hadn't happened.

![The Vulkan instancing sample running side by side on NVIDIA's proprietary driver and on NVK, both on the same RTX 4090 and the same NVIDIA kernel module](/images/nvk-nvrm-side-by-side.png)

That decouples us from our customers' driver updates. We ship NVK with our
application, so a driver fix now goes out with our next release and works
with whatever NVIDIA driver is already installed.

## What's left to do

We're not quite at NVIDIA's performance yet. The acceleration structures and
the traversal itself are on par, since the hardware does the same work either
way. The gap is in the shader compiler. NVIDIA's compiler has had years of
tuning on exactly the kind of long, register-hungry shaders that ray tracing
produces, and NAK is still catching up on register allocation, spilling and
scheduling around the traversal calls. This is where most of our effort goes
now, and the gap has been closing with every round of compiler work.

The other big item is upstreaming. All of this currently lives on a branch
on top of Mesa, and we'd like it to live in Mesa itself. That's going to take
a while. The branch is 139 commits and over 50,000 lines, touching NAK, NVK
and the common Vulkan BVH code, and every one of those commits has to go
through Mesa's review process.

## Where we are now

When something in ray tracing breaks today, we can look at every layer of it
ourselves, from our Rust code down to the bits in the BVH, and fix the
problem where it actually is.

This is what that looks like in practice. Our shaders are written in Rust,
and with NVK we can see the SASS for each line, with the Rust source
interleaved: the texture fetch, the FFMAs the gradient turned into, and the
reconvergence point of that `if`, each one next to the line that produced
it.

![SASS output from NAK for one of our Rust shaders, with the original Rust source interleaved line by line](/images/nvk-rust-to-sass.png)

<!-- TODO: links -->
