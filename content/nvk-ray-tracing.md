+++
title = "Reverse engineering NVIDIA's ray tracing hardware"
date = 2026-10-21
authors = ["Axel Gneiting"]
+++

NVIDIA makes the best ray tracing GPUs, and we have no plans to switch. We
do, however, keep running into problems with NVIDIA's driver. Most of them
are our own fault, in the sense that our renderer leans on features almost
nobody else uses yet: descriptor heaps, cluster acceleration structures, and
ray queries in nearly every pass. That is exactly where the shader compiler
bugs and the driver bugs live, and when you can't see what the compiler did
to your shader, debugging means changing the code until the problem goes
away and hoping it stays gone.

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
from reading the bytes back afterwards. So we wrote a small tool that does
exactly that: it builds a BLAS from a handful of triangles, copies the
result back to the CPU, and shows the bytes next to the geometry. Move a
vertex and you can see which bytes change.

The actual decoding we mostly left to an LLM (Codex), and it turned out to
be a really good fit. Figuring out a binary format is a loop: guess what a
field means, build some geometry that should change it, look at the bytes,
and throw the guess away when it's wrong. Codex runs that loop much faster
than a person and never gets tired of it. It also makes plenty of wrong
guesses, and makes them confidently. It spent a whole afternoon convinced
that the vertex positions were entangled through some shared exponent,
before working out that each vertex just stores the low bits that differ
from a reference vertex. So the rule became that nothing counts until the
hardware agrees: for every theory, we wrote our own bytes based on it and
checked that the RT core traced them correctly. By the end of the first
day, the GPU was tracing a BLAS we had written entirely by hand.

So what is NVIDIA hiding? Mostly very careful compression. The BVH is
12 wide, so there are few nodes, and each node stores its twelve child boxes
quantized down to one byte per plane. Triangles are packed into 128-byte
blocks, and inside a block each vertex only stores the low bits that differ
from the reference vertex. The compression is lossless, so you get back
exactly the floats you put in. On dense meshes this comes out at around 15
to 25 bytes per triangle, in line with what Arseny measured on Blackwell.
We wrote everything down in a spec detailed enough to build an encoder from,
and then built one.

## The ray tracing instructions

Knowing the format only gets you halfway. A shader still has to tell the RT
core to traverse it, and the instructions for that aren't documented either.
NVIDIA's own disassembler prints them as raw hex without names.

We started by compiling a set of small ray query shaders with NVIDIA's tools
to get a first look at the code. That doesn't tell you what the driver
really submits, though, so we captured that directly: a small preload
library scans the command streams the proprietary driver writes into its
mmapped buffers, follows them to the shader code, and dumps it.

From the captures we could work out how a traversal step works. The shader
writes the ray and either the BVH root or the state left over from the
previous step into the RT unit, starts it, and reads back either a hit
candidate that needs a decision from the shader, or a signal that traversal
is done. Opaque hits never reach the shader at all; the hardware handles
them on its own. `rayQueryProceedEXT` becomes a loop around that sequence,
and the rest of the ray query API is ordinary shader code that decodes the
result.

The hardware is picky about that sequence. Our first version emitted it as
separate instructions, and the compiler did what compilers do: it removed a
store it considered dead and moved some registers around. Every individual
instruction was still correct, but the RT core received garbage. NAK, NVK's
shader compiler, now treats the whole sequence as a single instruction until
the very end of compilation.

## From the first ray to the CTS

The first version had no BVH builder in the driver at all. We got ray query
shaders compiling, built the BLASes on the CPU with our own encoder,
uploaded them into normal buffers and wrote their device addresses into a
TLAS. The driver never needed to know they were acceleration structures.
That was enough to render a ray traced scene with NVK, three days after we
started looking at the format.

![The first ray traced image out of NVK: a shaderball lit with ray queries, traced against a BLAS built on the CPU by our own encoder](/images/nvk-first-light-shaderball.png)

Once the driver could build acceleration structures itself, existing Vulkan
software started working unmodified. Sascha Willems' ray query sample was the
first:

![Sascha Willems' ray traced shadows sample running on NVK on an RTX 5090 laptop GPU](/images/nvk-sascha-willems-ray-queries.png)

It also kept the problem small. Whenever an image was wrong, we could have
the proprietary driver build the same BLAS and see whether that fixed it,
which told us immediately whether the bug was in our BVH or in the shader.

After that we wrote a real BVH builder that runs on the GPU. It sits on top
of Mesa's common BVH code, which builds a binary tree; the NVK part collapses
that into 12-wide nodes and encodes the compressed triangle blocks. Ray
tracing pipelines followed, with each shader stage compiled as a real
function that the other stages call directly. Our mesh streaming works in
clusters, so we also implemented `VK_NV_cluster_acceleration_structure` on
the same builder.

Ray tracing pipelines are what most games use, which meant we could test
with the classic: Quake II RTX, path traced, on an open source driver.

![Quake II RTX running on NVK with ray tracing pipelines](/images/nvk-quake2-rtx.png)

Then we ran the Vulkan CTS until there was nothing left to fix. NVK now
passes the full ray tracing CTS.

<!-- TODO: CTS numbers for ray_query and ray_tracing_pipeline -->

## Running on NVIDIA's kernel driver

Fixing bugs ourselves only helps if the fixes reach our customers without
anyone having to change their setup. NVK normally runs on nouveau, the open
source kernel driver, and for us that was never an option. Nouveau doesn't
properly support memory compression, which costs a lot of performance, and
it can't run CUDA, which we and our customers need.

So we made NVK run on top of NVIDIA's proprietary kernel module instead. NVK
talks to the kernel module directly and uses none of NVIDIA's userspace
driver. The kernel interface isn't officially stable, but it changes slowly,
and we currently support every kernel driver release from 580 to 615.

That decouples us from our customers' driver updates. We ship NVK with our
application, so a driver fix now goes out with our next release and works
with whatever NVIDIA driver is already installed.

## Where we are now

When something in ray tracing breaks today, we can look at every layer of it
ourselves, from our Rust code down to the bits in the BVH, and fix the
problem where it actually is.

<!-- TODO: links, upstreaming plans -->
