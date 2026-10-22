+++
title = "Reverse engineering NVIDIA's ray tracing hardware"
date = 2026-10-21
authors = ["Axel Gneiting"]
+++

NVIDIA makes the best ray tracing GPUs, and we don't plan to switch. But we
kept running into problems with NVIDIA's driver, and most of them came from using features that very few people use yet. Our renderer
depends on descriptor heaps, cluster acceleration structures and ray queries
in nearly every pass. That's where we kept hitting shader compiler bugs and
other driver bugs. When you can't see what the compiler did with your shader,
all you can do is change the code until the problem goes away and hope it
stays gone.

Even when NVIDIA fixed something, the fix didn't really reach us. The
proprietary Vulkan driver only works with its matching kernel driver, so to
pick up a fix our customers had to update their whole NVIDIA driver. We had
to keep asking them to do that.

We are power users, and we want to be able to debug everything. Our engine is
written in Rust, and when a frame comes out wrong we want to be able to
follow it all the way down: through the Vulkan driver, the shader compiler
and the acceleration structure builder, to the SASS the GPU actually runs.
The only way to get that is to own the whole stack.

The open source option is NVK, the Vulkan driver for NVIDIA GPUs in Mesa. It
was already good enough for most of what we do, but it had no ray tracing. So
we added it. Today NVK has ray queries, ray tracing pipelines, a GPU BVH
builder and `VK_NV_cluster_acceleration_structure` on both Ada and Blackwell,
and it passes the Vulkan ray tracing CTS. It also runs on NVIDIA's own kernel
driver, which turned out to matter just as much as the ray tracing.

None of this is documented by NVIDIA. The rest of this post is about how we
figured it out.

## The acceleration structure format

We had always wondered how NVIDIA's acceleration structures were so good.
They trace fast, and they're small. When Arseny Kapoulkine
[measured acceleration structure sizes](https://zeux.io/2025/03/31/measuring-acceleration-structures/)
across vendors, NVIDIA's BLASes came out at roughly 2.5 times smaller than
AMD's and Intel's for the same scene. AMD and Intel document their formats
well enough that he could explain where their bytes go. For NVIDIA he
couldn't: "pretty much nothing is known about NVidia, but they clearly have a
significant competitive advantage here so maybe they have something to hide."

To trace rays on NVIDIA hardware without NVIDIA's driver, we had to find out
what that something was. The proprietary driver will happily build an
acceleration structure for any geometry you give it, and nothing stops you
from reading the bytes back afterwards.

So we wrote a small tool that does exactly that. It builds a BLAS with the
proprietary driver from a handful of triangles, copies the result back to
the CPU, and shows it next to the input geometry. Move a vertex and you can
see which bytes change.

Then we let an LLM (Codex) do most of the actual decoding. This turned out
to be a really good fit. Figuring out a binary format is mostly a loop:
guess what a field means, build some geometry that should change it, look
at the bytes, and throw the guess away when it's wrong. Codex can run that
loop much faster than a person, and it doesn't get tired of it. It also
makes plenty of wrong guesses, and confidently. For a whole afternoon it was
convinced the vertex positions were tied together through some shared
exponent, until it found out they're just stored as the differing low bits
relative to a reference vertex. So we only trusted a result once the
hardware agreed with it: we wrote our own bytes based on the theory and
checked that the RT core traced them correctly.

On the first day we got a BLAS that we had written completely by hand to
trace correctly on the GPU.

So what is NVIDIA hiding? Mostly very careful compression. The BVH is
12 wide, so there are few nodes, and each node stores its twelve child boxes
quantized down to one byte per plane. Triangles are packed into 128-byte
blocks, and inside a block each vertex only stores the low bits that differ
from a reference vertex. That compression is lossless, so you get back
exactly the floats you put in. On dense meshes this comes out at around 15
to 25 bytes per triangle, in line with what Arseny measured on Blackwell.
We wrote everything down in a spec detailed enough to build an encoder from,
and then built one.

## The ray tracing instructions

Knowing the format only gets you halfway. A shader also has to tell the RT
core to traverse it, and the instructions for that aren't documented
either. NVIDIA's disassembler prints them as raw hex without names.

We compiled a set of small ray query shaders with NVIDIA's tools to get a
first idea of what the code looks like. To see what the driver really
submits, we then captured it directly: a small preload library scans the
command streams the proprietary driver writes into its mmapped buffers,
follows them to the shader code, and dumps it.

From that we could work out how a traversal step works. The shader writes
the ray and either the BVH root or the state from the previous step into
the RT unit, starts it, and reads back either a hit candidate that needs a
decision from the shader, or a signal that traversal is done. Opaque hits
are handled entirely in hardware. `rayQueryProceedEXT` turns into a loop
around that sequence, and the rest of the ray query API is ordinary shader
code that decodes the result.

The hardware is picky about that sequence. Our first version emitted it as
separate instructions, and the compiler did what compilers do: it removed a
store it thought was dead and moved registers around. Every individual
instruction was still correct, but the RT core got garbage. NAK, NVK's
shader compiler, now treats the whole sequence as one instruction until the
very end of compilation.

## From the first ray to the CTS

The first version didn't have a BVH builder in the driver at all. We got
ray query shaders compiling, built the BLASes on the CPU with our own
encoder, uploaded them into normal buffers and put their device addresses
into a TLAS. The driver didn't need to know they were acceleration
structures. That was enough to render a ray traced scene with NVK, three
days after we started looking at the format.

It also kept the problem small. If an image was wrong we could have the
proprietary driver build the same BLAS and see whether that fixed it, which
told us right away whether the bug was in our BVH or in the shader side.

After that we wrote a real BVH builder that runs on the GPU. It's built on
Mesa's common BVH code, which builds a binary tree; the NVK part collapses
that into 12-wide nodes and encodes the compressed triangle blocks. Ray
tracing pipelines followed, with each shader stage compiled as a real
function that the other stages call directly.

Finally we ran the Vulkan CTS until there was nothing left to fix. NVK now
passes the full ray tracing CTS.

<!-- TODO: CTS numbers for ray_query and ray_tracing_pipeline -->

## Running on NVIDIA's kernel driver

Fixing bugs ourselves only helps if the fixes get to our customers without
asking them to change their setup. NVK normally runs on nouveau, the open
source kernel driver, but that wasn't an option for us. Nouveau doesn't
properly support memory compression, which costs a lot of performance, and
it can't run CUDA, which we and our customers need.

So we made NVK run on top of NVIDIA's proprietary kernel module instead. NVK
talks to the kernel module directly and doesn't use any of NVIDIA's
userspace driver. The kernel interface isn't officially stable, but it
changes slowly, and we currently support every kernel driver release from
580 to 615.

This decouples us from our customers' driver updates. We ship NVK with our
application, so when we fix a driver bug, the fix goes out with our next
release, and it works with whatever NVIDIA driver they already have
installed.

## Clusters

We stream meshes in clusters, so we also implemented
`VK_NV_cluster_acceleration_structure`. Each cluster gets its own small
acceleration structure without any absolute pointers in it, so when the
streaming system moves a cluster around in memory it's just a copy. BLASes
are then built over those clusters on the GPU.

## Where we are now

When something in ray tracing breaks now, we can look at every layer of it ourselves, from our Rust
code down to the bits in the BVH, and fix the problem where it actually is.

<!-- TODO: links, upstreaming plans -->
