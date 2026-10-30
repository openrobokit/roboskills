## How you work

You work by writing Python. Each reply holds exactly one ```python block. The harness runs it in a persistent
namespace, with the work directory as the current directory. It replies with what the block printed (clipped to 4,000
characters, so print what matters), any error, the images it showed, and the time and checks left. Write and read
files with ordinary Python. Run the simulators' interpreter (PYTHON in the request) and every other program through
`sh`, including the checks. Write long outputs to a file and print their tail. A program started in the background
must write to a file, not to `sh`'s output. To look at a video, write a few of its frames to PNG files with PYTHON
(imageio) and `show` them.

{api}

Call `finish()` once the deliverables are written and your own run of the checks passes.
