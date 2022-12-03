## 2 unrelated things are stacked into this repo
# Silicon-Unleashed
Final project for a vision course I took at NYU. Look in release for all code implementations, and full overview in report [here](https://github.com/ImanHosseini/Silicon-Unleashed/blob/main/Report.pdf). <br>
The idea was to do convolution over pretty much anything I could get my hands on, and in as many ways as time would allow me and compare the development experience and performance. So I took the easy task of convolution, and considered dimensions to be 128x128. The summary of performance is: on my XPS laptop my handcrafted SIMD (AVX) code (which you can see [here](https://github.com/ImanHosseini/Silicon-Unleashed/blob/main/fastconv.cpp) is remarkably fast at only 70 us while numpy's optimized gaussian blur is at ~ 1000 us. One MKL turned out to be trash at around 4500 us. CUDA is around 60 us (it's a GTX 1650 Ti) and my FPGA implementation at around 3000 us (only the computation part, so the time to get data in and out not considered, as in all other measurements, also this is based on running on an Arty7 board on 50 MHz).

<img src="https://raw.githubusercontent.com/ImanHosseini/Silicon-Unleashed/db80ebf70126586273849eb794a984b179d4cebd/poster.SVG" width="1000" />

# Blitz
I once needed to get all debian packages, and build them and match up C source code with assemblies for each function. 
### Parser
/blitz/blitz/parser.py parses the .s files, uses debug info and matches up source and assembly lines of each function.
### Hooking
But how do we just run 'apt deb-build' and get the .s files? Is there a magic way, that we don't have to care about the details of the bulid process/ change anything and still get the .s files? Well there is: just hook unlink syscall! The intermediate '.s' files generated get removed via unlink, now with our hooked unlink we can keep them. Achieved by:
```python
env['LD_PRELOAD'] = "/home/blitz/unlink/myunlink.so"
```
And you can see the code for the hooked unlink in 'unlink.c'.
### Dockers. Many Dockers
And of course we want to do this fast. On a many core server. And safely, we don't want to just mess up our system. So I dockerized it and 'spinner.py' spins up docker containers to do this, and a 'monitor2.py' process reads from debian apt and hands off packages to them to 'harvest' (once they finish they get new pkg from monitor), also monitor has a slick terminal-based UI using rich library (https://github.com/Textualize/rich) to track progress, show errors and in case something is stuck you can see it in the TUI and connect to that container and manually fix it. This is remarkably fast, on a 128 core system I got to 1M+ functions extracted at around 15 hrs and all cores had 90%+ utilization most of the time.

<img src="https://raw.githubusercontent.com/ImanHosseini/Silicon-Unleashed-Blitz/main/blitz/blitzy.png" width="1000" />

This stuff (Blitz) was done for this paper: https://www.ndss-symposium.org/wp-content/uploads/bar2022_23009_paper.pdf
