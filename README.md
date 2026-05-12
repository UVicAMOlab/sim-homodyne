# Homodyne Detection of Temporally Resolved Quantum States

This directory contains python code to simulate homodyne photocurrent analysis. We deal with **timing jitter**, **phase jitter**, **mode and marginal inference**, and **Wigner function reconstruction**.

The first section of the notebook contains the framework for simulating homodyne photocurrent.

## BHD Photocurrent Framework

Here, we go over the important functions that create our framework for simulating BHD.

`invtrans_sample`: This helper function performs inverse transform sampling and is used throughout the other functions.

`generate_basis`: This helper function takes a principal temporal mode $y$ and the corresponding time axis $t$, and generates a complete basis for the time axis.

`gen_photocurrent`: This function is the lynchpin of our method. It takes a large number of parameters and produces an array of `Ntraces` photocurrents.

`W_to_marge`: This helper function takes a Wigner function, and angle, and an x array and generates the marginal distribution of $W$ along the angle given.

`marge_to_cdf`: This helper function simply generates the CDF of the PDF. This is simple, but we use it enough to warrant an easy method.

`state_to_objects`: This function takes a QuTiP-defined state and x, p arrays, as well as an angle. It generates a Wigner function, as well as a marginal and a CDF along the given angle.

Everything after this is used to analyze the effect of realistic errors, and are explained within the notebook.

  *Owen Sandner*
  2026-04-13
