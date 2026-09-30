# Experiment 8: Matched Filtering, ISI, and Eye Diagrams

## 📑 Overview
This experiment explores the fundamental concepts of digital communication receivers. It demonstrates the principles of **matched-filter reception**, analyzes the impact of **Intersymbol Interference (ISI)** using **multipath channels**, and utilizes **eye diagrams** to visually assess noise margins, ISI, and timing sensitivity in a BPSK communication system.

---

## 🎯 Objectives
1. Demonstrate matched-filter reception and its role in maximizing the Signal-to-Noise Ratio (SNR).
2. Understand and implement Root Raised Cosine (RRC) pulse shaping to satisfy the Nyquist criterion.
3. Use eye diagrams to identify the effects of noise, ISI, and timing jitter.
4. Quantify system performance through Eye Height vs. SNR and BER vs. Timing Offset (Bathtub curve).

---

## 📚 Theoretical Background & Concepts Used

### 1. Root Raised Cosine (RRC) Pulse Shaping
To prevent ISI, the overall frequency response of the transmitter, channel, and receiver must satisfy the **Nyquist ISI criterion**. This is typically achieved by splitting a Raised Cosine (RC) filter equally between the transmitter and receiver, resulting in two **Root Raised Cosine (RRC)** filters.
* **Roll-off factor ($\beta$)**: Determines the excess bandwidth. $\beta = 0$ is minimum bandwidth (ideal sinc), $\beta = 1$ is maximum bandwidth (100% excess). Lower $\beta$ is spectrally efficient but harder to implement.
* **Filter Span**: The theoretical RRC impulse response is infinite in time. In practice, it must be truncated. The "span" defines how many symbol periods the filter covers.

### 2. Matched Filter Reception
A matched filter is a linear filter that maximizes the SNR at the sampling instant in the presence of additive stochastic noise. For a known transmit pulse $p(t)$, the optimal receive filter is $h(t) = p(T - t)$. Since the RRC filter is symmetric, the receive filter is identical to the transmit filter.

### 3. Intersymbol Interference (ISI) & Multipath
ISI occurs when the channel disperses the signal in time, causing previous or future symbols to bleed into the current symbol's sampling instant. A **multipath channel** models this by adding delayed and attenuated copies of the transmitted signal:
$$ y(t) = \sum_{k} g_k x(t - \tau_k) $$
where $g_k$ and $\tau_k$ are the gain and delay of the $k$-th path.

### 4. Eye Diagrams
An eye diagram is formed by superimposing multiple symbol-period traces of the received signal. 
* **Eye Height**: The vertical opening at the optimal sampling instant. It represents the **noise margin** (distance from the signal level to the decision threshold).
* **Eye Width**: The horizontal opening. It represents **timing sensitivity** (how much jitter the receiver can tolerate before sampling errors occur).
* **Slope of the eye**: Steeper slopes indicate higher sensitivity to timing jitter.

---

## 💻 How the Code Works

The Python script is structured into modular helper functions and sequential experiments, designed to run in a single Google Colab cell.

### Step-by-Step Workflow:
1. **System Initialization**: Generates a random BPSK symbol sequence ($\pm 1$) and upsamples it by inserting zeros (Samples Per Symbol, `sps = 8`).
2. **Transmit Chain**: Passes the upsampled symbols through an RRC filter to generate the continuous-time baseband signal.
3. **Channel Modeling**: 
   * Adds Additive White Gaussian Noise (AWGN) based on a specified SNR.
   * (Optional) Passes the signal through a multipath FIR filter to introduce ISI.
4. **Receive Chain**: 
   * Passes the noisy signal through the **matched RRC filter**.
   * **Mandatory Delay Compensation**: Calculates the total group delay of the Tx and Rx filters (`total_delay = 2 * filter_delay`) and shifts the signal array to align the symbols perfectly with the sampling grid.
5. **Detection & Hard Decision**: Samples the aligned signal at the symbol rate and applies a hard decision (`np.sign()`) to map the continuous amplitudes back to discrete $\pm 1$ symbols.
6. **Parameter Sweeps & Visualization**: 
   * Sweeps SNR to plot Eye Height vs. SNR.
   * Sweeps $\beta$ to observe bandwidth vs. time-domain trade-offs.
   * Sweeps fractional timing offsets to plot the BER "Bathtub" curve.

---

## ⚠️ Discrepancies, Reasons, and Mitigations

During the simulation and debugging process, several discrepancies between ideal theory and practical simulation were observed. 

### Discrepancy 1: 100% Bit Error Rate (BER) in Naive Implementation
* **Observation**: Initially, comparing the received samples directly to the transmitted symbols yielded a BER of 1.0 (100% errors), even at high SNR.
* **Reason**: The output of the matched filter is a continuous analog value (e.g., `0.94`, `-1.05`), while the transmitted symbols are discrete integers (`1`, `-1`). A direct equality check (`received == transmitted`) will always fail.
* **Mitigation**: Implemented a **hard decision slicer** using `np.sign()`. This maps any positive value to `+1` and any negative value to `-1` before calculating the BER, mimicking a real physical comparator circuit.

### Discrepancy 2: Residual ISI at Low Roll-off Factors ($\beta$)
* **Observation**: In the Roll-off Factor experiment, the eye diagram for $\beta = 0.1$ showed "fuzzy" crossings and slight eye closure, even without a multipath channel. Theory suggests an RRC filter should yield zero ISI.
* **Reason**: The theoretical RRC impulse response has infinite duration. The code truncates this response to a finite `span` (e.g., 8 symbols). A low $\beta$ results in a time-domain pulse that decays very slowly. Truncating a slowly decaying signal causes severe Gibbs phenomenon and residual ISI.
* **Mitigation**: To reduce this, the **filter span must be increased** (e.g., from 8 to 16 or 32 symbols). However, this is a trade-off: a larger span increases the filter's group delay and computational complexity.

### Discrepancy 3: BER Never Reaches Absolute Zero
* **Observation**: Even at optimal timing (offset = 0) and high SNR (20 dB), the BER is very low but rarely exactly `0.000`.
* **Reason**: AWGN is a continuous random process. Even with perfect Nyquist filtering (zero ISI), there is always a non-zero probability that a noise spike will push the signal sample across the decision threshold.
* **Mitigation**: In a real-world system, this is mitigated by:
  1. Increasing the transmit power (higher SNR).
  2. Using **Forward Error Correction (FEC)** codes (like LDPC or Turbo codes) to correct the residual random bit errors.

### Discrepancy 4: Edge Artifacts in the Timing Offset (Bathtub) Curve
* **Observation**: When sweeping the timing offset to $\pm 0.5$ symbol periods, the BER calculation occasionally threw index errors or showed unnatural spikes.
* **Reason**: Shifting the sampling index by a fractional symbol period pushes the sampling window outside the boundaries of the simulated array, causing Python's negative indexing or out-of-bounds errors to corrupt the data.
* **Mitigation**: Added robust **array padding** (`np.pad`) to the received signal before the timing sweep. This ensures that extreme timing offsets simply sample the padded zero/edge regions rather than wrapping around or crashing, accurately reflecting a receiver losing lock at extreme jitters.

---

## 🚀 How to Run

1. Open **Google Colab** (or any local Jupyter Notebook environment).
2. Ensure `numpy` and `matplotlib` are available (they are pre-installed in Colab).
3. Copy and paste the entire Python script into a **single code cell**.
4. Run the cell. The script will print the theoretical validations, observations, and render all required matplotlib figures inline.

---

## 📊 Expected Outputs

Running the code will generate the following visualizations and console outputs:
1. **Time-Domain Signals**: Plots of the Tx signal, noisy Rx signal, and Matched Filter output with explicit delay-compensation markers.
2. **Base Eye Diagram**: A clear, open eye diagram proving successful matched filtering and delay removal.
3. **Eye Height vs. SNR**: A monotonically increasing line graph proving that higher SNR yields larger noise margins.
4. **Roll-off Factor Comparison**: A $2 \times 3$ grid of eye diagrams showing the trade-off between spectral efficiency ($\beta$) and time-domain ISI.
5. **BER Bathtub Curve**: A log-scale plot showing minimum BER at $0$ offset, rising sharply as timing jitter approaches $\pm 0.5$ symbol periods.
6. **ISI Comparison**: Side-by-side eye diagrams comparing the pristine AWGN channel against the severely degraded multipath ISI channel.
