# DP-SSTF-Net

## Overview
- DP-SSTFNet is a dual-path EEG classification network combining frequency- and time-domain processing with strong symmetric map/token-level gated fusion.

## Input
- Input tensor: (B, C, T) where B=batch, C=channels/electrodes, T=time samples.

## Main components
- MiCR: channel-wise attention module applied on spectral and temporal inputs.
- PKIFreqEncoder: multi-kernel frequency encoder producing feature maps (B, d, F).
- HistFreqAtt: histogram-inspired frequency attention that weights frequency bins.
- ETTE: channel-token encoder that converts spectral maps into per-channel tokens (B, C, d).
- LSSStack: lightweight sequence stack used for modeling across tokens or time.
- MiTPE: temporal patch embedding that produces time tokens (B, Lt, d).
- SPGFMap1D / SPGFToken: gated fusion modules for map-level and token-level fusion.
- GFU: vector-level gated fusion used for cross-representation merging.
- Head: a small MLP that maps final fused vector to class logits.

## Forward summary
1. Compute log power spectrum from raw signal via real FFT.
2. Obtain attended spectral/temporal versions with MiCR, encode with PKIFreqEncoder and MiTPE.
3. Build channel tokens with ETTE and model them with LSSStack.
4. Fuse attended/raw representations using SPGF modules and apply LSSStack on fused tokens.
5. Merge frequency and temporal vectors via GFU and produce logits with the head.

## Usage
- Import the model and call it with an input tensor: instantiate `DP_SSTFNet(...)` and run `model(x)`.


Code will be uploaded when paper is accepted. 
