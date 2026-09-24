# Adaptive Vision --- Phase 4 Research Paper Source

## Source

Research paper:

**A Unified Lightweight YOLO Framework with Adaptive Frame Control and
Selective Re-Detection for Real-Time Road-Scene Monitoring**

Authors: Kanav Kapoor, Archit Bharti, Rhythem Sabharwal, Vishesh Bhatia

This file contains the relevant source material and verified details for
implementing **Phase 4 --- Environmental Adaptation / Illumination
Handling**.

Treat this file as a source-of-truth extract from the research paper. Do
not replace unspecified details with assumptions without explicitly
documenting them.

------------------------------------------------------------------------

## Paper Section: Illumination Handling

The relevant paper text states:

> To handle contrast degradation during night driving and stormy weather
> without deploying a heavy neural enhancement model, we monitor the
> average frame luminance **Y**. When **Y** falls below an empirical
> low-light threshold (**τ_illum = 45** on an 8-bit scale), the frame is
> routed through Contrast Limited Adaptive Histogram Equalization
> (CLAHE) applied specifically to the lightness channel in the
> **L*a*b**\* color space. This localized adjustment boosts edge
> visibility around underexposed objects without amplifying
> high-frequency noise across the background.

------------------------------------------------------------------------

## Verified Phase 4 Specification

### 1. Illumination measurement

The paper monitors the **average frame luminance Y**.

**Important:** The paper extract does **not explicitly define the
mathematical computation of average Y** (for example, it does not state
whether Y is computed as a grayscale mean, an RGB luminance-weighted
mean, an HSV V-channel mean, or another formula).

Therefore:

-   Do **not** claim that the paper specifies a particular luminance
    formula.
-   If an implementation requires a concrete formula, document the
    chosen method explicitly as an implementation assumption.
-   Do not silently attribute that formula to the paper.

### 2. Low-light threshold

The paper explicitly specifies:

**τ_illum = 45**

on an **8-bit scale**.

Low-light condition:

**Y \< 45**

When this condition is satisfied, the frame is routed through CLAHE.

### 3. CLAHE color space and channel

The paper explicitly specifies:

-   Color space: **L*a*b**\*
-   CLAHE target: **lightness channel L**

CLAHE must therefore operate on the **L channel**, not independently on
RGB channels.

### 4. CLAHE parameters

The paper section does **not specify** values for:

-   `clipLimit`
-   `tileGridSize`
-   any other CLAHE-specific OpenCV parameters

Therefore, these values must not be presented as values from the paper.

If implementation requires them, choose and document a standard/default
configuration separately as an implementation assumption.

### 5. Purpose of the stage

The stated purpose is to handle contrast degradation during:

-   night driving
-   stormy weather

The paper states that the localized adjustment is intended to:

-   boost edge visibility around underexposed objects
-   avoid amplifying high-frequency background noise

------------------------------------------------------------------------

## Pipeline Position

Phase 4 belongs between the Adaptive Frame Controller and the primary
lightweight YOLO detector:

``` text
Adaptive Frame Controller
        ↓
Environmental Adaptation / Illumination Handling
        ↓
Primary Lightweight YOLO
```

Phase 4 should not modify the behavior of the Phase 3 Adaptive Frame
Controller.

------------------------------------------------------------------------

## Implementation Constraints Derived From the Paper

-   Monitor average frame luminance **Y**.
-   Use the paper's low-light threshold **τ_illum = 45** on the 8-bit
    scale.
-   Apply CLAHE only when the low-light condition is satisfied.
-   Apply CLAHE to the **L channel of L*a*b**\*.
-   Do not apply CLAHE independently to RGB channels.
-   Normal/adequately illuminated frames should bypass the enhancement
    stage.
-   Do not implement selective re-detection, uncertainty filtering,
    fusion, tracking, or later phases as part of Phase 4.

------------------------------------------------------------------------

## Explicitly Unspecified by the Paper

The following details are **not explicitly specified in the relevant
paper section**:

1.  Exact mathematical formula used to compute average frame luminance
    **Y**.
2.  CLAHE `clipLimit`.
3.  CLAHE `tileGridSize`.
4.  Any additional OpenCV-specific CLAHE settings.

These must be treated as implementation decisions/assumptions, not as
paper-derived facts.

------------------------------------------------------------------------

## Source-of-Truth Rule

When implementing Phase 4:

**Paper-specified values must be followed exactly.**

For unspecified values:

**Do not invent or attribute them to the paper.**

Instead, report them as:

> Paper does not specify --- using \[chosen implementation value\].

This distinction is important for research reproducibility and for
accurately describing the methodology in the final report.
