'''
    Purpose: Compare repository coated-sphere RCS against imported DGTD results.

    This test reads:
      - dgtd_simulations/dgtd_material_shell.txt
      - dgtd_simulations/rcs_g*.dat

    Assumed DGTD setup from material file:
      - PEC core radius a (m)
      - dielectric shell thickness d (m)
      - shell material (epsilon_r, mu_r, sigma)
      - background air/vacuum

    Output:
      - Console summary of error metrics per DGTD file
      - output/dgtd_comparison_<name>.txt with frequency-by-frequency comparison
'''

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from CoatedSphere import CoatedSphere
from DielectricMaterial import DielectricMaterial
from getRCS import RCS_vs_freq_shell


def parse_material_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    patterns = {
        "a": r"a\s*=\s*([0-9.eE+-]+)",
        "eps_r": r"εr\s*=\s*([0-9.eE+-]+)",
        "mu_r": r"μr\s*=\s*([0-9.eE+-]+)",
        "sigma": r"σ\s*=\s*([0-9.eE+-]+)",
        "d": r"d\s*=\s*([0-9.eE+-]+)",
    }

    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if not m:
            raise ValueError(f"Could not parse '{key}' from {path}")
        out[key] = float(m.group(1))

    return out


def load_dgtd_rcs(path):
    # Columns: theta(rad), phi(rad), frequency(Hz), rcs, normalization_term
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    freq = data[:, 2].astype(float)
    rcs = data[:, 3].astype(float)
    return freq, rcs


def error_metrics(reference, estimate):
    denom = np.maximum(np.abs(reference), 1e-30)
    abs_err = np.abs(estimate - reference)
    rel_err = abs_err / denom

    rmse = np.sqrt(np.mean((estimate - reference) ** 2))
    mape = 100.0 * np.mean(rel_err)

    return {
        "max_abs": float(np.max(abs_err)),
        "mean_abs": float(np.mean(abs_err)),
        "max_rel": float(np.max(rel_err)),
        "mean_rel": float(np.mean(rel_err)),
        "rmse": float(rmse),
        "mape_percent": float(mape),
    }


def write_comparison_table(path, freq, dgtd_rcs, model_rcs, valid_mask):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    abs_err = np.abs(model_rcs - dgtd_rcs)
    rel_err = abs_err / np.maximum(np.abs(dgtd_rcs), 1e-30)

    with open(path, "w", encoding="utf-8") as f:
        f.write("frequency_hz\tdgtd_rcs\tmodel_rcs\tabs_error\trel_error\tvalid\n")
        for i in range(len(freq)):
            f.write(
                f"{freq[i]:.9e}\t{dgtd_rcs[i]:.9e}\t{model_rcs[i]:.9e}\t"
                f"{abs_err[i]:.9e}\t{rel_err[i]:.9e}\t{int(valid_mask[i])}\n"
            )


def write_comparison_plot(path, freq, dgtd_rcs, model_rcs, valid_mask, name, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    safe_dgtd = np.maximum(np.abs(dgtd_rcs), 1e-30)
    rel_err = np.abs(model_rcs - dgtd_rcs) / safe_dgtd

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax0.loglog(freq, dgtd_rcs, label="DGTD", linewidth=2.0)
    ax0.loglog(freq, np.abs(model_rcs), label="Model", linewidth=1.5)
    ax0.set_ylabel("RCS (m^2)")
    ax0.grid(True, which="both", ls="--", alpha=0.5)
    ax0.legend()

    if metrics is None:
        subtitle = f"{name} | no valid overlapping samples"
    else:
        subtitle = (
            f"{name} | valid={int(np.sum(valid_mask))}/{len(valid_mask)} | "
            f"mean rel={metrics['mean_rel']:.3e} | max rel={metrics['max_rel']:.3e}"
        )
    ax0.set_title(f"DGTD vs Model RCS\n{subtitle}")

    if np.any(valid_mask):
        ax1.semilogy(freq[valid_mask], rel_err[valid_mask], label="Relative error", color="tab:red")
    if np.any(~valid_mask):
        ax1.scatter(
            freq[~valid_mask],
            np.full(np.sum(~valid_mask), 1e0),
            s=16,
            color="black",
            marker="x",
            label="Non-finite model samples",
        )

    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("|Model-DGTD| / |DGTD|")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    print("------\nDGTD PEC+SHELL COMPARISON TEST\n------\n")

    root = os.path.dirname(os.path.abspath(__file__))
    material_file = os.path.join(root, "dgtd_simulations", "dgtd_material_shell.txt")
    dgtd_files = sorted(glob.glob(os.path.join(root, "dgtd_simulations", "rcs_g*.dat")))

    if not dgtd_files:
        raise FileNotFoundError("No rcs_g*.dat files found under dgtd_simulations/")

    params = parse_material_file(material_file)
    core_radius = params["a"]
    shell_width = params["d"]

    air = DielectricMaterial(1.0, 0.0, name="Air")
    pec = DielectricMaterial(1e8, 0.0, 1.0, 0.0, name="PEC")
    shell = DielectricMaterial(params["eps_r"], params["sigma"], params["mu_r"], 0.0, name="DGTD Shell")
    coated = CoatedSphere(pec, shell, core_radius=core_radius, shell_width=shell_width)

    sensor_location = [0, 0, -2000]
    c0 = 299792458.0

    print("Material/model setup from dgtd_material_shell.txt:")
    print(f"  core radius a     = {core_radius:.6f} m")
    print(f"  shell width d     = {shell_width:.6f} m")
    print(f"  shell epsilon_r   = {params['eps_r']:.6f}")
    print(f"  shell mu_r        = {params['mu_r']:.6f}")
    print(f"  shell sigma       = {params['sigma']:.6f} S/m")
    print("")

    for dgtd_path in dgtd_files:
        name = os.path.splitext(os.path.basename(dgtd_path))[0]
        freq, dgtd_rcs = load_dgtd_rcs(dgtd_path)

        ratio = coated.radius * freq / c0
        _, model_rcs_complex = RCS_vs_freq_shell(
            coated,
            ratio,
            air,
            sensor_location,
            show_plot=0,
        )
        model_rcs = np.real(np.asarray(model_rcs_complex))

        valid_mask = np.isfinite(dgtd_rcs) & np.isfinite(model_rcs)
        valid_count = int(np.sum(valid_mask))
        total_count = len(freq)

        if valid_count == 0:
            metrics = None
        else:
            metrics = error_metrics(dgtd_rcs[valid_mask], model_rcs[valid_mask])

        out_file = os.path.join(root, "output", f"dgtd_comparison_{name}.txt")
        plot_file = os.path.join(root, "output", f"dgtd_comparison_{name}.png")
        write_comparison_table(out_file, freq, dgtd_rcs, model_rcs, valid_mask)
        write_comparison_plot(plot_file, freq, dgtd_rcs, model_rcs, valid_mask, name, metrics)

        print(f"File: {name}")
        print(f"  Samples:    {total_count}")
        print(f"  Valid:      {valid_count}/{total_count}")
        if metrics is None:
            print("  Metrics:    no valid overlapping samples")
        else:
            print(f"  Max abs:    {metrics['max_abs']:.6e}")
            print(f"  Mean abs:   {metrics['mean_abs']:.6e}")
            print(f"  Max rel:    {metrics['max_rel']:.6e}")
            print(f"  Mean rel:   {metrics['mean_rel']:.6e}")
            print(f"  RMSE:       {metrics['rmse']:.6e}")
            print(f"  MAPE (%):   {metrics['mape_percent']:.3f}")

        dropped = np.where(~valid_mask)[0]
        if dropped.size > 0:
            first_bad = int(dropped[0])
            print(f"  First non-finite sample index: {first_bad}")
            print(f"  First non-finite frequency:    {freq[first_bad]:.6e} Hz")

        print(f"  Saved:      {os.path.relpath(out_file, root)}")
        print(f"  Plot:       {os.path.relpath(plot_file, root)}")
        print("")

    print("Finished DGTD comparison test.\n")
