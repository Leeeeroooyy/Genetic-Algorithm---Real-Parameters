import csv
import math
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIMENSION = 10
MAX_EVALUATIONS = 10_000
RUNS = 10
POPULATION_SIZE = 50
ELITE_COUNT = 2
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.9
BITS = 32
BCD_DIGITS = 7  # 8 nibbles = sign nibble + 7 decimal digits

SIGMAS = [0.001, 0.01, 0.1]
MUTATION_PROBABILITIES = [0.05, 0.1, 0.3]
MAIN_VARIANTS = {
    "binary IEEE 754": "IEEE 754",
    "binary Fixed-Point": "Fixed-Point",
    "binary BCD": "BCD",
    "real N sigma=0.01 pm=0.1": "real N\nσ=0.01\npm=0.1",
    "real U pm=0.1": "real U\npm=0.1",
}
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED = "#0b0b0b", "#898781"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
})


def sphere(x):
    return np.sum(x ** 2, axis=1)


def rosenbrock(x):
    return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


def schwefel(x):
    return 418.9828872724338 * x.shape[1] -np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def shifted_sphere(x):
    return sphere(x - 1.234)


FUNCTIONS = {
    "Sphere": (sphere, -5.12, 5.12),
    "Rosenbrock": (rosenbrock, -2.048, 2.048),
    "Schwefel": (schwefel, -500.0, 500.0),
}


def integers_to_bits(values):
    """uint32 array (n, D) -> bit array (n, D * 32), most significant bit first."""
    shifts = np.arange(BITS - 1, -1, -1, dtype=np.uint64)
    bits = (values[..., None].astype(np.uint64) >> shifts) & 1
    return bits.astype(np.uint8).reshape(len(values), -1)


def bits_to_integers(bits):
    weights = 2 ** np.arange(BITS - 1, -1, -1, dtype=np.uint64)
    return (bits.reshape(len(bits), -1, BITS).astype(np.uint64) @ weights).astype(np.uint32)


def ieee754_encode(x, low, high):
    return integers_to_bits(x.astype(np.float32).view(np.uint32))


def ieee754_decode(bits, low, high):
    with np.errstate(invalid="ignore"):
        x = bits_to_integers(bits).view(np.float32).astype(np.float64)
    x[np.isnan(x)] = high
    return np.clip(x, low, high)


def fixed_point_fraction_bits(low, high):
    integer_bits = math.ceil(math.log2(max(abs(low), abs(high)) + 1))
    return BITS - 1 - integer_bits  # 1 bit is the sign (two's complement)


def fixed_point_encode(x, low, high):
    scaled = np.round(x * 2.0 ** fixed_point_fraction_bits(low, high)).astype(np.int32)
    return integers_to_bits(scaled.view(np.uint32))


def fixed_point_decode(bits, low, high):
    scaled = bits_to_integers(bits).view(np.int32)
    return np.clip(scaled / 2.0 ** fixed_point_fraction_bits(low, high), low, high)


def bcd_decimal_places(low, high):
    integer_digits = len(str(int(max(abs(low), abs(high)))))
    return BCD_DIGITS - integer_digits


def bcd_encode(x, low, high):
    places = 10 ** np.arange(BCD_DIGITS - 1, -1, -1)
    magnitude = np.round(np.abs(x) * 10 ** bcd_decimal_places(low, high)).astype(np.int64)
    digits = (magnitude[..., None] // places) % 10
    sign = np.where(x < 0, 8, 0)[..., None]
    nibbles = np.concatenate([sign, digits], axis=2)
    bits = (nibbles[..., None] >> np.arange(3, -1, -1)) & 1
    return bits.astype(np.uint8).reshape(len(x), -1)


def bcd_decode(bits, low, high):
    places = 10 ** np.arange(BCD_DIGITS - 1, -1, -1)
    nibbles = bits.reshape(len(bits), -1, 8, 4) @ np.array([8, 4, 2, 1])
    digits = np.minimum(nibbles[..., 1:], 9)  # invalid codes 10-15 are read as 9
    magnitude = (digits @ places) / 10 ** bcd_decimal_places(low, high)
    x = np.where(nibbles[..., 0] >= 8, -magnitude, magnitude)
    return np.clip(x, low, high)


ENCODINGS = {
    "IEEE 754": (ieee754_encode, ieee754_decode),
    "Fixed-Point": (fixed_point_encode, fixed_point_decode),
    "BCD": (bcd_encode, bcd_decode),
}


def tournament_selection(fitness, count, rng):
    candidates = rng.integers(0, len(fitness), size=(count, TOURNAMENT_SIZE))
    winners = np.argmin(fitness[candidates], axis=1)
    return candidates[np.arange(count), winners]


def single_point_crossover(parents, rng):
    """Parents (0, 1), (2, 3), ... swap their tails after a random cut point."""
    first, second = parents[0::2], parents[1::2]
    length = parents.shape[1]
    points = rng.integers(1, length, size=len(first))
    points[rng.random(len(first)) >= CROSSOVER_PROBABILITY] = length  # no crossover
    take_first = np.arange(length) < points[:, None]
    return np.concatenate([np.where(take_first, first, second), np.where(take_first, second, first)])


def genetic_algorithm(function, create_population, decode, mutate, rng):
    """Returns the best-so-far value after every fitness evaluation."""
    population = create_population(POPULATION_SIZE)
    fitness = function(decode(population))
    evaluated = [fitness]
    evaluations = len(fitness)

    while evaluations < MAX_EVALUATIONS:
        child_count = min(POPULATION_SIZE - ELITE_COUNT, MAX_EVALUATIONS - evaluations)
        parents = population[tournament_selection(fitness, child_count + child_count % 2, rng)]
        children = mutate(single_point_crossover(parents, rng))[:child_count]
        children_fitness = function(decode(children))

        elite = np.argsort(fitness)[:ELITE_COUNT]
        population = np.concatenate([population[elite], children])
        fitness = np.concatenate([fitness[elite], children_fitness])
        evaluated.append(children_fitness)
        evaluations += child_count

    return np.minimum.accumulate(np.concatenate(evaluated))


def binary_ga(function, low, high, rng, encoding):
    encode, decode = ENCODINGS[encoding]
    bit_probability = 1 / (DIMENSION * BITS)

    def create_population(size):
        return encode(rng.uniform(low, high, (size, DIMENSION)), low, high)

    def mutate(bits):
        return bits ^ (rng.random(bits.shape) < bit_probability).astype(np.uint8)

    return genetic_algorithm(function, create_population, lambda bits: decode(bits, low, high), mutate, rng)


def real_ga(function, low, high, rng, mutation, probability, sigma=0.0):
    """sigma is a fraction of the domain width."""
    def create_population(size):
        return rng.uniform(low, high, (size, DIMENSION))

    def mutate(x):
        mask = rng.random(x.shape) < probability
        if mutation == "normal":
            new_values = np.clip(x + rng.normal(0, sigma * (high - low), x.shape), low, high)
        else:
            new_values = rng.uniform(low, high, x.shape)
        return np.where(mask, new_values, x)

    return genetic_algorithm(function, create_population, lambda x: x, mutate, rng)


def create_variants():
    variants = {f"binary {name}": {"encoding": name} for name in ENCODINGS}
    for sigma in SIGMAS:
        for probability in MUTATION_PROBABILITIES:
            variants[f"real N sigma={sigma} pm={probability}"] = {
                "mutation": "normal", "sigma": sigma, "probability": probability}
    for probability in MUTATION_PROBABILITIES:
        variants[f"real U pm={probability}"] = {"mutation": "uniform", "probability": probability}
    return variants


def run_variant(settings, function, low, high):
    """Runs one variant RUNS times, returns an array (RUNS, MAX_EVALUATIONS)."""
    algorithm = binary_ga if "encoding" in settings else real_ga
    return np.array([algorithm(function, low, high, np.random.default_rng(seed), **settings)
                     for seed in range(RUNS)])


def statistics(final_values):
    return {
        "best": final_values.min(),
        "worst": final_values.max(),
        "mean": final_values.mean(),
        "median": np.median(final_values),
        "std": final_values.std(ddof=1),
    }


def fmt(value):
    return f"{value:.3g}" if 1e-3 <= abs(value) < 1e4 or value == 0 else f"{value:.2e}"


def print_table(title, rows):
    print(f"\n{'=' * 85}\n{title}\n{'=' * 85}")
    print(f"{'Variant':<28} {'Best':>10} {'Worst':>10} {'Mean':>10} {'Median':>10} {'Std':>10}")
    for row in rows:
        values = " ".join(f"{fmt(row[key]):>10}" for key in ("best", "worst", "mean", "median", "std"))
        print(f"{row['variant']:<28} {values}")


def run_experiments(variants):
    histories = {}
    rows = []
    for function_name, (function, low, high) in FUNCTIONS.items():
        function_rows = []
        for variant_name, settings in variants.items():
            histories[function_name, variant_name] = run_variant(settings, function, low, high)
            stats = statistics(histories[function_name, variant_name][:, -1])
            function_rows.append({"function": function_name, "variant": variant_name, **stats})
        print_table(f"{function_name} - {DIMENSION}D, {RUNS} runs, {MAX_EVALUATIONS} evaluations", function_rows)
        rows += function_rows
    return histories, rows


def control_experiment(variants):
    """Sphere with the optimum moved from 0 to 1.234 in every dimension."""
    rows = []
    for variant_name in MAIN_VARIANTS:
        history = run_variant(variants[variant_name], shifted_sphere, -5.12, 5.12)
        rows.append({"function": "Shifted Sphere", "variant": variant_name, **statistics(history[:, -1])})
    print_table("Control: Sphere shifted to 1.234", rows)
    return rows


def save_summary(rows):
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_convergence(histories):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    evaluations = np.arange(1, MAX_EVALUATIONS + 1)
    for ax, function_name in zip(axes, FUNCTIONS):
        for color, (variant_name, label) in zip(COLORS, MAIN_VARIANTS.items()):
            history = np.maximum(histories[function_name, variant_name], 1e-300)
            style = "--" if variant_name.startswith("binary") else "-"
            ax.plot(evaluations, np.median(history, axis=0), style, color=color, lw=1.6,
                    label=label.replace("\n", " "))
            ax.fill_between(evaluations, *np.percentile(history, [25, 75], axis=0), color=color, alpha=0.12, lw=0)
        ax.set_yscale("log")
        ax.set_title(function_name, loc="left", fontweight="bold")
        ax.set_xlabel("evaluations")
    axes[0].set_ylabel("best f (median, IQR band)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUTPUT_DIR / "convergence.png", dpi=150)
    plt.close(fig)


def draw_boxplot(histories):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, function_name in zip(axes, FUNCTIONS):
        data = [np.maximum(histories[function_name, name][:, -1], 1e-300) for name in MAIN_VARIANTS]
        boxes = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops={"color": INK, "lw": 1.2},
                           whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                           flierprops={"marker": "o", "ms": 3, "mec": MUTED})
        for patch, color in zip(boxes["boxes"], COLORS):
            patch.set(facecolor=color, alpha=0.75, edgecolor=color)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(MAIN_VARIANTS) + 1), list(MAIN_VARIANTS.values()))
        ax.set_title(function_name, loc="left", fontweight="bold")
    axes[0].set_ylabel(f"final best f ({RUNS} runs)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "boxplot.png", dpi=150)
    plt.close(fig)


def draw_sweep(histories):
    rows = [f"N σ={sigma}" for sigma in SIGMAS] + ["U (full range)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, function_name in zip(axes, FUNCTIONS):
        names = [[f"real N sigma={s} pm={p}" for p in MUTATION_PROBABILITIES] for s in SIGMAS]
        names.append([f"real U pm={p}" for p in MUTATION_PROBABILITIES])
        medians = np.array([[np.median(histories[function_name, n][:, -1]) for n in row] for row in names])
        logs = np.log10(np.maximum(medians, 1e-300))
        ax.imshow(logs, cmap="Blues_r", aspect="auto")
        for i in range(medians.shape[0]):
            for j in range(medians.shape[1]):
                dark = logs[i, j] < logs.min() + 0.45 * (logs.max() - logs.min())
                ax.text(j, i, fmt(medians[i, j]), ha="center", va="center", color="white" if dark else INK)
        ax.set_xticks(range(len(MUTATION_PROBABILITIES)), [f"pm={p}" for p in MUTATION_PROBABILITIES])
        ax.set_yticks(range(len(rows)), rows)
        ax.grid(False)
        ax.set_title(f"{function_name}: median best f (darker = better)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sweep.png", dpi=150)
    plt.close(fig)


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    variants = create_variants()

    histories, rows = run_experiments(variants)
    rows += control_experiment(variants)
    save_summary(rows)

    draw_convergence(histories)
    draw_boxplot(histories)
    draw_sweep(histories)
    print(f"\nDone in {time.time() - start:.1f} s. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
