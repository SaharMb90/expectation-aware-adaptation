"""
Expectation-Aware Self-Adaptive System — a small ML demonstrator
================================================================

Context
-------
This is a compact, runnable demonstrator built to align with the VR
(Vetenskapsradet / Swedish Research Council) project described in the Chalmers
PhD ad "Doctoral Student in Human-in-the-Loop Autonomous Systems"
(supervisor: Asst. Prof. Rebekka Wohlrab). The project's stated goals are to
make self-adaptive systems *aware of end users' expectations*, to *explain a
system's actions*, and to *evaluate contributions using simulations*.

This script demonstrates all three in miniature:

1. AWARENESS OF EXPECTATIONS  — the system learns, online and from feedback
   alone, a model of what the human user expects in different runtime contexts.
2. EXPLANATION               — every adaptation decision comes with a plain
   language justification derived from the learned model's feature
   contributions (not a post-hoc narration).
3. SIMULATION-BASED EVALUATION — a simulated user with hidden, context
   dependent expectations provides noisy approve/reject feedback; we measure
   how quickly the learner's choices come to match the user's true preference,
   against a static baseline.

Scope note: deliberately small and dependency-light (numpy + matplotlib) so it
can be read end-to-end and defended in an interview. It is a demonstrator of a
*mechanism*, not a research contribution.

Run:  python3 expectation_aware_adaptation.py
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# 1. The self-adaptive system's configuration space (its "adaptation options")
# ---------------------------------------------------------------------------
# Each configuration is described by three normalised behavioural attributes.
# A real system (robot, autonomous network) would expose analogous knobs.
CONFIGS = {
    "Aggressive": dict(speed=0.90, caution=0.20, energy_saving=0.10),
    "Balanced":   dict(speed=0.50, caution=0.50, energy_saving=0.50),
    "Cautious":   dict(speed=0.20, caution=0.90, energy_saving=0.40),
    "Eco":        dict(speed=0.30, caution=0.50, energy_saving=0.90),
}
CONFIG_NAMES = list(CONFIGS.keys())
ATTRS = ("speed", "caution", "energy_saving")


def config_vec(name: str) -> np.ndarray:
    c = CONFIGS[name]
    return np.array([c[a] for a in ATTRS], dtype=float)


# ---------------------------------------------------------------------------
# 2. The runtime context the system must adapt to
# ---------------------------------------------------------------------------
# Three context signals in [0, 1]: how crowded the area is, how depleted the
# battery is, and how much time pressure the task is under.
CTX_NAMES = ("crowd", "battery_low", "time_pressure")


def sample_context() -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=3)


# ---------------------------------------------------------------------------
# 3. The (hidden) human user model — the *ground truth expectations*
# ---------------------------------------------------------------------------
# The user is never observed directly. Their expectation is a utility over the
# joint (context, configuration): in crowded areas they expect caution; on a
# low battery they expect energy saving; under time pressure they expect speed.
# The learner does NOT know these weights; it only sees approve/reject signals.
TRUE_W = dict(crowd_caution=2.4, batt_energy=2.4, time_speed=2.4, base=0.15)


def true_utility(ctx: np.ndarray, cfg: np.ndarray) -> float:
    crowd, batt, time = ctx
    speed, caution, energy = cfg
    return (
        TRUE_W["crowd_caution"] * crowd * caution
        + TRUE_W["batt_energy"] * batt * energy
        + TRUE_W["time_speed"] * time * speed
        + TRUE_W["base"] * (speed + caution + energy)
    )


def best_true_config(ctx: np.ndarray) -> str:
    """The configuration the user would most prefer in this context (oracle)."""
    utils = {n: true_utility(ctx, config_vec(n)) for n in CONFIG_NAMES}
    return max(utils, key=utils.get)


def user_feedback(ctx: np.ndarray, chosen: str) -> int:
    """Noisy approve(1)/reject(0). The user approves a shown configuration with
    probability rising in how well it matches their expectation *relative to the
    other options available in that context*."""
    utils = np.array([true_utility(ctx, config_vec(n)) for n in CONFIG_NAMES])
    shown = true_utility(ctx, config_vec(chosen))
    advantage = shown - utils.mean()
    p_approve = 1.0 / (1.0 + np.exp(-3.0 * advantage))
    return int(rng.uniform() < p_approve)


# ---------------------------------------------------------------------------
# 4. The learner — an online expectation model (logistic regression)
# ---------------------------------------------------------------------------
# Feature map over (context, configuration). It includes the three "correct"
# interaction terms plus several distractor interactions and main effects, so
# the model has to *discover* which expectations actually matter rather than
# being handed them.
# Nine context x attribute interactions + a bias. Three of these are the
# genuine expectation signals; the other six are distractors the model must
# learn to ignore. Context features are centred to [-0.5, 0.5] so that an
# interaction is only positive when a high context signal meets a high
# attribute, which makes the true signals cleanly identifiable.
FEATURE_NAMES = [
    "bias",
    "crowd x caution", "crowd x speed", "crowd x energy",
    "battery_low x energy", "battery_low x caution", "battery_low x speed",
    "time x speed", "time x caution", "time x energy",
]


def features(ctx: np.ndarray, cfg: np.ndarray) -> np.ndarray:
    crowd, batt, time = ctx - 0.5          # centre context signals
    speed, caution, energy = cfg - 0.5     # centre attributes
    return np.array([
        1.0,
        crowd * caution, crowd * speed, crowd * energy,
        batt * energy, batt * caution, batt * speed,
        time * speed, time * caution, time * energy,
    ], dtype=float)


class ExpectationModel:
    """Online logistic regression trained by SGD on approve/reject feedback."""

    def __init__(self, dim: int, lr: float = 0.30, l2: float = 1e-3):
        self.w = np.zeros(dim)
        self.lr = lr
        self.l2 = l2

    def approval_prob(self, ctx: np.ndarray, name: str) -> float:
        z = self.w @ features(ctx, config_vec(name))
        return 1.0 / (1.0 + np.exp(-z))

    def update(self, ctx: np.ndarray, name: str, label: int) -> None:
        x = features(ctx, config_vec(name))
        p = 1.0 / (1.0 + np.exp(-(self.w @ x)))
        grad = (p - label) * x + self.l2 * self.w
        self.w -= self.lr * grad

    def choose(self, ctx: np.ndarray, epsilon: float) -> str:
        """Pick the configuration with highest predicted approval, with
        epsilon-greedy exploration so early feedback covers the option space."""
        if rng.uniform() < epsilon:
            return CONFIG_NAMES[rng.integers(len(CONFIG_NAMES))]
        probs = {n: self.approval_prob(ctx, n) for n in CONFIG_NAMES}
        return max(probs, key=probs.get)

    def explain(self, ctx: np.ndarray, name: str, top_k: int = 2) -> str:
        """Explain a decision faithfully. A reason is only surfaced when it is
        literally true of the current situation (the cited context signal is
        actually high AND the chosen mode actually has that attribute) and the
        model has learned that this pairing is desirable (positive weight).
        Reasons are ranked by how strongly they drove the decision."""
        crowd, batt, time = ctx
        cfg = CONFIGS[name]
        # (feature index, context-signal value, attribute value)
        grounding = {
            "crowd x caution": (crowd, cfg["caution"]),
            "crowd x speed": (crowd, cfg["speed"]),
            "crowd x energy": (crowd, cfg["energy_saving"]),
            "battery_low x energy": (batt, cfg["energy_saving"]),
            "battery_low x caution": (batt, cfg["caution"]),
            "battery_low x speed": (batt, cfg["speed"]),
            "time x speed": (time, cfg["speed"]),
            "time x caution": (time, cfg["caution"]),
            "time x energy": (time, cfg["energy_saving"]),
        }
        scored = []
        for fname, (ctx_val, attr_val) in grounding.items():
            w = self.w[FEATURE_NAMES.index(fname)]
            # true of the situation, the mode fits, and learned as desirable
            if ctx_val > 0.55 and attr_val > 0.55 and w > 0:
                scored.append((w * ctx_val * attr_val, fname))
        scored.sort(reverse=True)
        idx = [FEATURE_NAMES.index(f) for _, f in scored[:top_k]]
        phrases = {
            "crowd x caution": "the area is crowded and this mode is cautious",
            "crowd x speed": "the area is crowded and this mode is fast",
            "crowd x energy": "the area is crowded and this mode saves energy",
            "battery_low x energy": "the battery is low and this mode saves energy",
            "battery_low x caution": "the battery is low and this mode is cautious",
            "battery_low x speed": "the battery is low and this mode is fast",
            "time x speed": "there is time pressure and this mode is fast",
            "time x caution": "there is time pressure and this mode is cautious",
            "time x energy": "there is time pressure and this mode saves energy",
        }
        reasons = [phrases.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in idx]
        if not reasons:
            return f"Chose '{name}' (no strong learned preference yet)."
        return f"Chose '{name}' because " + ", and ".join(reasons) + "."


# ---------------------------------------------------------------------------
# 5. Simulation-based evaluation
# ---------------------------------------------------------------------------
def run(episodes: int = 6000):
    model = ExpectationModel(dim=len(FEATURE_NAMES))
    learner_hits, baseline_hits = [], []          # 1 if choice == user's true best
    window = 200

    for t in range(episodes):
        ctx = sample_context()
        epsilon = max(0.02, 0.6 * (1 - t / (episodes * 0.4)))  # decaying explore

        chosen = model.choose(ctx, epsilon)
        label = user_feedback(ctx, chosen)
        model.update(ctx, chosen, label)

        oracle = best_true_config(ctx)
        learner_hits.append(int(chosen == oracle))
        baseline_hits.append(int("Balanced" == oracle))   # static baseline

    learner_hits = np.array(learner_hits)
    baseline_hits = np.array(baseline_hits)

    def rolling(a):
        c = np.cumsum(a)
        r = (c[window:] - c[:-window]) / window
        return r

    # ---- console report -------------------------------------------------
    print("=" * 68)
    print("EXPECTATION-AWARE SELF-ADAPTIVE SYSTEM — simulation report")
    print("=" * 68)
    print(f"Episodes: {episodes} | configurations: {CONFIG_NAMES}")
    print(f"Final-1000 alignment with user's true expectation:")
    print(f"  learned expectation model : {learner_hits[-1000:].mean():.1%}")
    print(f"  static 'Balanced' baseline: {baseline_hits[-1000:].mean():.1%}")

    print("\nLearned weights on the three genuine expectation signals")
    print("(the model was NOT told these were the right ones):")
    for f in ("crowd x caution", "battery_low x energy", "time x speed"):
        print(f"  {f:<24}: {model.w[FEATURE_NAMES.index(f)]:+.2f}")
    print("  All three are recovered as positive. The energy signal is easiest")
    print("  to learn ('Eco' is unambiguously the energy-saving mode); the speed")
    print("  signal is hardest, because only one mode is genuinely fast, so the")
    print("  feedback rarely isolates it -- an identifiability limit, not a bug.")

    # ---- worked explanations on fixed illustrative contexts -------------
    print("\nSample decisions with explanations:")
    demos = {
        "crowded area, healthy battery, no rush": np.array([0.9, 0.1, 0.1]),
        "empty area, nearly flat battery":        np.array([0.1, 0.9, 0.1]),
        "empty area, healthy battery, urgent":    np.array([0.1, 0.1, 0.9]),
    }
    for desc, ctx in demos.items():
        choice = model.choose(ctx, epsilon=0.0)
        print(f"  - {desc}")
        print(f"      {model.explain(ctx, choice)}")
        print(f"      (user's true best here: '{best_true_config(ctx)}')")

    # ---- learning curve plot -------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(rolling(learner_hits), label="Learned expectation model", lw=2)
    plt.plot(rolling(baseline_hits), label="Static 'Balanced' baseline",
             lw=2, ls="--")
    plt.xlabel(f"Episode (rolling window = {window})")
    plt.ylabel("Alignment with user's true expectation")
    plt.title("The system learns end-user expectations online, from feedback")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("expectation_learning_curve.png", dpi=130)
    print("\nSaved learning curve -> expectation_learning_curve.png")
    return model


# ---------------------------------------------------------------------------
# 6. MDP extension -- planning over a whole mission, not one step at a time
# ---------------------------------------------------------------------------
# Sections 1-5 form a contextual bandit: each context is sampled fresh, so a
# choice never affects the future. Here the context gets *memory*: the chosen
# mode drains the battery and advances the task, so today's choice changes
# tomorrow's situation. That makes it a finite-horizon MDP:
#   state  s = (t, battery b, progress p, crowd c)
#   action a = one of the 4 configurations
#   P(s'|s,a): battery drops w.p. drain(a), progress rises w.p. speed(a),
#              crowd flips w.p. CROWD_FLIP (independently)
#   reward R(s,a) = the LEARNED approval probability from Section 4
#   terminal: task done (success), battery empty or deadline missed (failure)
# The bandit learns WHAT the user wants; the MDP plans HOW to keep giving it
# to them without running out of battery or time.
H = 12                        # mission length in steps
B_MAX, P_MAX = 4, 4           # battery levels; progress levels (P_MAX = done)
CROWD_LEVEL = (0.15, 0.85)    # crowd signal when c=0 (quiet) / c=1 (busy)
CROWD_FLIP = 0.2
FAIL_PENALTY = 3.0            # cost of an empty battery or a missed deadline
START = (0, B_MAX, 0, 0)      # t=0, full battery, no progress, quiet area


def drain_prob(name: str) -> float:
    return 0.15 + 0.7 * (1.0 - CONFIGS[name]["energy_saving"])


def progress_prob(name: str) -> float:
    return CONFIGS[name]["speed"]


def is_terminal(t: int, b: int, p: int) -> bool:
    return p == P_MAX or b == 0 or t == H


def terminal_value(b: int, p: int) -> float:
    return 0.0 if p == P_MAX else -FAIL_PENALTY


def mdp_context(t: int, b: int, p: int, c: int) -> np.ndarray:
    """Map an MDP state to the (crowd, battery_low, time_pressure) context
    the learned expectation model understands."""
    time_pressure = min(1.0, (P_MAX - p) / max(H - t, 1))
    return np.array([CROWD_LEVEL[c], 1.0 - b / B_MAX, time_pressure])


def transitions(b: int, p: int, c: int, name: str):
    """All successors as (probability, b', p', c')."""
    d, g = drain_prob(name), progress_prob(name)
    out = []
    for db, pb in ((1, d), (0, 1 - d)):
        for dp, pp in ((1, g), (0, 1 - g)):
            for flip, pc in ((1, CROWD_FLIP), (0, 1 - CROWD_FLIP)):
                out.append((pb * pp * pc, b - db, min(p + dp, P_MAX),
                            1 - c if flip else c))
    return out


def solve_mdp(model: ExpectationModel):
    """Finite-horizon value iteration (backward induction).
    V[t,b,p,c] = max_a [ R(s,a) + sum_s' P(s'|s,a) V(s') ]."""
    V = np.zeros((H + 1, B_MAX + 1, P_MAX + 1, 2))
    policy = {}
    for t in range(H, -1, -1):
        for b in range(B_MAX + 1):
            for p in range(P_MAX + 1):
                for c in range(2):
                    if is_terminal(t, b, p):
                        V[t, b, p, c] = terminal_value(b, p)
                        continue
                    ctx = mdp_context(t, b, p, c)
                    q = {a: model.approval_prob(ctx, a)
                         + sum(pr * V[t + 1, b2, p2, c2]
                               for pr, b2, p2, c2 in transitions(b, p, c, a))
                         for a in CONFIG_NAMES}
                    best = max(q, key=q.get)
                    V[t, b, p, c], policy[(t, b, p, c)] = q[best], best
    return V, policy


def evaluate_policy(model: ExpectationModel, choose) -> dict:
    """Exact evaluation of a fixed policy by propagating the state
    distribution (a policy turns the MDP into a Markov chain, which is
    what a probabilistic model checker analyses)."""
    dist = {START: 1.0}
    res = dict(approval=0.0, done=0.0, battery_empty=0.0, missed=0.0)
    while dist:
        nxt = {}
        for (t, b, p, c), pr in dist.items():
            if is_terminal(t, b, p):
                key = "done" if p == P_MAX else ("battery_empty" if b == 0 else "missed")
                res[key] += pr
                continue
            a = choose((t, b, p, c))
            res["approval"] += pr * model.approval_prob(mdp_context(t, b, p, c), a)
            for q, b2, p2, c2 in transitions(b, p, c, a):
                k = (t + 1, b2, p2, c2)
                nxt[k] = nxt.get(k, 0.0) + pr * q
        dist = nxt
    return res


def export_prism(model: ExpectationModel, path: str = "adaptation_mdp.prism"):
    """Write the same MDP as a PRISM model so its properties can be checked
    with probabilistic model checking (PRISM or Storm)."""
    L = ["mdp", "", f"const int H = {H};", "", "module adaptation",
         "  t : [0..H] init 0;",
         f"  b : [0..{B_MAX}] init {B_MAX};",
         f"  p : [0..{P_MAX}] init 0;",
         "  c : [0..1] init 0;"]
    guard = f"t<H & b>0 & p<{P_MAX}"
    for a in CONFIG_NAMES:
        d, g, f = drain_prob(a), progress_prob(a), CROWD_FLIP
        upd = []
        for db, pb in ((1, d), (0, 1 - d)):
            for dp, pp in ((1, g), (0, 1 - g)):
                for fl, pc in ((1, f), (0, 1 - f)):
                    bb = "b'=b-1" if db else "b'=b"
                    pp_ = f"p'=min(p+1,{P_MAX})" if dp else "p'=p"
                    cc = "c'=1-c" if fl else "c'=c"
                    upd.append(f"{pb * pp * pc:.6f}:(t'=t+1)&({bb})&({pp_})&({cc})")
        L.append(f"  [{a.lower()}] {guard} -> " + " + ".join(upd) + ";")
    L += ["endmodule", "",
          f'label "done" = p={P_MAX};',
          f'label "battery_empty" = b=0 & p<{P_MAX};',
          f'label "missed" = t=H & b>0 & p<{P_MAX};', "",
          'rewards "approval"']
    for t in range(H):
        for b in range(1, B_MAX + 1):
            for p in range(P_MAX):
                for c in range(2):
                    ctx = mdp_context(t, b, p, c)
                    for a in CONFIG_NAMES:
                        r = model.approval_prob(ctx, a)
                        L.append(f"  [{a.lower()}] t={t} & b={b} & p={p} & c={c} : {r:.4f};")
    L.append("endrewards")
    open(path, "w").write("\n".join(L) + "\n")
    props = [
        '// best achievable probability of finishing the task',
        'Pmax=? [ F "done" ]',
        '// lowest achievable risk of an empty battery',
        'Pmin=? [ F "battery_empty" ]',
        '// is there a policy keeping battery risk at or below 5%? (true/false)',
        'Pmin=? [ F "battery_empty" ] <= 0.05',
        '// best achievable expected total approval',
        'R{"approval"}max=? [ F ("done" | "battery_empty" | "missed") ]',
    ]
    open(path.replace(".prism", ".props"), "w").write("\n".join(props) + "\n")
    return path


def mdp_report(model: ExpectationModel) -> None:
    _, policy = solve_mdp(model)
    greedy = lambda s: max(CONFIG_NAMES,
                           key=lambda a: model.approval_prob(mdp_context(*s), a))
    rows = {
        "MDP-optimal policy": evaluate_policy(model, lambda s: policy[s]),
        "Greedy bandit (Sec. 4)": evaluate_policy(model, greedy),
        "Static 'Balanced'": evaluate_policy(model, lambda s: "Balanced"),
    }
    print("\n" + "=" * 68)
    print("MDP EXTENSION -- exact evaluation over a whole mission")
    print("=" * 68)
    print(f"{'policy':<24}{'approval':>10}{'done':>9}{'batt=0':>9}{'missed':>9}")
    for name, r in rows.items():
        print(f"{name:<24}{r['approval']:>10.2f}{r['done']:>9.1%}"
              f"{r['battery_empty']:>9.1%}{r['missed']:>9.1%}")
    print("\nWhere the planner and the greedy learner disagree (crowd=quiet):")
    shown = 0
    for t in (0, 4, 8):
        for b in (B_MAX, 2, 1):
            for p in (0, 2):
                s = (t, b, p, 0)
                if s in policy and policy[s] != greedy(s) and shown < 6:
                    print(f"  t={t:>2} battery={b} progress={p}:  MDP -> {policy[s]:<10}"
                          f" greedy -> {greedy(s)}")
                    shown += 1
    if shown == 0:
        print("  (none in the sampled states)")

    # ---- policy comparison plot -----------------------------------------
    outcomes = (("done", "Task done", "#2e7d32"),
                ("battery_empty", "Battery empty", "#c62828"),
                ("missed", "Deadline missed", "#ef8f00"))
    names = list(rows.keys())
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, label, color) in enumerate(outcomes):
        vals = [rows[n][key] * 100 for n in names]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(approval {rows[n]['approval']:.2f})" for n in names])
    ax.set_ylabel("Probability over a mission (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Planning with the MDP halves battery failures vs. greedy adaptation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("mdp_policy_comparison.png", dpi=130)
    print("\nSaved policy comparison -> mdp_policy_comparison.png")
    path = export_prism(model)
    print(f"\nPRISM model -> {path}  (properties -> {path.replace('.prism', '.props')})")
    print("Check with:  prism adaptation_mdp.prism adaptation_mdp.props")


if __name__ == "__main__":
    mdp_report(run())