import os
import joblib
import json
import colorama
os.environ['PRADA_DATA_DIR']='/cw/dtaijupiter/NoCsBack/dtai/timo/prada_data'
import prada
import veritas
import numpy as np
import pandas as pd

from datetime import datetime
from dataclasses import dataclass

SEED = 5823

TIMEOUT = 30*60
NUM_ADV_EX = 500

LBRACE = "{"
RBRACE = "}"

NFOLDS = 5
DNAMES = [
    "Img[bin]",
    "Electricity",
    #"CovtypeNumeric",
    #"Covtype",
    "MagicTelescope",
    "BankMarketing",
    "Bioresponse",
    "MiniBooNE",
    "DefaultCreditCardClients",
    "EyeMovements",
    "Diabetes130US",
    "Jannis",
    "Heloc",
    "Credit",
    "California",
    "Albert",
    "CompasTwoYears",
    "RoadSafety",
    #"AtlasHiggs",
    "SantanderCustomerSatisfaction",
    #"BreastCancer",
    "Vehicle",
    "Spambase",
    "Phoneme",
    "Nomao",
    "Banknote",
    "Adult",
    "Ijcnn1",
    "Webspam",
    "Mnist[2v4]",
    "FashionMnist[2v4]",
    "Houses[bin]",
    "CpuAct[bin]",
    "MercedesBenzManufacturing[bin]",
    "BikeSharingDemand[bin]",
    "Yprop41[bin]",
    "Abalone[bin]",
    "DryBean[6vRest]",
    "Volkert[2v7]",
    "Seattlecrime6[bin]"
    #"HiggsBig",
    #"KddCup99",
]

DNAMES_REGRESSION = [
    "WineQuality",
    "Houses",
    "Ailerons",
    "Abalone",
    "CpuSmall",
    "Elevators",
    "House16H"
]

DNAMES_SUB = [
    "Electricity",
    "MagicTelescope",
    "MiniBooNE",
    "EyeMovements",
    "Jannis",
    "Heloc",
    "Credit",
    "California",
    "Albert",
    "CompasTwoYears",
    "Vehicle",
    "Spambase",
    "Phoneme",
    "Nomao",
    "Adult",
    "Ijcnn1",
    "Mnist[2v4]",
    "Houses[bin]",
    "DryBean[6vRest]",
    "Volkert[2v7]",
]

DNAMES_SUBSUB = [
    "Electricity",
    "MiniBooNE",
    "Jannis",
    "Credit",
    "California",
    "CompasTwoYears",
    "Vehicle",
    "Spambase",
    "Phoneme",
    "Adult",
    "Ijcnn1",
    "Mnist[2v4]",
    "DryBean[6vRest]",
    "Volkert[2v7]",
]

DNAMES_SUBSUBSUB = [
    "California",
    "Adult",
    "Spambase",
    "Phoneme"
]


def get_dataset(dname, seed, fold, silent):
    d = prada.get_dataset(dname, seed=seed, silent=silent)
    d.load_dataset()
    d.robust_normalize()
    #d.transform_target()
    d.scale_target()
    d.astype(veritas.FloatT)

    if d.is_binary():
        d.use_balanced_accuracy()

    dtrain, dtest = d.train_and_test_fold(fold, nfolds=NFOLDS)
    dtrain, dvalid = dtrain.split(0, nfolds=NFOLDS-1)

    return d, dtrain, dvalid, dtest


@dataclass
class HyperParamResult:
    at: object
    train_time: float
    params: dict

    mtrain: float
    mvalid: float
    mtest: float

    nleafs: int
    nnzleafs: int

def pareto_front(models, mkey="mvalid", skey="nnzleafs"):
    """
    1. Let 𝑖:=1
    2. Add 𝐴𝑖 to the Pareto frontier.
    3. Find smallest 𝑗>𝑖 such that value(𝐴𝑗)>value(𝐴𝑖).
    4. If no such 𝑗 exists, stop. Otherwise let 𝑖:=𝑗 and repeat from step 2.

    https://math.stackexchange.com/a/101141
    """
    models.sort(key=lambda m: m[mkey], reverse=True)
    models.sort(key=lambda m: m[skey]) # stable sort
    n = len(models)
    onfront = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        onfront[i] = True
        j = n
        for k in range(i+1, n):
            if models[k][mkey] > models[i][mkey]:
                j = k
                break
        if j < n:
            i = j
        else:
            break

    return onfront

def pareto_front_xy(x, y):
    yperm = np.argsort(y)[::-1]
    x = x[yperm]
    y = y[yperm]
    xperm = np.argsort(x, kind="stable")
    x = x[xperm]
    y = y[xperm]
    perm = yperm[xperm]

    n = len(x)
    onfront = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        onfront[i] = True
        j = n
        for k in range(i+1, n):
            if y[k] > y[i]:
                j = k
                break
        if j < n:
            i = j
        else:
            break

    # Check if on convex hull
    from scipy.spatial import ConvexHull
    xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    ch = ConvexHull(xy)
    onhull = np.zeros_like(onfront)
    onhull[ch.vertices] = True
    onhull &= onfront

    onfront_inv_perm = np.zeros_like(onfront)
    onfront_inv_perm[perm] = onfront
    onhull_inv_perm = np.zeros_like(onfront)
    onhull_inv_perm[perm] = onhull
    return onfront_inv_perm, onhull_inv_perm


def convex_hull_front_xy(xy):
    from scipy.spatial import ConvexHull
    points = np.vstack([
        xy,
        [xy[:,0].max(), xy[:,1].max()],
        [0.0, 0.0]
    ])
    ch = ConvexHull(points)
    hullv = [x for x in ch.vertices if x < xy.shape[0]]

    return xy[hullv, :]




def plot_pareto_front(ax, models, compr_models=None):
    from scipy.spatial import ConvexHull

    onfront = pareto_front(models, mkey="mtest", skey="nnzleafs")
    models_onfront = [m for b, m in zip(onfront, models) if b]

    # Add a point to the convex hull points to force it to the bottom right corner,
    # then remove it again
    hullpoints = np.array([[m["nnzleafs"], m["mtest"]] for m in models_onfront])
    hullpoints = np.vstack([hullpoints, [hullpoints[:,0].max(), hullpoints[:,1].min()]])
    if hullpoints.shape[0] > 2:
        hullv = sorted(ConvexHull(hullpoints).vertices)
        hullv.remove(onfront.sum())
    else:
        hullv = None

    xs = np.array([m["nnzleafs"] for m in models])
    ys = np.array([m["mtest"] for m in models])
    x0, y0 = max(xs), min(ys)

    ax.scatter(xs, ys, c=onfront.astype(float), s=10*(onfront.astype(float)+1))
    #ax.invert_yaxis()
    ax.set_xlabel("model size")
    ax.set_ylabel("bal. acc. test set")
    ax.set_xscale("log")
    #yticks = ax.get_yticks()
    #ax.set_yticks(yticks)
    #ax.set_yticklabels([f"{1.0-x:.3f}" for x in yticks])

    if hullv:
        ax.plot(xs[onfront][hullv], ys[onfront][hullv], c="gray", ls=":", lw=1)
        ax.plot([xs[onfront][hullv[0]]]*2, [y0, ys[onfront][hullv[0]]], c="gray", ls=":", lw=1)
        ax.plot([x0, xs[onfront][hullv[-1]]], [ys[onfront][hullv[-1]]]*2, c="gray", ls=":", lw=1)

    if compr_models is not None:
        pareto_models = {
            params_hash(m["params"]): m for m in models if m["on_pareto_front"]
        }

        for m in compr_models:
            h = params_hash(m["params"])
            tm = pareto_models[h]

            xs = [tm["nnzleafs"], m["nnzleafs"]]
            ys = [tm["mtest"], m["mtest"]]
            print("--->", xs, np.round(ys, 3))
            ax.plot(xs, ys, color="red", lw=1)
            ax.plot([xs[1]], [ys[1]], "x", color="red")

def plot_pareto_fronts(dname, train_results, compr_results):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots()

    trs = []
    cos = []

    for i, (params_hash, folds) in enumerate(compr_results.items()):
        trfolds = train_results[params_hash]
        fold_indexes = list(trfolds.keys() & folds.keys())
        tr = np.zeros((len(fold_indexes), 2))
        co = np.zeros((len(fold_indexes), 2))

        on_any_pareto_front = all([m["on_any_pareto_front"] for m in trfolds.values()])
        if not on_any_pareto_front:
            continue

        if len(fold_indexes) != NFOLDS:
            print(f"missing folds for {dname}: {fold_indexes}",
                  [m["on_pareto_front"] for m in trfolds.values()])
            continue

        for fold in fold_indexes:
            trm = trfolds[fold]
            com = folds[fold]
            tr[fold, :] = [trm["nleafs"], trm["mtest"]]
            co[fold, :] = [com["nleafs"], com["mtest"], com["verification"]["compr"]["nocs_timeout"]]

        trs.append(tr)
        cos.append(co)

        #ax.scatter(tr[:, 0], tr[:, 1], marker=".", c="black", s=5, alpha=0.1)
        #ax.scatter(tr[:, 0].mean(), tr[:, 1].mean(), marker=".", c="lightgray", s=10)
        #ax.scatter(co[:, 0].mean(), co[:, 1].mean(), marker="d", c="lightgray", s=10)
        #ax.plot([tr[:, 0].mean(), co[:, 0].mean()],
        #        [tr[:, 1].mean(), co[:, 1].mean()], ls="-", color="lightgray", lw=0.5)

    tr_mn = np.array([np.mean(tr, axis=0) for tr in trs])
    tr_std = np.array([np.std(tr, axis=0) for tr in trs])
    co_mn = np.array([np.mean(co, axis=0) for co in cos])
    co_std = np.array([np.std(co, axis=0) for co in cos])

    #tr_mn = np.array([tr[fold, :] for tr in trs])
    #co_mn = np.array([co[fold, :] for co in cos])

    # Front without compression
    onfront, onhull = pareto_front_xy(tr_mn[:, 0], tr_mn[:, 1])
    #chull = util.convex_hull_front_xy(tr_mn[onfront])
    #chull = chull[chull[:, 0].argsort(), :]
    pareto = tr_mn[onfront, :]
    pareto = np.hstack(
        (
            pareto,
            #pareto[:, 1:2] - np.convolve(tr_std[onfront, 1], np.ones(5)/5, "same").reshape(-1, 1),
            #pareto[:, 1:2] + np.convolve(tr_std[onfront, 1], np.ones(5)/5, "same").reshape(-1, 1),
            pareto[:, 1:2] - 0.5*tr_std[onfront, 1:2],
            pareto[:, 1:2] + 0.5*tr_std[onfront, 1:2]
        )
    )
    pareto = pareto[pareto[:, 0].argsort(), :]

    lxgb, = ax.plot(pareto[:,0], pareto[:, 1], "-o", color="navy", ms=3)
    #ax.plot(pareto[:,0], pareto[:, 2], color="navy", lw=1, ls=":")
    #ax.plot(pareto[:,0], pareto[:, 3], color="navy", lw=1, ls=":")
    ax.fill_between(pareto[:,0], pareto[:, 2], pareto[:, 3], fc="navy", alpha=0.1)

    # HOW MANY compressed are not dominated by xgb front?
    compr_not_dominated = np.zeros(co_mn.shape[0], dtype=bool)
    for i in range(co_mn.shape[0]):   #   smaller              OR    better performance
        front_smaller = pareto[:, 0] < co_mn[i, 0]
        front_accurater = pareto[:, 1] > co_mn[i, 1]
        is_dominated = any(front_smaller & front_accurater)
        is_not_dominated = not is_dominated
        is_not_dominated |= all(co_mn[i, 0] <= pareto[:, 0])
        is_not_dominated |= all(co_mn[i, 1] >= pareto[:, 1])
        compr_not_dominated[i] = is_not_dominated

    ax.scatter(
        tr_mn[:, 0],
        tr_mn[:, 1],
        marker=".",
        c=["lightgray" if b else "red" for b in compr_not_dominated],
        s=10,
    )
    ax.scatter(
        co_mn[:, 0],
        co_mn[:, 1],
        marker="d",
        c=["lightgray" if b else "red" for b in compr_not_dominated],
        s=10,
    )
    for i in range(co_mn.shape[0]):
        ax.plot(
            [tr_mn[i, 0], co_mn[i, 0]],
            [tr_mn[i, 1], co_mn[i, 1]],
            c="lightgray" if compr_not_dominated[i] else "red",
            lw=0.5, alpha=0.5,
        )

    # Front for compressed
    onfront, onhull = pareto_front_xy(co_mn[:, 0], co_mn[:, 1])
    pareto = co_mn[onfront, :]
    pareto = np.hstack(
        (
            pareto,
            #pareto[:, 1:2] - np.convolve(co_std[onfront, 1], np.ones(5)/5, "same").reshape(-1, 1),
            #pareto[:, 1:2] + np.convolve(co_std[onfront, 1], np.ones(5)/5, "same").reshape(-1, 1),
            pareto[:, 1:2] - 0.5*co_std[onfront, 1:2],
            pareto[:, 1:2] + 0.5*co_std[onfront, 1:2]
        )
    )
    pareto = pareto[pareto[:, 0].argsort(), :]

    lcompr, = ax.plot(pareto[:,0], pareto[:, 1], "-d", color="green", ms=3)
    #ax.plot(pareto[:,0], pareto[:, 2], color="green", lw=1, ls=":")
    #ax.plot(pareto[:,0], pareto[:, 3], color="green", lw=1, ls=":")
    ax.fill_between(pareto[:,0], pareto[:, 2], pareto[:, 3], fc="green", alpha=0.1)

    ## Front with compression
    #all_mn = np.vstack([tr_mn, co_mn])
    #is_compr = np.ones(all_mn.shape[0], dtype=bool)
    #is_compr[:tr_mn.shape[0]] = False
    #onfront = util.pareto_front_xy(all_mn[:, 0], all_mn[:, 1])
    #pareto = all_mn[onfront, :]
    #pareto = pareto[pareto[:, 0].argsort(), :]

    #lboth, = ax.plot(pareto[:,0], pareto[:, 1], "--", color="orange", ms="3")


    #goodness_score = num_not_dominated_over_folds[0] / num_not_dominated_over_folds[1]
    goodness_score = compr_not_dominated.mean()

    ax.set_xlabel("num. of leaves")
    ax.set_ylabel("bal. acc. test set")
    ax.set_xscale("log")
    ax.set_title(f"{dname}, SCORE {goodness_score*100:.1f}%")

    #ax.legend([lxgb, lcompr, lboth], ["Pareto XGB", "Pareto compressed", "Pareto both"])
    ax.legend([lxgb, lcompr], ["Pareto XGB", "Pareto compressed"])

    return fig, ax, goodness_score


def plot_it_nogekeer(dname, train_results, compr_results, nnz):
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.5, 10))
    fig.subplots_adjust(bottom=0.45, top=0.95)

    if nnz:
        ax.set_xlabel("num. non-zero leaves")
    else:
        ax.set_xlabel("num. of leaves")
    ax.set_ylabel("bal. acc. test set")
    ax.set_xscale("log")
    ax.set_title(f"{dname}")

    custom_lines = [Line2D([0], [0], marker="o", color="navy", lw=1, ms=4),
                    Line2D([0], [0], marker="o", color="green", lw=1, ms=4),
                    Line2D([0], [0], marker="o", color="orange", lw=1, ms=4),
                    Line2D([0], [0], ls=":", marker="x", color="gray",
                           markerfacecolor="red", markeredgecolor="red", lw=1, ms=7)]

    ax.legend(custom_lines, ["XGB", "Compressed", "LeafRefine+L1", "Non-enumerable"])

    trs = []
    cos = []
    ves = []

    param_hashes = train_results.keys() & compr_results.keys()

    for params_hash in param_hashes:
        trfolds = train_results[params_hash]
        try:
            folds005 = compr_results[params_hash]["005"]
            folds010 = compr_results[params_hash]["010"]
            foldslr0 = compr_results[params_hash]["lr_lrl1"]
        except KeyError:
            continue

        fold_indexes = list(trfolds.keys() & folds005.keys() & folds010.keys())
        tr = np.zeros((len(fold_indexes), 4))
        co = np.zeros((len(fold_indexes), 12))
        ve = np.zeros((len(fold_indexes), 9))

        if len(fold_indexes) != NFOLDS:
            print(f"missing folds for {dname}: {fold_indexes}")
            continue

        leafs = "nnzleafs" if nnz else "nleafs"

        for fold in fold_indexes:
            trm = trfolds[fold]
            com05 = folds005[fold]
            com10 = folds010[fold]
            comlr = foldslr0[fold]
            print(com05["verification"]["orig"]["nocs_time"],
                  com05["verification"]["compr"]["nocs_time"])
            tr[fold, :] = [
                trm[leafs],    # 0
                trm["mtest"],  # 1
                com05["verification"]["orig"]["nocs"],  # 2
                com05["verification"]["orig"]["nocs_timeout"],  # 3
            ]
            co[fold, :] = [
                com05[leafs],
                com05["mtest"],
                com05["verification"]["compr"]["nocs"],
                com05["verification"]["compr"]["nocs_timeout"],
                com10[leafs],
                com10["mtest"],
                com10["verification"]["compr"]["nocs"],
                com10["verification"]["compr"]["nocs_timeout"],
                comlr[leafs],
                comlr["mtest"],
                comlr["verification"]["refined"]["nocs"],
                comlr["verification"]["refined"]["nocs_timeout"],
            ]
            ve[fold, :] = [
                com05["verification"]["orig"]["exact_emp_rob"],
                com05["verification"]["compr"]["exact_emp_rob"],
                comlr["verification"]["refined"]["exact_emp_rob"],

                (com05["verification"]["orig"]["exact_emp_rob_time"] /
                com05["verification"]["orig"]["exact_emp_rob_n"]),
                (com05["verification"]["compr"]["exact_emp_rob_time"] /
                com05["verification"]["compr"]["exact_emp_rob_n"]),
                (comlr["verification"]["refined"]["exact_emp_rob_time"] /
                comlr["verification"]["refined"]["exact_emp_rob_n"]),

                (com05["verification"]["orig"]["approx_emp_rob_time"] /
                com05["verification"]["orig"]["approx_emp_rob_n"]),
                (com05["verification"]["compr"]["approx_emp_rob_time"] /
                com05["verification"]["compr"]["approx_emp_rob_n"]),
                (comlr["verification"]["refined"]["approx_emp_rob_time"] /
                comlr["verification"]["refined"]["approx_emp_rob_n"]),
            ]

        trs.append(tr)
        cos.append(co)
        ves.append(ve)

    if len(cos) == 0:
        print(f"skpping {dname}")
        return fig, ax, 0.0

    tr_mn = np.array([np.mean(tr, axis=0) for tr in trs])
    tr_std = np.array([np.std(tr, axis=0) for tr in trs])
    co_mn = np.array([np.mean(co, axis=0) for co in cos])
    co_std = np.array([np.std(co, axis=0) for co in cos])

    # Set nocs to max_nocs if timeout
    max_nocs = max(tr_mn[:, 2].max(), co_mn[:, 2].max()) * 10
    tr_mn[tr_mn[:, 3] > 0.9, 2] = max_nocs
    co_mn[co_mn[:, 3] > 0.9, 2] = max_nocs
    co_mn[co_mn[:, 11] > 0.9, 10] = max_nocs

    ve_df = pd.DataFrame(
        "-",
        index=range(tr_mn.shape[0]),
        columns=pd.MultiIndex.from_product(
            [["NumLeaf", "BalAccTest", "EmpRobust", "KantchelianTime", "VeritasTime"],
             ["XGB", "Compr", "LRL1"]]
        ),
    )
    ve_mn = np.array([np.mean(ve, axis=0) for ve in ves])
    ve_df.loc[:, ("NumLeaf", "XGB")] = tr_mn[:, 0].round(0).astype(int)
    ve_df.loc[:, ("NumLeaf", "Compr")] = co_mn[:, 0].round(0).astype(int)
    ve_df.loc[:, ("NumLeaf", "LRL1")] = co_mn[:, 8].round(0).astype(int)
    ve_df.loc[:, ("NOCs", "XGB")] = [f"{n:.0f}" if n!=max_nocs else "NA" for n in tr_mn[:, 2]]
    ve_df.loc[:, ("NOCs", "Compr")] = [f"{n:.0f}" if n!=max_nocs else "NA" for n in co_mn[:, 2]]
    ve_df.loc[:, ("NOCs", "LRL1")] = [f"{n:.0f}" if n!=max_nocs else "NA" for n in co_mn[:, 10]]
    ve_df.loc[:, ("BalAccTest", "XGB")] = (100*tr_mn[:, 1]).round(1)
    ve_df.loc[:, ("BalAccTest", "Compr")] = (100*co_mn[:, 1]).round(1)
    ve_df.loc[:, ("BalAccTest", "LRL1")] = (100*co_mn[:, 9]).round(1)
    ve_df.loc[:, ("EmpRobust", "XGB")] = ve_mn[:, 0].round(3)
    ve_df.loc[:, ("EmpRobust", "Compr")] = ve_mn[:, 1].round(3)
    ve_df.loc[:, ("EmpRobust", "LRL1")] = ve_mn[:, 2].round(3)
    ve_df.loc[:, ("KantchelianTime", "XGB")] = ve_mn[:, 3].round(2)
    ve_df.loc[:, ("KantchelianTime", "Compr")] = ve_mn[:, 4].round(2)
    ve_df.loc[:, ("KantchelianTime", "LRL1")] = ve_mn[:, 5].round(2)
    ve_df.loc[:, ("VeritasTime", "XGB")] = ve_mn[:, 6].round(3)
    ve_df.loc[:, ("VeritasTime", "Compr")] = ve_mn[:, 7].round(3)
    ve_df.loc[:, ("VeritasTime", "LRL1")] = ve_mn[:, 8].round(3)
    df = ve_df.sort_values(("NumLeaf", "Compr"), axis="index").reset_index(drop=True)

    fig.text(0.5, 0.02, df.to_string(), fontfamily="monospace", linespacing=1.5,
             ha="center", va="bottom", fontsize=8)

    print("VERIFICATION")
    print(ve_df.sort_values(("NumLeaf", "Compr"), axis="index").reset_index(drop=True))

    def plotline(m, e, ix, iy, im, c):
        ordr = m[:, ix].argsort()
        ax.plot(m[ordr, ix], m[ordr, iy], "-o", c=c, lw=1.0, ms=4)
        ax.errorbar(m[ordr, ix], m[ordr, iy],
                    yerr=0.5*e[ordr, iy],
                    #xerr=e[ordr, ix],
                    alpha=0.5, zorder=-1,
                    fmt="-o", c=c, elinewidth=0.5, capsize=2, capthick=0.5)

        nonverif = m[:, im] > 0.9
        #print(m[:, [ix, im]])
        ax.plot(m[nonverif, ix], m[nonverif, iy], " ", marker="x", c="red", zorder=100, ms=7)

    # Plot number of leaves vs. performance
    #plotline(tr_mn, tr_std, 0, 1, 3, "navy")
    #plotline(co_mn, co_std, 0, 1, 3, "green")
    ##plotline(co_mn, co_std, 4, 5, 3, "orange")
    #plotline(co_mn, co_std, 8, 9, 11, "orange")

    # Plot number of ocs vs. performance
    plotline(tr_mn, tr_std, 2, 1, 3, "navy")
    plotline(co_mn, co_std, 2, 1, 3, "green")
    plotline(co_mn, co_std, 10, 9, 11, "orange")
    ax.axvline(x=max_nocs, ls=":", color="gray", lw=1.0)
    ax.set_xlabel("Number of OCs")

    mask = (ax.get_xticks() <= max_nocs/5) & (ax.get_xticks() >= 1)
    ax.set_xticks([t for b, t in zip(mask, ax.get_xticks()) if b])
    mask = (ax.get_xticks(minor=True) <= max_nocs/5) & (ax.get_xticks(minor=True) >= 1)
    ax.set_xticks([t for b, t in zip(mask, ax.get_xticks(minor=True)) if b], minor=True)

    for i in range(co_mn.shape[0]):
        ax.plot(
            #[tr_mn[i, 0], co_mn[i, 0]],
            [tr_mn[i, 2], co_mn[i, 2]],
            [tr_mn[i, 1], co_mn[i, 1]],
            c="gray",
            ls="--",
            lw=0.5, alpha=0.5, zorder=-1
        )
        #ax.plot(
        #    [tr_mn[i, 0], co_mn[i, 8]],
        #    [tr_mn[i, 1], co_mn[i, 9]],
        #    c="gray",
        #    ls="-.",
        #    lw=0.5, alpha=0.5, zorder=-1
        #)

    #ax.legend([lxgb, lcompr], ["Pareto XGB", "Pareto compressed"])

    return fig, ax, 0.0


def params_hash(params):
    import struct
    import hashlib

    keys = sorted(params.keys())
    h = hashlib.sha256()

    for k in keys:
        h.update(k.encode("ascii"))
        value = params[k]
        if isinstance(value, float):
            h.update(struct.pack("f", value))
        elif isinstance(value, int):
            h.update(struct.pack("i", value))
        elif isinstance(value, str):
            h.update(value.encode("ascii"))
        else:
            print("don't know how to hash")

    return h.hexdigest()

def nowstr():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

def read_json_printfile(fname):
    jsons = []
    with open(fname) as fh:
        for line in fh.readlines():
            if not line.startswith(LBRACE):
                continue

            j = json.loads(line)
            j["file"] = fname
            jsons.append(j)
    return jsons

def read_jsons():
    jsons = []
    for f in os.listdir("results/"):
        if not f.endswith(".txt"):
            continue
        f = os.path.join("results", f)
        if not os.path.isfile(f):
            continue
        jsons += read_json_printfile(f)
    return jsons

def get_or_insert(d, key, value_generator):
    if key in d:
        return d[key]
    else:
        d[key] = value_generator()
        return d[key]

def get_key(*args):
    if len(args) == 1:
        j = args[0]
        model_type = j["model_type"]
        linclf_type = j["linclf_type"]
        seed = j["seed"]
    else:
        model_type, linclf_type, seed = args
    return f"{model_type}-{linclf_type}-{seed}"


def read_hyperparams(fname):
    jsons = read_json_printfile(fname)
    params = {}
    for j in jsons:
        dname = j["dname"]
        fold = j["fold"]
        key = get_key(j)

        forkey = get_or_insert(params, key, lambda: {})
        fordname = get_or_insert(forkey, dname, lambda: {})
        fordname[fold] = j["params"]

    return params


def write_train_results(hyperparams):
    joblib.dump(hyperparams, "processed_results/train.joblib", compress=9)


def load_train_results():
    return joblib.load("processed_results/train.joblib")


def get_method_name(*args):
    if len(args) == 1:
        j = args[0]
        cmd = j["cmd"]
        model_type = j["model_type"]
        paramset = j["paramset"] if cmd == "train" else ""
        abserr = j["abserr"] if cmd == "compress" else 0.01
    else:
        cmd, model_type, paramset, abserr = args
    suffix = f"compr{abserr*1000:04.0f}" if cmd == "compress" else paramset
    return f'{model_type}_{suffix}'


def read_results(jsons=None):
    if  jsons is None:
        jsons = read_jsons()

    # Organize by:
    # - dname
    # - seed
    # - abserr (0.005, 0.01, 0.02)
    # - fold (0, 1, ..., 4)
    # - model_type
    #    + xgb         rf         dt           J48 (WEKA)       
    #    + xgb.compr   rf.compr   dt.compr     J48.compr?   (per abserr)
    #    + xgb.small   rf.small   dt.small     J48.small
    #
    # Big dict:
    # [seed][model_type] = DataFrame
    #   with multi-index
    #     - dname x fold
    #   with columns
    #     - mtrain
    #     - mvalid
    #     - mtest
    #     - nnodes
    #     - nleaves
    #     - nnz_leaves
    #     - train / compress time
    #
    # compression specific columns
    #     - abserr
    #     - linclf_type

    organized = {}
    for j in jsons:
        dname = j["dname"]
        fold = j["fold"]
        key = get_key(j)
        method_name = get_method_name(j)

        index = pd.MultiIndex.from_product(
            [DNAMES, list(range(NFOLDS))], names=["dname", "fold"]
        )
        columns = ["mtrain", "mvalid", "mtest", "ntrees", "nnodes", "nnzleafs", "time"]
        formethod = get_or_insert(organized, method_name, lambda: {})
        df = get_or_insert(
            formethod, key, lambda: pd.DataFrame(np.nan, index=index, columns=columns)
        )

        for k in columns:
            try:
                df.loc[(dname, fold), k] = j[k]
            except KeyError:
                print("KEY ERROR", k, method_name, "in", j["file"])
                continue

    return organized

def write_compress_results(organized):
    joblib.dump(organized, "processed_results/compress.joblib")


def load_compress_results():
    return joblib.load("processed_results/compress.joblib")


def print_metrics(lab, m0, m1, abserr):
    RST = colorama.Style.RESET_ALL
    RED = colorama.Fore.RED
    GRN = colorama.Fore.GREEN

    s = " " * len(lab)
    m0m = np.mean(m0)
    m1m = np.mean(m1)
    m0s = np.std(m0)
    m1s = np.std(m1)
    diff = m1m-m0m
    clr = RED if diff > abserr else GRN
    print(lab, " ".join(f"{x:.3f}" for x in m0), "|", f"{m0m:.3f}±{m0s:.3f}")
    print(s,   " ".join(f"{x:.3f}" for x in m1), "|", f"{m1m:.3f}±{m1s:.3f}", f"{clr}{diff:.3f}{RST}")

def print_nnzleafs(lab, nnz0, nnz1):
    s = " " * len(lab)
    m0m = np.mean(nnz0)
    m1m = np.mean(nnz1)
    m0s = np.std(nnz0)
    m1s = np.std(nnz1)
    m1r = [x/y for x, y in zip(nnz0, nnz1)]
    m1rm = np.mean(m1r)
    m1rs = np.std(m1r)
    print(lab, " ".join(f"{x:5.0f}" for x in nnz0), "|", f"{m0m:5.0f}±{m0s:<5.0f}")
    print(s,   " ".join(f"{x:5.0f}" for x in nnz1), "|", f"{m1m:5.0f}±{m1s:<5.0f}")
    print(s,   " ".join(f"{r:4.0f}×" for r in m1r), "|", f"{m1rm:5.0f}±{m1rs:<5.0f}")

def get_essential_stats(organized):
    columns = ["n", "bal.acc.train", "bal.acc.test", "bal.acc.test loss", "nnz.leafs", "nnz.fraction", "time"]
    dnames = [n[0:15] for n in organized.keys()]
    index = pd.MultiIndex.from_product([organized.keys(), ["xgb", "xgb.compr", "dt", "dt.compr"]])

    df = pd.DataFrame(" ", columns=columns, index=index)

    columns_raw = [
        "bal.acc.train.mean",
        "bal.acc.train.std",
        "bal.acc.test.mean",
        "bal.acc.test.std",
        "nnz.leafs",
        "time",
    ]
    dfraw = pd.DataFrame("-", columns=columns_raw, index=index)

    nnz_fractions = []

    for dname, fordata in organized.items():
        n = len(fordata["xgb"])
        df.loc[(dname, "xgb"), "n"] = n
        tny = "\\textcolor{gray}{\\scriptsize " #}
        for model_type, formodel in fordata.items():
            print(f"{dname} {model_type}")
            for k0, k1 in [("mtrain_rec", "bal.acc.train"), ("mtest_rec", "bal.acc.test")]:
                m0 = np.array([j[k0][0]*100.0 for j in formodel.values()])
                m1 = np.array([j[k0][-1]*100.0 for j in formodel.values()])
                df.loc[(dname, model_type), k1] = f"{m0.mean():5.1f}\\% {tny}±{m0.std():<4.1f}}}"
                df.loc[(dname, f"{model_type}.compr"), k1] = f"{m1.mean():5.1f}\\% {tny}±{m1.std():<4.1f}}}"
                if "test" in k1:
                    df.loc[(dname, f"{model_type}.compr"), "bal.acc.test loss"] = (
                        f"{np.mean(m1-m0):+4.1f}\\%"
                    )
                dfraw.loc[(dname, model_type), f"{k1}.mean"] = np.mean(m0)
                dfraw.loc[(dname, model_type), f"{k1}.std"] = np.std(m0)
                dfraw.loc[(dname, f"{model_type}.compr"), f"{k1}.mean"] = np.mean(m1)
                dfraw.loc[(dname, f"{model_type}.compr"), f"{k1}.std"] = np.std(m1)

                print_metrics(k0[0:3], m0, m1, 0.01)

            nnzleaf0 = np.array([j["nnzleafs_rec"][0] for j in formodel.values()])
            nnzleaf1 = np.array([j["nnzleafs_rec"][-1] for j in formodel.values()])
            nnzleaf_frac = nnzleaf0 / nnzleaf1

            nnz_fractions += list(nnzleaf_frac)

            df.loc[(dname, model_type), "nnz.leafs"] = f"{nnzleaf0.mean():4.0f} {tny}±{nnzleaf0.std():<4.0f}}}"
            df.loc[(dname, f"{model_type}.compr"), "nnz.leafs"] = f"{nnzleaf1.mean():4.0f} {tny}±{nnzleaf1.std():<4.0f}}}"
            df.loc[(dname, f"{model_type}.compr"), "nnz.fraction"] = f"{nnzleaf_frac.mean():5.1f}×"
            dfraw.loc[(dname, model_type), "nnz.leafs"] = nnzleaf0.mean()
            dfraw.loc[(dname, f"{model_type}.compr"), "nnz.leafs"] = nnzleaf1.mean()

            time0 = np.array([j["clf_train_time"] for j in formodel.values()])
            time1 = np.array([j["compr_time"] for j in formodel.values()])
            df.loc[(dname, f"{model_type}.compr"), "time"] = f"{np.mean(time1):5.1f}s"
            dfraw.loc[(dname, model_type), "time"] = time0.mean()
            dfraw.loc[(dname, f"{model_type}.compr"), "time"] = time1.mean()

            #for k in ["bal.acc.train", "bal.acc.test", "nnz.leafs", "nnz.fraction"]:
            #    df.loc[(dname, " "), k] = " "


            #s = f"{dname} {model_type} {n}"
            #print(s)
            #print_metrics("TRA", mtrain0, mtrain1, abserr)
            #print_metrics("VAL", mvalid0, mvalid1, abserr)
            #print_metrics("TES", mtest0, mtest1, abserr)
            #print_nnzleafs("NNZ", nnzleaf0, nnzleaf1)

    print(dfraw.to_string())

    interesting = [
        "Albert",
        "California",
        "Spambase",
        "DefaultCredit",
        "BankMarketing",
        "Houses",
        "DryBean",
        "Abalone",
        "Yprop",
        "Vehicle",
        "Nomao",
    ]
    bad = [
        "RoadSafety",
        "Santander",
        "Banknote",
        "Bioresponse",
        "Adult",
        "Mercedes",
        "BreastCancer",
        "EyeMovements",
    ]
    dnames = []
    for n in organized.keys():
        n = n[0:15]
        for ni in interesting:
            if ni in n:
                n = f"\\textcolor{{olive}}{{{n}}}"
        for ni in bad:
            if ni in n:
                n = f"\\textcolor{{red}}{{{n}}}"
        dnames.append(n)

    df.index = pd.MultiIndex.from_product([dnames, ["XGB", "XGB.Compr", "DT", "DT.Cmpr"]])
    df.columns = ["n", "b.acc.train", "b.acc.test", "b.acc.test $\\Delta$", "nnz.leafs", "nnz.factor", "time"]

    df = df.drop(columns=["n"])

    print(df.to_string())

    #print(df.iloc[0:4])

    with open("tex/table.tex", "w") as fh:
        for k in range(0, len(df.index)//4, 14):
            #print(df.index[k*4:(k+14)*4])
            fh.write(df.iloc[k*4:(k+14)*4].to_latex(column_format="llrrlrrrr"))
            fh.write("\n\n")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    #ax.set_xscale("log")
    ax.hist(np.log10(nnz_fractions), bins=20)
    xticks = ax.get_xticks()
    ax.set_xticklabels([10**x for x in xticks])



    plt.show()
    


def write_verification_results(organized):
    joblib.dump(organized, "processed_results/verification.joblib")


def load_verification_results():
    return joblib.load("processed_results/verification.joblib")
