import urllib.request
import os
import glob
import zipfile
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-paper")

from matplotlib.ticker import FuncFormatter

# Download data sources from ssa.gov
def dl_ssa():
    # Ensure the directory 'ds' exists
    if not os.path.exists("ds"):
        os.makedirs("ds")

    # Download data sources from ssa.gov
    for _ in [
        "https://www.ssa.gov/oact/babynames/state/namesbystate.zip",
        "https://www.ssa.gov/oact/HistEst/Death/2022/DeathProbsE_F_Alt2_TR2022.csv",
        "https://www.ssa.gov/oact/HistEst/Death/2022/DeathProbsE_M_Alt2_TR2022.csv",
    ]:
        if not os.path.isfile("ds/{}".format(os.path.basename(_))):
            with urllib.request.urlopen(_) as p, open(
                "ds/{}".format(os.path.basename(_)), "wb"
            ) as f:
                f.write(p.read())

    # Unzip
    for _ in glob.glob(r"ds/*.zip"):
        with zipfile.ZipFile(_, "r") as z:
            z.extractall("ds/namesbystate")


# Generate plot
def gen_plot(df, _reg="US", _nam="Rebecca", _sex="F"):
    dm = pd.read_csv("./ds/DeathProbsE_{}_Alt2_TR2022.csv".format(_sex), skiprows=1)
    a = dm.set_index("Year").loc[pd.Timestamp.now().year].rename("pd")
    a.index = a.index.astype("int64")
    a.index.name = "age"

    if _reg == "US":
        b = df[(df.sex == _sex) & (df.name == _nam)].groupby(["age"]).n.sum()
    else:
        b = (
            df[(df.sex == _sex) & (df.name == _nam) & (df.state == _reg)]
            .groupby(["age"])
            .n.sum()
        )

    c = pd.concat(
        [
            b,
            a,
        ],
        axis=1,
    ).dropna()

    c["pl"] = 1 - c.pd
    c["plc"] = c.pl.cumprod()
    c["adj"] = c.n * c.plc
    c["by"] = pd.Timestamp.now().year - c.index
    c["adjc"] = c.adj.cumsum()

    m = c[c.adjc >= c.adjc.max() / 2].iloc[0].name
    d = c[["n", "adj"]]
    d.index.name = ""

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    d.n.rename("Born").plot()
    d.adj.rename("Expected Living (2022)").plot(kind="area")
    plt.title(
        "{}: {} Age Distribution; {:,.0f} Expected".format(
            _nam, _reg, d["adj"].sum().round(0)
        )
    )
    plt.axvline(x=m, c="k", ls=":", lw=0.8)
    ax.set_xlim(0, 100)
    ax.get_yaxis().set_major_formatter(
        # FuncFormatter(lambda a, b: "{:,.0f}K".format(a * 1e-3))
        FuncFormatter(lambda a, b: "{:,.0f}".format(a))
    )
    ax.annotate(
        "{:.0f}".format(m),
        xy=(m, 0),
        xycoords="data",
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="center",
        size=8,
        weight="bold",
        bbox={"fc": "white", "ec": "white", "pad": 4, "alpha": 1},
    )
    plt.legend(frameon=0)
    plt.savefig("nameage-{}-{}.pdf".format(_reg.lower(), _nam.lower()))
    plt.close()


if __name__ == "__main__":
    # Download data sources from ssa.gov
    dl_ssa()

    # Create a pandas DataFrame; combine each state's birth records
    df = pd.DataFrame()
    for _ in glob.glob("ds/namesbystate/*.TXT"):
        print("Adding: {}".format(_))
        a = pd.read_csv(_, header=None, names=["state", "sex", "by", "name", "n"])
        df = pd.concat([df, a])

    df["age"] = pd.Timestamp.now().year - df.by

    # Generate nameage plots
    gen_plot(df, _nam="John", _sex="M")
    gen_plot(df, _nam="Paul", _sex="M")
    gen_plot(df, _nam="Lucy", _sex="F")
    gen_plot(df, _nam="Rita", _sex="F")
