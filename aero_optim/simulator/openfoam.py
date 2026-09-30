import logging
import os
import shutil
import signal
import subprocess

import numpy as np
import pandas as pd

from aero_optim.mesh.foam_cascade_mesh import cascade_mattia_patch_types
from aero_optim.mesh.foam_mesh import convert_to_foam
from aero_optim.simulator.cascade_qoi import QOI_NAMES, qoi_history, summarize
from aero_optim.simulator.simulator import Simulator

logger = logging.getLogger(__name__)

SUMMARY_FILE = "qois.csv"


class OpenFOAMSimulator(Simulator):
    """
    Runs one cascade blade with OpenFOAM at several operating points and computes its QoIs.

    Per blade (`<outdir>/OPENFOAM/openfoam_g<gid>_c<cid>/`): the mesh is converted once
    (`mesh/`), then one case per operating point is launched as an independent process.
    When all of a blade's runs have finished, the measurement-plane samples are reduced to
    QoIs (mean, std and variance over the sampling window) and written to `qois.csv`.
    A failed run yields NaN QoIs for that operating point instead of stopping the study.

    `df_dict[gid][cid]` is a dict `{operating_point: one-row DataFrame}`, the layout the
    cascade optimizers already use.

    Config `"simulator"` entries:
    - ref_input (str): case template directory with `system/`, `constant/` and `0.orig/`.
    - operating_points (dict): name -> {"Uinlet": [ux, uy, uz], "pOutlet": p, "Tinlet": T}.
    - exec_cmd (str): solver command, default "rhoSimpleFoam".
    - end_time, sample_start, sample_interval (int): run length and QoI sampling window.
    - pitch (float): periodicity in y, default 0.04039.
    - keep_samples (bool): keep the raw plane samples, default False.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        sim = config["simulator"]
        self.ops: dict[str, dict] = sim["operating_points"]
        self.end_time: int = sim.get("end_time", 6000)
        self.sample_start: int = sim.get("sample_start", 4000)
        self.sample_interval: int = sim.get("sample_interval", 1)
        self.pitch: float = sim.get("pitch", 0.04039)
        self.keep_samples: bool = sim.get("keep_samples", False)
        self.sim_pro: list[tuple[dict, subprocess.Popen]] = []
        self.returncodes: dict[tuple[int, int], dict[str, int | None]] = {}

    def set_solver_name(self):
        self.solver_name = "openfoam"

    def process_config(self):
        sim = self.config["simulator"]
        for key in ("ref_input", "operating_points"):
            if key not in sim:
                raise Exception(f"ERROR -- no <{key}> entry in {sim}")
        sim.setdefault("exec_cmd", "rhoSimpleFoam")

    def execute_sim(self, meshfile: str, gid: int = 0, cid: int = 0):
        """
        **Prepares and launches** the runs of one blade, or reloads its finished results.
        """
        sim_outdir = self.get_sim_outdir(gid, cid)
        self.df_dict.setdefault(gid, {})
        summary = os.path.join(sim_outdir, SUMMARY_FILE)
        if os.path.isfile(summary):
            logger.info(f"g{gid}, c{cid} results found in {sim_outdir}")
            self.df_dict[gid][cid] = self._to_df_dict(pd.read_csv(summary, index_col=0))
            return
        shutil.rmtree(sim_outdir, ignore_errors=True)
        self.returncodes[(gid, cid)] = {op: None for op in self.ops}
        try:
            self.pre_process(meshfile, sim_outdir)
        except Exception as e:
            logger.error(f"g{gid}, c{cid} pre-processing failed: {e}")
            self.returncodes[(gid, cid)] = {op: -1 for op in self.ops}
            self.df_dict[gid][cid] = self.post_process(gid, cid)
            return
        for op in self.ops:
            case = os.path.join(sim_outdir, op)
            with open(os.path.join(case, "log.solver"), "wb") as log:
                proc = subprocess.Popen(
                    self.exec_cmd, cwd=case, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                    env={**os.environ, "PWD": os.path.abspath(case)},
                )
            self.sim_pro.append(({"gid": gid, "cid": cid, "op": op}, proc))
        logger.info(f"g{gid}, c{cid} launched {list(self.ops)} in {sim_outdir}")

    def pre_process(self, meshfile: str, sim_outdir: str):
        """
        **Converts** the mesh once and **creates** one case per operating point.
        """
        mesh_case = os.path.join(sim_outdir, "mesh")
        for sub in ("system", "constant"):
            shutil.copytree(os.path.join(self.ref_input, sub), os.path.join(mesh_case, sub))
        convert_to_foam(meshfile, mesh_case, cascade_mattia_patch_types(self.pitch), init_fields=False)

        for op, values in self.ops.items():
            case = os.path.join(sim_outdir, op)
            for sub in ("system", "constant"):
                shutil.copytree(os.path.join(self.ref_input, sub), os.path.join(case, sub))
            shutil.copytree(os.path.join(mesh_case, "constant", "polyMesh"),
                            os.path.join(case, "constant", "polyMesh"))
            shutil.copytree(os.path.join(self.ref_input, "0.orig"), os.path.join(case, "0"))
            u = " ".join(str(v) for v in values["Uinlet"])
            with open(os.path.join(case, "0", "include", "operatingPoint"), "w") as f:
                f.write(f"Uinlet ({u});\npOutlet {values['pOutlet']};\nTinlet {values['Tinlet']};\n")
            with open(os.path.join(case, "system", "include", "runControl"), "w") as f:
                f.write(f"endTime {self.end_time};\nsampleStart {self.sample_start};\n"
                        f"sampleInterval {self.sample_interval};\n")
            open(os.path.join(case, "case.foam"), "w").close()

    def monitor_sim_progress(self) -> int:
        """
        **Updates** the running processes, post-processes finished blades and
        **returns** the number of running processes (one core each).
        """
        running = []
        for dict_id, proc in self.sim_pro:
            returncode = proc.poll()
            if returncode is None:
                running.append((dict_id, proc))
                continue
            gid, cid, op = dict_id["gid"], dict_id["cid"], dict_id["op"]
            if returncode != 0:
                logger.error(f"g{gid}, c{cid} {op} failed with return code {returncode}")
            self.returncodes[(gid, cid)][op] = returncode
            if all(rc is not None for rc in self.returncodes[(gid, cid)].values()):
                self.df_dict[gid][cid] = self.post_process(gid, cid)
        self.sim_pro = running
        return len(self.sim_pro)

    def post_process(self, gid: int, cid: int) -> dict[str, pd.DataFrame]:
        """
        **Reduces** each operating point's samples to QoIs and writes the blade summary.
        """
        sim_outdir = self.get_sim_outdir(gid, cid)
        rows = {}
        for op, returncode in self.returncodes.pop((gid, cid)).items():
            case = os.path.join(sim_outdir, op)
            sets_dir = os.path.join(case, "postProcessing", "MP")
            row = {q: np.nan for q in QOI_NAMES}
            row["status"] = "failed"
            if returncode == 0:
                try:
                    history = qoi_history(sets_dir)
                    history.to_csv(os.path.join(case, "qoi_history.csv"))
                    row = summarize(history)
                    row["status"] = "ok"
                    if not self.keep_samples:
                        shutil.rmtree(sets_dir, ignore_errors=True)
                except Exception as e:
                    logger.error(f"g{gid}, c{cid} {op} post-processing failed: {e}")
            rows[op] = row
        summary = pd.DataFrame.from_dict(rows, orient="index")
        os.makedirs(sim_outdir, exist_ok=True)
        summary.to_csv(os.path.join(sim_outdir, SUMMARY_FILE))
        logger.info(f"g{gid}, c{cid} QoIs:\n{summary[['status'] + QOI_NAMES].to_string()}")
        return self._to_df_dict(summary)

    @staticmethod
    def _to_df_dict(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {op: summary.loc[[op]].reset_index(drop=True) for op in summary.index}

    def kill_all(self):
        logger.info(f"{len(self.sim_pro)} remaining simulation(s) will be killed")
        for _, proc in self.sim_pro:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
