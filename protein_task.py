import json
import os
import pickle

import numpy as np
import pandas as pd
import torch


AA_3_TO_1_DICT = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
    "UNK": "U",
    "-": "-",
}


class ProteinTask:
    def __init__(self, task=None, protein_job=None, protein_of=None):
        """
        ProteinTask class contains info about protein and set mutation
        task: json
        """
        if task is None:
            self.task = {
                "input_protein": {
                    "path": None,  # input protein path
                    "chains": None,  # input protein chains
                    "regions": None,  # input protein regions
                },
                # input mutants in format (aa_wt, residue_position, aa_mt, chain_id)
                "mutants": {},
                # positions to calculate embeddings for in original numbering system
                "obs_positions": [],
            }
        else:
            self.task = task
        if protein_job is None:
            self.protein_job = {
                "protein_wt": {
                    # processed protein dataframe with introduced mutants
                    "protein": pd.DataFrame(),
                    # positions in the internal numbering system
                    "obs_positions": {},
                },
                "protein_mt": {
                    # processed protein dataframe with introduced mutants
                    "protein": pd.DataFrame(),
                    # positions in the internal numbering system
                    "obs_positions": {},
                },
            }
        else:
            self.protein_job = protein_job
        if protein_of is None:
            self.protein_of = {
                "protein_wt": {"protein": pd.DataFrame(), "features": {}},
                "protein_mt": {"protein": pd.DataFrame(), "features": {}},
            }
        else:
            self.protein_of = protein_of

    def set_input_protein_task(self, protein_path=None, chains=None, regions=None, task=None):
        if task is None:
            self.task["input_protein"] = {
                "path": protein_path,
                "chains": chains,
                "regions": regions,
            }
        else:
            self.task = task

    def set_task_mutants(self, mutants):
        self.task["mutants"] = mutants

    def add_mutation(self, chain_id, resi, aa_mt, aa_wt):
        self.task["mutants"][(aa_wt, resi, chain_id)] = aa_mt

    def get_mutations(self):
        return self.task["mutants"]

    def set_wildtype_observable_position(self, obs_positions):
        self.protein_job["protein_wt"]["obs_positions"] = obs_positions

    def set_mutate_observable_position(self, obs_positions):
        self.protein_job["protein_mt"]["obs_positions"] = obs_positions

    def set_wildtype_of_protein(self, of_features, of_protein=None, of_cycles=None):
        if of_protein is None:
            of_protein = pd.DataFrame()
        if of_cycles is None:
            of_cycles = {}
        self.protein_of["protein_wt"] = {
            "protein": of_protein,
            "features": of_features,
            "cycles": of_cycles,
        }

    def set_mutate_of_protein(self, of_features, of_protein=None, of_cycles=None):
        if of_protein is None:
            of_protein = pd.DataFrame()
        if of_cycles is None:
            of_cycles = {}
        self.protein_of["protein_mt"] = {
            "protein": of_protein,
            "features": of_features,
            "cycles": of_cycles,
        }

    def set_observable_positions(self, obs_positions=None):
        if obs_positions is None:
            prot_wt = self.get_wildtype_protein()
            if self.task["input_protein"]["chains"] is not None:
                chains = self.task["input_protein"]["chains"]
            else:
                chains = prot_wt.chain_id_original.unique()
            for chain in chains:
                start = min(
                    prot_wt.loc[
                        prot_wt["chain_id_original"] == chain, "residue_number_original"
                    ]
                )
                end = prot_wt.loc[
                    (prot_wt["chain_id_original"] == chain) & (prot_wt.atom_name == "CA"),
                ].shape[0]
                self.task["obs_positions"].append(
                    {"resi": start, "shift_from_resi": end - 1, "chain_id": chain}
                )
        else:
            self.task["obs_positions"] = obs_positions

    def add_observable_positions(self, resi=None, shift_from_resi=0, chain_id=None, name=None):
        if name is None:
            name = f"{resi}_{chain_id}"
        self.task["obs_positions"].append(
            {
                "resi": resi,
                "shift_from_resi": shift_from_resi,
                "chain_id": chain_id,
                "name": name,
            }
        )

    def get_mutate_protein(self):
        return self.protein_job["protein_mt"]["protein"]

    def get_wildtype_protein(self):
        return self.protein_job["protein_wt"]["protein"]

    def get_mutate_observable_positions(self):
        return self.protein_job["protein_mt"]["obs_positions"]

    def get_wildtype_observable_positions(self):
        return self.protein_job["protein_wt"]["obs_positions"]

    def get_wildtype_protein_of(self):
        return self.protein_of["protein_wt"]["protein"]

    def get_mutate_protein_of(self):
        return self.protein_of["protein_mt"]["protein"]

    def get_wildtype_protein_of_features(self):
        return self.protein_of["protein_wt"]["features"]

    def get_mutate_protein_of_features(self):
        return self.protein_of["protein_mt"]["features"]

    def get_wildtype_protein_of_cycles(self):
        return self.protein_of["protein_wt"]["cycles"]

    def get_mutate_protein_of_cycles(self):
        return self.protein_of["protein_mt"]["cycles"]

    def get_wt_aa(self, position, chain):
        protein = self.get_wildtype_protein()
        aa = protein.loc[
            (protein["residue_number_original"] == position)
            & (protein["chain_id_original"] == chain),
            "residue_name",
        ].unique()[0]
        return AA_3_TO_1_DICT[aa]

    def get_index_for_observable_position(self, obs_pos, protein_mt, mutations=None):
        new_index = {}
        resi_pos = obs_pos["resi"]
        if mutations is not None:
            pos_del = [m[1] for m in mutations if mutations[m] == "-"]
            if resi_pos in pos_del:
                if resi_pos == 1:
                    resi_pos = 2
                else:
                    resi_pos = resi_pos - 1
        # resi_range = list(range(resi_pos,
        #                         resi_pos + obs_pos["shift_from_resi"] + 1))
        protein_mt_ = protein_mt[protein_mt["atom_name"] == "CA"].reset_index(drop=True)
        resi_obs_start = protein_mt_[
            (protein_mt_["residue_number_original"] == resi_pos)
            & (protein_mt_["chain_id_original"] == obs_pos["chain_id"])
            & (protein_mt_["insertion"] == "")
        ].index.values

        if len(resi_obs_start) == 0:
            AssertionError("Couldn't find observable residue in muatated protein", obs_pos)
        if len(resi_obs_start) != 1:
            AssertionError(
                "Several residues corresponds to the observable residue",
                len(resi_obs_start),
                obs_pos,
            )
            raise
        resi_obs_start = resi_obs_start[0]
        resi_obs = list(range(resi_obs_start, resi_obs_start + obs_pos["shift_from_resi"] + 1))
        for idx in resi_obs:
            original_resi_num = protein_mt_.loc[idx, "residue_number_original"]
            original_resi_chain = obs_pos["chain_id"]
            original_aa_wt = self.get_wt_aa(original_resi_num, original_resi_chain)

            # change position name if current residue was inserted
            is_indel = protein_mt_.loc[idx, "mask"]
            if is_indel:
                pos_id = f"{original_aa_wt}_{original_resi_chain}_{original_resi_num}i"
            else:
                pos_id = f"{original_aa_wt}_{original_resi_chain}_{original_resi_num}"
            new_index[pos_id] = idx
        return new_index

    def load_task(self, path):
        # ToDo: check format of json dict
        self.task = json.load(open(path))

    def save_protein_job(self, path):
        pickle.dump(self.protein_job, open(path, "wb"))

    def save_protein_of_w_features(self, path):
        pickle.dump(self.protein_of, open(path, "wb"))

    def save(self, path):
        pickle.dump(self, open(path, "wb"))


def get_protein_task(df, idx, path):
    row = df.iloc[idx]
    pkl_path = os.path.join(path, f"{row['id']}.pkl")
    with open(pkl_path, "rb") as f:
        features = pickle.load(f)
    return features


def get_feature_tensor(task, feature_names=None):
    if feature_names is None:
        feature_names = ["pair", "single", "plddt", "lddt_logits"]

    # Check feature names
    for f_name in feature_names:
        assert f_name in [
            "msa",
            "pair",
            "lddt_logits",
            "distogram_logits",
            "aligned_confidence_probs",
            "predicted_aligned_error",
            "plddt",
            "single",
            "tm_logits",
        ], (
            f"Unknown feature name: {f_name}! "
            "Known feature names: 'msa', 'pair', 'lddt_logits', 'distogram_logits', "
            "'aligned_confidence_probs', 'predicted_aligned_error', "
            "'plddt', 'single', 'tm_logits'."
        )

    pair = {
        "wt": task.get_wildtype_protein_of_features(),
        "mt": task.get_mutate_protein_of_features(),
    }

    for protein_type, features in pair.items():
        feature_tensor = [
            torch.from_numpy(
                np.concatenate(
                    [f[f_name] for f_name in feature_names],
                    axis=0,
                )
            )
            for f in list(features.values())
        ]
        pair[protein_type] = torch.stack(feature_tensor, dim=0)

    pair["metadata"] = task.task["input_protein"]

    return pair
