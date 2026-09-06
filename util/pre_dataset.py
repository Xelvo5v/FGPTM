import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from Bio.PDB import DSSP
from Bio.PDB.PDBParser import PDBParser
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


DEFAULT_DATA_ROOT = "./data"
DEFAULT_DSSP_PATH = "./data/mkdssp"


class PTMSeqStructureDataset(Dataset):
    def __init__(
        self,
        ptm_type,
        split,
        data_root=DEFAULT_DATA_ROOT,
        dssp_path=DEFAULT_DSSP_PATH,
        split_dir=None,
        repeat=None,
        sequence_center=15,
        windowsize=15,
        build_cache=True,
        cache_workers=1,
    ):
        self.ptm_type = ptm_type
        self.split = split
        self.data_root = Path(data_root)
        self.dssp_path = str(dssp_path)
        self.sequence_center = sequence_center
        self.windowsize = windowsize
        self.cache_workers = max(1, int(cache_workers))
        self.pdb_base_path = self.data_root / f"pdb_structure_{ptm_type}"
        self.split_dir = self.resolve_split_dir(split_dir, repeat)

        self.data_path = self.split_dir / f"{split}.txt"
        self.cache_path = self.split_dir / "graph_cache" / f"{split}_graph_feature_{ptm_type}.pkl"
        self.legacy_cache_path = self.data_root / ptm_type / f"{split}_graph_feature_{ptm_type}.pkl"

        self.samples = self.read_txt(self.data_path)
        self.sequences_used_data = [sample["sequence"] for sample in self.samples]
        self.labels_used_data = [sample["label"] for sample in self.samples]
        self.sequence_feature = self.get_sequence_feature(self.sequences_used_data, self.get_max_ratios())
        self.length = len(self.samples)

        self.g_data_dict = self.load_or_build_graph_cache(build_cache)

    def resolve_split_dir(self, split_dir, repeat):
        if split_dir is not None and repeat is not None:
            raise ValueError("Use either split_dir or repeat, not both")
        if split_dir is not None:
            return Path(split_dir)
        if repeat is not None:
            return self.data_root / self.ptm_type / "splits" / f"repeat_{repeat}"
        return self.data_root / self.ptm_type

    def read_txt(self, data_path):
        if not data_path.exists():
            raise FileNotFoundError(f"PTM split file not found: {data_path}")

        samples = []
        with data_path.open("r") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid data row in {data_path}:{line_no}: {line}")

                sequence = parts[0][
                    (self.sequence_center - self.windowsize):(self.sequence_center + self.windowsize + 1)
                ]
                samples.append(
                    {
                        "sequence": sequence,
                        "label": int(parts[1]),
                        "uniprot_id": parts[2],
                        "modify_site": int(parts[3]) - 1,
                    }
                )
        if not samples:
            raise ValueError(f"No samples found in {data_path}")
        return samples

    def load_or_build_graph_cache(self, build_cache):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                g_data_dict = self._load_or_build_graph_cache(build_cache)
                dist.barrier()
                return g_data_dict

            dist.barrier()
            if self.cache_path.exists():
                with self.cache_path.open("rb") as f:
                    return pickle.load(f)
            if build_cache:
                return self.build_graph_cache()
            raise FileNotFoundError(f"Graph cache not found: {self.cache_path}")

        return self._load_or_build_graph_cache(build_cache)

    def _load_or_build_graph_cache(self, build_cache):
        self.migrate_legacy_cache()
        if self.cache_path.exists():
            with self.cache_path.open("rb") as f:
                g_data_dict = pickle.load(f)
            if build_cache:
                g_data_dict = self.complete_graph_cache(g_data_dict)
            return g_data_dict

        if not build_cache:
            raise FileNotFoundError(f"Graph cache not found: {self.cache_path}")

        return self.build_graph_cache()

    def migrate_legacy_cache(self):
        if self.split_dir != self.data_root / self.ptm_type:
            return
        if self.cache_path.exists() or not self.legacy_cache_path.exists():
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.legacy_cache_path), str(self.cache_path))

    def build_graph_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = self.process_cache_indices(
            range(self.length),
            desc=f"Building {self.ptm_type} {self.split} graph cache",
        )

        with self.cache_path.open("wb") as f:
            pickle.dump(save_dict, f)
        return save_dict

    def complete_graph_cache(self, g_data_dict):
        missing_indices = []
        for idx, sample in enumerate(self.samples):
            key = self.get_cache_key(sample["uniprot_id"], sample["modify_site"])
            if key not in g_data_dict:
                missing_indices.append(idx)
        if not missing_indices:
            return g_data_dict

        g_data_dict.update(
            self.process_cache_indices(
                missing_indices,
                desc=f"Completing {self.ptm_type} {self.split} graph cache",
            )
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump(g_data_dict, f)
        return g_data_dict

    def process_cache_indices(self, indices, desc):
        indices = list(indices)
        protein_indices = {}
        for idx in indices:
            sample = self.samples[idx]
            protein_indices.setdefault(sample["uniprot_id"], []).append(idx)

        groups = list(protein_indices.items())
        if self.cache_workers == 1:
            group_items = (
                self.process_cache_group(group)
                for group in tqdm(groups, desc=desc)
            )
        else:
            with ThreadPoolExecutor(max_workers=self.cache_workers) as executor:
                group_items = tqdm(
                    executor.map(self.process_cache_group, groups),
                    total=len(groups),
                    desc=(
                        f"{desc} ({self.cache_workers} workers, "
                        f"{len(indices)} sites/{len(groups)} proteins)"
                    ),
                )
                return {
                    key: data
                    for items in group_items
                    for key, data in items
                }
        return {
            key: data
            for items in group_items
            for key, data in items
        }

    def process_cache_index(self, idx):
        sample = self.samples[idx]
        key = self.get_cache_key(sample["uniprot_id"], sample["modify_site"])
        data = self.process_pdb(sample["uniprot_id"], sample["modify_site"], idx)
        return key, data

    def process_cache_group(self, group):
        uniprot_id, indices = group
        dssp_feature, coord_all_AA = self.load_pdb_features(uniprot_id)
        items = []
        for idx in indices:
            sample = self.samples[idx]
            key = self.get_cache_key(sample["uniprot_id"], sample["modify_site"])
            data = self.build_graph_data(
                dssp_feature,
                coord_all_AA,
                sample["modify_site"],
                idx,
            )
            items.append((key, data))
        return items

    def get_cache_key(self, uniprot_id, modify_site):
        return f"{uniprot_id}_{modify_site}"

    def get_sequence_feature(self, aa_list, max_ratios):
        joiner = ""
        motif_list = [max_ratios[i]["char"] for i in range(len(max_ratios))]
        motif = joiner.join(motif_list)
        motif_sequence_AAI14_BLOSUM62 = self.AAI14_BLOSUM62(motif)
        motif_sequence_AAI14_BLOSUM62_difference = np.zeros((len(aa_list), len(max_ratios)))
        for sequence_index in range(len(aa_list)):
            sequence_AAI14_BLOSUM = self.AAI14_BLOSUM62(aa_list[sequence_index])
            list_feature = []
            for i in range(sequence_AAI14_BLOSUM.shape[0]):
                list_feature.append(
                    max_ratios[i]["ratio"]
                    * np.sqrt(np.sum(np.square(sequence_AAI14_BLOSUM[i] - motif_sequence_AAI14_BLOSUM62[i])))
                )
            motif_sequence_AAI14_BLOSUM62_difference[sequence_index] = list_feature
        return motif_sequence_AAI14_BLOSUM62_difference

    def get_max_ratios(self):
        positive_sequences = [
            list(self.sequences_used_data[i])
            for i in range(len(self.sequences_used_data))
            if self.labels_used_data[i] == 1
        ]
        if not positive_sequences:
            positive_sequences = [list(seq) for seq in self.sequences_used_data]

        array = np.array(positive_sequences)
        column_ratios = {}
        for i in range(array.shape[1]):
            column = array[:, i]
            unique_chars, counts = np.unique(column, return_counts=True)
            ratios = counts / np.sum(counts)
            column_ratios[i] = dict(zip(unique_chars, ratios))

        max_ratios = {}
        for i in range(array.shape[1]):
            column_ratio = column_ratios[i]
            max_char = max(column_ratio, key=column_ratio.get)
            max_ratios[i] = {"char": max_char, "ratio": column_ratio[max_char]}
        return max_ratios

    def get_aa_id(self, aa_string):
        token2index = {}
        aa_vocab = list("$ACDEFGHIKLMNPQRSTWYV")
        for i in range(21):
            token2index[aa_vocab[i]] = i
        aa_string = list(aa_string)
        for i in range(len(aa_string)):
            if aa_string[i] not in aa_vocab:
                aa_string[i] = "$"
        return np.array([token2index[residue] for residue in aa_string])

    def AAI14_BLOSUM62(self, gene):
        with open("./ckpts/AAI.txt") as f:
            records = f.readlines()[1:]
        AAI = []
        for i in records:
            array = i.rstrip().split()[1:] if i.rstrip() != "" else None
            AAI.append(array)
        AAI = np.array(
            [float(AAI[i][j]) for i in range(len(AAI)) for j in range(len(AAI[i]))]
        ).reshape((14, 21))
        AAI = AAI.transpose()
        GENE_BE = {}
        AA = "ACDEFGHIKLMNPQRSTWYV$"
        for i in range(len(AA)):
            GENE_BE[AA[i]] = i
        n = len(gene)
        gene_array = np.zeros((n, AAI.shape[1]))
        for i in range(n):
            if gene[i] in AA:
                gene_array[i] = AAI[(GENE_BE[gene[i]])]
            else:
                gene_array[i] = AAI[(GENE_BE["$"])]

        with open("./ckpts/blosum62.txt") as f:
            records = f.readlines()[1:]
        blosum62 = []
        for i in records:
            array = i.rstrip().split() if i.rstrip() != "" else None
            blosum62.append(array)
        blosum62 = np.array(
            [float(blosum62[i][j]) for i in range(len(blosum62)) for j in range(len(blosum62[i]))]
        ).reshape((20, 21))
        blosum62 = blosum62.transpose()
        GENE_BE = {}
        AA = "ARNDCQEGHILKMFPSTWYV$"
        for i in range(len(AA)):
            GENE_BE[AA[i]] = i
        n = len(gene)
        gene_array_1 = np.zeros((n, 20))
        for i in range(n):
            if gene[i] in AA:
                gene_array_1[i] = blosum62[(GENE_BE[gene[i]])]
            else:
                gene_array_1[i] = blosum62[(GENE_BE["$"])]
        return np.hstack((gene_array, gene_array_1))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        key = self.get_cache_key(sample["uniprot_id"], sample["modify_site"])
        if key not in self.g_data_dict:
            self.g_data_dict[key] = self.process_pdb(sample["uniprot_id"], sample["modify_site"], idx)
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                with self.cache_path.open("wb") as f:
                    pickle.dump(self.g_data_dict, f)
        return self.g_data_dict[key]

    def process_pdb(self, uniprot_id, modify_site, idx):
        dssp_feature, coord_all_AA = self.load_pdb_features(uniprot_id)
        return self.build_graph_data(dssp_feature, coord_all_AA, modify_site, idx)

    def load_pdb_features(self, uniprot_id):
        pdb_path = self.pdb_base_path / f"{uniprot_id}.pdb"
        if pdb_path.exists():
            try:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("1", str(pdb_path))
                model = structure[0]
                dssp_kwargs = {}
                if self.dssp_path:
                    dssp_kwargs["dssp"] = self.dssp_path
                dssp = DSSP(model, str(pdb_path), **dssp_kwargs)
                dssp_feature = self.get_aa_dssp_node(dssp)
                coord_all_AA, _ = self.get_aa_atom_coords(pdb_path)
                return dssp_feature, coord_all_AA
            except Exception as exc:
                print(f"Failed to build graph for {pdb_path}: {exc}")
        return None, None

    def build_graph_data(self, dssp_feature, coord_all_AA, modify_site, idx):
        if dssp_feature is not None and coord_all_AA is not None and modify_site < len(coord_all_AA):
            try:
                edge_index, dssp_feature = self.get_aa_edge(coord_all_AA, dssp_feature, modify_site)
            except Exception as exc:
                sample = self.samples[idx]
                pdb_path = self.pdb_base_path / f"{sample['uniprot_id']}.pdb"
                print(f"Failed to build graph for {pdb_path} site {modify_site}: {exc}")
                dssp_feature, edge_index = self.empty_graph()
        else:
            dssp_feature, edge_index = self.empty_graph()

        return Data(
            x=dssp_feature,
            edge_index=edge_index.long(),
            sequence_feature=torch.from_numpy(self.sequence_feature[idx]).float(),
            aa_id=torch.from_numpy(self.get_aa_id(self.sequences_used_data[idx])).long(),
            modif_site=torch.tensor(modify_site, dtype=torch.long),
            label=torch.FloatTensor([float(self.labels_used_data[idx])]),
        )

    def empty_graph(self):
        dssp_feature = torch.zeros((2, 15))
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        return dssp_feature, edge_index

    def get_aa_edge(self, coord_all_AA, dssp_feature, modify_site, dr=10):
        AA_index = modify_site
        point = coord_all_AA[AA_index]
        dist = ((coord_all_AA - point.unsqueeze(0)) ** 2).sum(dim=-1).sqrt()
        num_aa = len(dist)
        idx_all = torch.arange(num_aa)

        idx_mask_long = torch.ones_like(dist, dtype=torch.bool)
        idx_mask_long[max(modify_site - 2, 0):min(modify_site + 3, num_aa)] = False
        idx_mask_long = idx_mask_long & (dist < dr)

        idx_mask_short = torch.zeros_like(dist, dtype=torch.bool)
        idx_mask_short[max(modify_site - 2, 0):min(modify_site + 3, num_aa)] = True
        idx_mask_short = idx_mask_short & (dist < dr / 2)

        target_idx = torch.cat([idx_all[idx_mask_long], idx_all[idx_mask_short]], dim=0)
        source_idx = torch.full_like(target_idx, modify_site)
        edge_index = torch.stack([source_idx, target_idx], dim=0)

        abs_idx = (AA_index - idx_all).abs().unsqueeze(1)
        dssp_feature = torch.cat([torch.from_numpy(dssp_feature), dist.unsqueeze(1), abs_idx], dim=1)
        return edge_index, dssp_feature

    def get_aa_dssp_node(self, dssp):
        SS_type = "HBEGITS-"
        dssp_feature = []
        for i in range(len(dssp)):
            SS_vec = np.zeros(8, dtype=np.float32)
            SS = dssp.property_list[i][2]
            SS_vec[SS_type.find(SS)] = 1
            PHI = dssp.property_list[i][4]
            PSI = dssp.property_list[i][5]
            ASA = np.array([dssp.property_list[i][3]], dtype=np.float32)
            angle = np.array([PHI, PSI], dtype=np.float32)
            radian = angle * (np.pi / 180)
            feature = np.concatenate((np.sin(radian), np.cos(radian), ASA, SS_vec), axis=0)
            dssp_feature.append(feature)
        return np.array(dssp_feature, dtype=np.float32)

    def get_aa_atom_coords(self, pdb_path):
        coord_all_Atom = []
        coord_all_AA = []
        with open(pdb_path, "r") as f:
            filt_atom = "CA"
            for line in f:
                kind = line[:6].strip()
                if kind not in ["ATOM"]:
                    continue
                atom, _, _, _, x, y, z, _ = self.pdb_split(line)
                if atom == filt_atom:
                    coord_all_AA.append([x, y, z])
                coord_all_Atom.append([x, y, z])
        return torch.FloatTensor(coord_all_AA), torch.FloatTensor(coord_all_Atom)

    def pdb_split(self, line):
        atom_type = "CNOS$"
        aa_trans_DICT = {
            "ALA": "A", "CYS": "C", "CCS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "MSE": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }
        aa_type = "ACDEFGHIKLMNPQRSTVWY$"
        Atom_order = int(line[6:11].strip()) - 1
        atom = line[11:16].strip()
        amino = line[16:21].strip()
        AA_order = int(line[22:28].strip()) - 1
        x = line[28:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()
        atom_single_name = line.strip()[-1]
        atom_single_name_vec = np.zeros(len(atom_type))
        atom_single_name_vec[atom_type.find(atom_single_name)] = 1
        AA_single_name_vec = np.zeros(len(aa_type))
        AA_single_name_vec[aa_type.find(aa_trans_DICT[amino])] = 1
        atom_feature_combine = np.concatenate(
            (atom_single_name_vec.reshape(1, -1), AA_single_name_vec.reshape(1, -1)), axis=1
        )
        return atom, amino, AA_order, Atom_order, float(x), float(y), float(z), atom_feature_combine

    def __len__(self):
        return self.length


PTM_Seq_Structure_dataset = PTMSeqStructureDataset


def get_dataset(
    ptm_type,
    split,
    data_root=DEFAULT_DATA_ROOT,
    dssp_path=DEFAULT_DSSP_PATH,
    build_cache=True,
    split_dir=None,
    repeat=None,
    cache_workers=1,
):
    return PTMSeqStructureDataset(
        ptm_type=ptm_type,
        split=split,
        data_root=data_root,
        dssp_path=dssp_path,
        build_cache=build_cache,
        split_dir=split_dir,
        repeat=repeat,
        cache_workers=cache_workers,
    )


def get_dataloader(
    ptm_type,
    batch_size,
    num_workers,
    data_root=DEFAULT_DATA_ROOT,
    dssp_path=DEFAULT_DSSP_PATH,
    split_dir=None,
    repeat=None,
    cache_workers=1,
):
    train_dataset = get_dataset(
        ptm_type,
        "train",
        data_root=data_root,
        dssp_path=dssp_path,
        split_dir=split_dir,
        repeat=repeat,
        cache_workers=cache_workers,
    )
    test_dataset = get_dataset(
        ptm_type,
        "test",
        data_root=data_root,
        dssp_path=dssp_path,
        split_dir=split_dir,
        repeat=repeat,
        cache_workers=cache_workers,
    )

    train_sampler = DistributedSampler(train_dataset) if torch.distributed.is_initialized() else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if torch.distributed.is_initialized() else None

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train_sampler is None,
        "num_workers": num_workers,
        "pin_memory": True,
        "sampler": train_sampler,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    test_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "sampler": test_sampler,
    }
    if num_workers > 0:
        test_loader_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)
    return train_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Build PTM graph cache")
    parser.add_argument("--ptm_type", default="Nitrosylation", type=str)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, type=str)
    parser.add_argument("--dssp_path", default=DEFAULT_DSSP_PATH, type=str)
    parser.add_argument("--repeat", default=None, type=int)
    parser.add_argument("--cache_workers", default=1, type=int)
    args = parser.parse_args()

    get_dataset(
        args.ptm_type,
        args.split,
        args.data_root,
        args.dssp_path,
        build_cache=True,
        repeat=args.repeat,
        cache_workers=args.cache_workers,
    )
