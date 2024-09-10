import os
import sys
from typing import Tuple

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


sys.path[-1] = os.path.join(os.path.dirname(__file__))
from utils import get_feature_tensor, get_protein_task


current_dir = os.getcwd()
os.chdir(os.path.dirname(os.sep.join(__file__.split(os.sep)[:-1])))
cfg: DictConfig = OmegaConf.load("src/config/config.yaml")
os.chdir(current_dir)


def prepare_train_data() -> Tuple[torch.Tensor, torch.Tensor]:
    # Load training data
    df_train = pd.read_csv(f"./{cfg.general.data_dir}/prostata_filtered.csv")
    train_targets = torch.tensor(df_train["ddg"], dtype=torch.float32)

    # Load protein features.
    # P.s. Protein is represented as a ```ProteinTask``` class object.
    all_train_features = []
    for idx in trange(len(df_train), desc="Preparing train data"):
        task = get_protein_task(
            df_train, idx=idx, path=f"./{cfg.general.data_dir}/prostata_test_task"
        )

        mutation = task.task["mutants"]
        mutation_key, _ = next(iter(mutation.items()))
        res_name, position, chain_id = mutation_key
        residue_name = "_".join((res_name, chain_id, str(position)))
        feature_index = task.protein_job["protein_wt"]["obs_positions"][residue_name]
        feature_tensor = get_feature_tensor(
            task, feature_names=["pair", "lddt_logits", "plddt"]
        )

        all_train_features.append(
            torch.cat(
                (feature_tensor["wt"][feature_index], feature_tensor["mt"][feature_index]),
                dim=0,
            )
        )
    train_features = torch.stack(all_train_features, dim=0)
    return train_features, train_targets


def prepare_test_data() -> Tuple[torch.Tensor, torch.Tensor]:
    df_test_ssym = pd.read_csv(f"./{cfg.general.data_dir}/ssym.csv")
    df_test_s669 = pd.read_csv(f"./{cfg.general.data_dir}/s669.csv")
    df_test = pd.concat((df_test_ssym, df_test_s669), axis="rows", ignore_index=True)

    # test_targets = torch.tensor(df_test["ddg"], dtype=torch.float32) # test DDG not available
    test_targets = torch.zeros(
        df_test.shape[0], dtype=torch.float32
    )  # Note that this is FAKE target

    test_features = []
    path_to_test_tasks = {
        "ssym": f"./{cfg.general.data_dir}/ssym_test_task",
        "s669": f"./{cfg.general.data_dir}/s669_test_task",
    }

    for idx in trange(len(df_test), desc="Preparing test data"):
        source = df_test.iloc[idx]["source"]
        task = get_protein_task(df_test, idx=idx, path=path_to_test_tasks[source])

        mutation = task.task["mutants"]
        mutation_key, _ = next(iter(mutation.items()))
        res_name, position, chain_id = mutation_key
        residue_name = "_".join((res_name, chain_id, str(position)))
        feature_index = task.protein_job["protein_wt"]["obs_positions"][residue_name]
        feature_tensor = get_feature_tensor(
            task, feature_names=["pair", "lddt_logits", "plddt"]
        )
        test_features.append(
            torch.cat(
                (feature_tensor["wt"][feature_index], feature_tensor["mt"][feature_index]),
                dim=0,
            )
        )

    test_features = torch.stack(test_features, dim=0)
    return test_features, test_targets


def split_train_data(
    all_train_features: torch.Tensor, all_train_targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    train_features, val_features, train_targets, val_targets = train_test_split(
        all_train_features,
        all_train_targets,
        test_size=cfg.general.test_size,
        random_state=cfg.general.random_state,
    )

    return train_features, val_features, train_targets, val_targets


def get_dataset(features: torch.Tensor, targets: torch.Tensor) -> TensorDataset:
    return TensorDataset(features, targets[:, None])


def get_datasets(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:

    train_dataset = get_dataset(train_features, train_targets)
    val_dataset = get_dataset(val_features, val_targets)
    test_dataset = get_dataset(test_features, test_targets)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(dataset: TensorDataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.general.batch_size,
        shuffle=shuffle,
        num_workers=cfg.general.num_workers,
        persistent_workers=True,
    )


def get_dataloaders(
    train_dataset: TensorDataset, val_dataset: TensorDataset, test_dataset: TensorDataset
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataloader = get_dataloader(train_dataset, shuffle=True)
    val_dataloader = get_dataloader(val_dataset)
    test_dataloader = get_dataloader(test_dataset)

    return train_dataloader, val_dataloader, test_dataloader


def do_pipeline() -> Tuple[DataLoader, DataLoader, DataLoader]:
    all_train_features, all_train_targets = prepare_train_data()
    test_features, test_targets = prepare_test_data()

    train_features, val_features, train_targets, val_targets = split_train_data(
        all_train_features, all_train_targets
    )
    train_dataset, val_dataset, test_dataset = get_datasets(
        train_features, train_targets, val_features, val_targets, test_features, test_targets
    )
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = do_pipeline()
