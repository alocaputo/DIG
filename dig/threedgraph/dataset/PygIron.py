import os.path as osp
from ase.io import read
import ase
import numpy as np
from ase import atoms
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader

class Iron(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`MD17` dataset 
        which is from `"Machine learning of accurate energy-conserving molecular force fields" <https://advances.sciencemag.org/content/3/5/e1603015.short>`_ paper. 
        MD17 is a collection of eight molecular dynamics simulations for small organic molecules. 

        Args:
            root (string): The dataset folder will be located at root/name.
            name (string): The name of dataset. Available dataset names are as follows: :obj:`aspirin`, :obj:`benzene_old`, :obj:`ethanol`, :obj:`malonaldehyde`, 
                :obj:`naphthalene`, :obj:`salicylic`, :obj:`toluene`, :obj:`uracil`. (default: :obj:`benzene_old`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Example:
        --------

        >>> dataset = MD17(name='aspirin')
        >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=10000, seed=42)
        >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> data = next(iter(train_loader))
        >>> data
        Batch(batch=[672], force=[672, 3], pos=[672, 3], ptr=[33], y=[32], z=[672])

        Where the attributes of the output data indicates:

        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The property (energy) for the graph (molecule).
        * :obj:`force`: The 3D force for atoms.
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs

    """

        def __init__(self, root='dataset/', realname='iron', name='aspirin', transform=None, pre_transform=None, pre_filter=None):

        self.name = name
        self.real_name=realname
        self.folder = osp.join(root, self.name)
        self.real_folder = root
        self.url = 'http://quantum-machine.org/gdml/data/npz/' + self.name + '_dft.npz'

        super(Iron, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.name + '_dft.npz'

    @property
    def processed_file_names(self):
        return self.name + '_pyg.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if available
        DTYPE = torch.float64  # data type to use for data and model


        #data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        atoms = read(f"../xyz/single_spherical.xyz", index=":")
        print(osp.join(self.real_folder, self.real_name))
        atoms = read(f"{self.real_folder}{self.real_name}/{self.real_name}.xyz", index=":")

        data_list = []
        M = {}

        def extend_atoms(atoms, source, target):
            #global M
            M = {}
            if source not in M.keys():
                M[source] = {}
            if target not in M[source].keys():
                M[source][target] = ase.build.find_optimal_cell_shape(atoms.get_cell(), target, "sc")
            supercell = ase.build.make_supercell(atoms, M[source][target])
            supercell.info["energy"] = atoms.info["energy"] * int(target / source)
            return supercell

        for i in tqdm(range(len(atoms))):
            n = atoms[i].get_global_number_of_atoms()
            # print(f"number of atoms: {n}\n")
            if n == 1:
                atoms[i] = extend_atoms(atoms[i], 1, 54)
                # n = atoms.get_global_number_of_atoms()
            cell = torch.tensor(atoms[i].cell, dtype=DTYPE, device=DEVICE)
            x = torch.tensor(atoms[i].get_positions(), dtype=DTYPE, device=DEVICE)
            z = torch.tensor(atoms[i].get_array("numbers", copy=True), dtype=torch.long, device=DEVICE)
            y = torch.tensor(atoms[i].info["energy"], dtype=DTYPE, device=DEVICE)
            f = torch.tensor(atoms[i].get_array("force", copy=True), dtype=DTYPE, device=DEVICE)
            R_i = torch.tensor(x, dtype=torch.float32)
            z_i = torch.tensor(z, dtype=torch.int64)
            E_i = torch.tensor(y, dtype=torch.float32)
            F_i = torch.tensor(f, dtype=torch.float32)
            data = Data(pos=R_i, z=z_i, y=E_i, force=F_i)

            data_list.append(data)

        """
        E = data['E']
        F = data['F']
        R = data['R']
        z = data['z']

        data_list = []
        for i in tqdm(range(len(E))):
            R_i = torch.tensor(R[i], dtype=torch.float32)
            z_i = torch.tensor(z, dtype=torch.int64)
            E_i = torch.tensor(E[i], dtype=torch.float32)
            F_i = torch.tensor(F[i], dtype=torch.float32)
            data = Data(pos=R_i, z=z_i, y=E_i, force=F_i)

            data_list.append(data)
        """
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict


if __name__ == '__main__':
    dataset = Iron(name='aspirin')
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    print(dataset.data.y.shape)
    print(dataset.data.force.shape)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=10000, seed=42)
    print(split_idx)
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
        split_idx['test']]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))
    print(data)
