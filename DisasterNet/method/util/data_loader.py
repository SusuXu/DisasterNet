from torch.utils.data import Dataset, DataLoader

class tr_Dataset(Dataset):
    def __init__(self, aLS, aLF, aBD, obs, varls, varlf, ft, loc):
        self.aLS, self.aLF, self.aBD, self.obs, self.varls, self.varlf, self.ft, self.loc =  aLS, aLF, aBD, obs, varls, varlf, ft, loc
    def __getitem__(self, idx):
        return (self.aLS[idx], self.aLF[idx], self.aBD[idx], self.obs[idx], self.varls[idx], self.varlf[idx], self.ft[idx], self.loc[idx])
    def __len__(self):
        return self.aLS.size(0)
