
def test_Planetoid():
    from graphgallery.datasets import Planetoid
    dataset = Planetoid
    for d in dataset.available_datasets():
        data = dataset(d, verbose=None)
        graph = data.graph
        splits = data.split_nodes()
        
def test_NPZDataset():
    from graphgallery.datasets import NPZDataset
    dataset = NPZDataset
    for d in dataset.available_datasets():
        data = dataset(d, verbose=None)
        graph = data.graph
        
def test_KarateClub():
    from graphgallery.datasets import KarateClub
    dataset = KarateClub
    for d in dataset.available_datasets():
        data = dataset(d, verbose=None)
        graph = data.graph
        
def test_MUSAE():
    from graphgallery.datasets import MUSAE
    dataset = MUSAE
    for d in dataset.available_datasets():
        data = dataset(d, verbose=None)
        graph = data.graph
        
    # attenuated
    for d in dataset.available_datasets():
        data = dataset(d, verbose=None, attenuated=True)
        graph = data.graph        
        
def test_PPI():
    from graphgallery.datasets import PPI
    data = PPI(verbose=True)
    graph = data.graph
    
def test_Reddit():
    from graphgallery.datasets import Reddit
    data = Reddit(verbose=True)
    graph = data.graph
    
def test_TUDataset():
    from graphgallery.datasets import TUDataset
    data = TUDataset("COLLAB", verbose=True)   
    graph = data.graph
    
if __name__ == "__main__":
    test_Planetoid()
    test_NPZDataset()
    test_KarateClub()
    test_PPI()
    test_Reddit()
    test_TUDataset()