import graphgallery as gg
from graphgallery.typing import Backend
from typing import Optional

__all__ = ["random_seed"]


def random_seed(seed: int = None, backend: Optional[Backend] = None):
    backend = gg.backend(backend)
    np.random.seed(seed)
    random.seed(seed)
    if backend == "tensorflow":
        tf.random.set_seed(seed)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
