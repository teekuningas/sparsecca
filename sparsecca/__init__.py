__version__ = "0.3.0"

from ._cca_pmd import cca as cca_pmd
from ._multicca_pmd import multicca as multicca_pmd
from ._multicca_lp import lp_pmd 
from ._multicca_pmd_permute import multicca_permute
from ._pmd import pmd
from ._cca_ipls import scca as cca_ipls
from ._utils_pmd import l2n, soft, scale, binary_search
