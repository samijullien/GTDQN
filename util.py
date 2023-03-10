from rpy2 import robjects
from rpy2.robjects.packages import importr



def Rscript(size, seed, path):
    """assortmentGen.R script"""
    robjects.r(f'NUMBER_OF_ITEMS <- as.numeric({size})')

    # Load model
    copula = importr("copula")
    robjects.r(f'load(paste0("{path}","/copulaModel.RData"))')

    # Generate data frame with proper column names
    robjects.r('itemDesc <-c("Length", "Depth", "Height", "Shelf_life", "Base_Demand", "Cost", "Price")')
    if seed is not None:
        robjects.r(f'set.seed({seed})')
    robjects.r('items <- as.data.frame(rMvdc(NUMBER_OF_ITEMS, assortmentCopula))')
    robjects.r('names(items) <- itemDesc')

    # Round shelf life to obtain at least 1 for every item
    robjects.r('items$Shelf_life <- ceiling(items$Shelf_life)')
    robjects.r(f'write.csv(items, paste0("{path}","/assortment.csv"))')

def save__init__args(values, underscore=False, overwrite=False, subclass_only=False):
    prefix = "_" if underscore else ""
    self = values["self"]
    args = list()
    Classes = type(self).mro()
    if subclass_only:
        Classes = Classes[:1]
    for Cls in Classes:  # class inheritances
        if "__init__" in vars(Cls):
            args += getfullargspec(Cls.__init__).args[1:]
    for arg in args:
        attr = prefix + arg
        if arg in values and (not hasattr(self, attr) or overwrite):
            setattr(self, attr, values[arg])

def compute_cumulants(moments):
    res = moments.clone()
    res[1] = res[1] - res[0] ** 2
    res[2] = moments[2] - 3 * moments[1] * moments[0] + 2 * moments[0] ** 3
    res[3] = (
        moments[3]
        - 4 * moments[2] * moments[0]
        - 3 * moments[1] ** 2
        + 12 * moments[1] * moments[2]
        - 6 * moments[0] ** 4
    )
    return res

def compute_moments(cumulants):
    res = cumulants.clone()
    res[1] = res[1] + res[0] ** 2
    res[2] = cumulants[2] + 3 * cumulants[1] * cumulants[0] + cumulants[0] ** 3
    res[3] = (
        cumulants[3]
        + 4 * cumulants[2] * cumulants[0]
        + 3 * cumulants[1] ** 2
        + 6 * cumulants[1] * cumulants[0] ** 2
        + cumulants[0] ** 4
    )
    return res

def centered_compute_cumulants(moments):
    res = moments.clone()
    res[-1] = res[-1] - 3 * res[-2] ** 2
    return res

def centered_compute_moments(cumulants):
    res = cumulants.clone()
    res[-1] = res[-1] + 3 * res[-2] ** 2
    return res
