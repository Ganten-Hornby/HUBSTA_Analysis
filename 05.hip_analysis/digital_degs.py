from scipy import stats
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.interpolate import UnivariateSpline

from scipy.signal import savgol_filter
from tqdm import tqdm
from typing import List,Optional, Union, Literal, Tuple
import statsmodels.api as sm
from anndata import AnnData
import anndata as ad
import pickle
import os

from scipy.spatial import KDTree

def polyfit_degs(

    adata: AnnData,
    genes: Optional[List[str]] = None,
    factors: str = "gradient",
    degree: int = 3,
    get_rid_of_zero: bool = False
) -> None:
    """Differential genes expression tests using polynomial regression.
    Args:
        adata (AnnData): Annotated data object containing gene expression data.
        genes (Optional[List[str]], optional): List of genes to perform differential expression tests on. If None, all genes in `adata` will be used. Defaults to None.
        factors (str, optional): Name of the column in `adata.obs` that contains the factors for regression analysis. Defaults to "gradient".
        degree (int, optional): Degree of the polynomial regression model. Defaults to 3.
        get_rid_of_zero (bool, optional): Flag indicating whether to exclude cells with zero expression values from the analysis. Defaults to False.
    Returns:
        deg_df (pd.DataFrame): DataFrame containing the results of the differential gene expression analysis.
            Columns:
                - gene: Gene names.
                - p_value: P-value of the F-test for the overall significance of the polynomial regression model.
                - rsquared_adj: Adjusted R-squared value of the polynomial regression model.
                - rsquared: R-squared value of the polynomial regression model.
                - degree_i: Coefficients of the polynomial regression model for each degree, where i ranges from 0 to the specified degree.
                - degree_i_pvalues: P-values of the coefficients of the polynomial regression model for each degree, where i ranges from 0 to the specified degree.
    Raises:
        ValueError: Raised if `genes` is provided but does not correspond to the columns in `adata`.
        Exception: Raised if factors from the model formula `fullModelFormulaStr` are invalid.
    """
    if genes is None:
        genes = adata.var.index.tolist()
    df_factors = adata.obs[[factors]].copy()
    df_factors['expression'] = 0
    deg_df = pd.DataFrame(genes, index = genes, columns =['gene'])
    deg_df['p_value'] = 0
    deg_df['rsquared_adj'] = 0
    deg_df['rsquared'] = 0
    deg_df[['degree_'+str(degree-i) for i in range(degree+1)]] = 0
    deg_df[['degree_'+str(degree-i)+'_pvalues' for i in range(degree+1)]] = 0
    for i in tqdm(genes):
        expression = adata[:, i].X.toarray().flatten()
        df_factors.loc[:,"expression"] = expression
        if get_rid_of_zero:
            df_factors_copy = df_factors.copy()
            df_factors_copy = df_factors_copy[df_factors_copy['expression']!=0]
            x = df_factors_copy[factors].values
            y = df_factors_copy['expression'].values
            
            X = np.vander(x.ravel(), degree + 1)
            model = sm.OLS(y, X)
            results = model.fit()
            deg_df.loc[i, 'p_value'] = results.f_pvalue
            deg_df.loc[i, ['degree_'+str(degree-i) for i in range(degree+1)]] = results.params
            deg_df.loc[i, ['degree_'+str(degree-i)+'_pvalues' for i in range(degree+1)]] = results.pvalues
            deg_df.loc[i, 'rsquared_adj'] = results.rsquared_adj
            deg_df.loc[i, 'rsquared'] = results.rsquared
            
        else:
            x = df_factors[factors].values
            y = df_factors['expression'].values
            X = np.vander(x.ravel(), degree + 1)
            model = sm.OLS(y, X)
            results = model.fit()
            deg_df.loc[i, 'p_value'] = results.f_pvalue
            deg_df.loc[i, ['degree_'+str(degree-i) for i in range(degree+1)]] = results.params
            deg_df.loc[i, ['degree_'+str(degree-i)+'_pvalues' for i in range(degree+1)]] = results.pvalues
            deg_df.loc[i, 'rsquared_adj'] = results.rsquared_adj
            deg_df.loc[i, 'rsquared'] = results.rsquared
        adata[:, i].X = df_factors['expression'].values

    return deg_df

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0116/digital/single/43_A03590E1G4_WT202403310064.h5ad')

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata, zero_center=False, max_value=10)

adata_digital_column = polyfit_degs(adata, factors = 'digital_column')
adata_digital_column.to_csv('/data/work/05.cluster/FuseMap/0116/digital/single/43_A03590E1G4_WT202403310064_digital_column.csv')

adata_digital_column = polyfit_degs(adata, factors = 'digital_layer')
adata_digital_column.to_csv('/data/work/05.cluster/FuseMap/0116/digital/single/43_A03590E1G4_WT202403310064_digital_layer.csv')


adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0116/digital/single/B03607C4E6_WT2024071214941.h5ad')

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata, zero_center=False, max_value=10)

adata_digital_column = polyfit_degs(adata, factors = 'digital_column')
adata_digital_column.to_csv('/data/work/05.cluster/FuseMap/0116/digital/single/B03607C4E6_WT2024071214941_digital_column.csv')

adata_digital_column = polyfit_degs(adata, factors = 'digital_layer')
adata_digital_column.to_csv('/data/work/05.cluster/FuseMap/0116/digital/single/B03607C4E6_WT2024071214941_digital_layer.csv')