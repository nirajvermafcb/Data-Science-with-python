{
  "metadata": {
    "kernelspec": {
      "name": "python"
    },
    "language_info": {
      "name": "python",
      "version": "3.5.1"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0,
  "cells": [
    {
      "cell_type": "markdown",
      "source": "**I'll explain the steps involved in PCA with codes without implemeting scikit-learn.In the end we'll see the shortcut(alternative) way to apply PCA using Scikit-learn.The main aim of this tutorial is to explain what actually happens in background when you apply PCA algorithm.**",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load in \n# Input data files are available in the \"../input/\" directory.\n# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory\n\nfrom subprocess import check_output\nprint(check_output([\"ls\", \"../input\"]).decode(\"utf8\"))\n\n# Any results you write to the current directory are saved as output.",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "# 1)Let us first import all the necessary libraries",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "import numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport matplotlib as mpl\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n%matplotlib inline",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "# 2)Loading the dataset\nTo import the dataset we will use Pandas library.It is the best Python library to play with the dataset and has a lot of functionalities. ",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "df = pd.read_csv('../input/HR_comma_sep.csv')",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "columns_names=df.columns.tolist()\nprint(\"Columns names:\")\nprint(columns_names)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "df.columns.tolist() fetches all the columns and then convert it into list type.This step is just to check out all the column names in our data.Columns are also called as features of our datasets.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "df.head()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "df.head() displays first five rows of our datasets.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "df.corr()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "**df.corr()** compute pairwise correlation of columns.Correlation shows how the two variables are related to each other.Positive values shows as one variable increases other variable increases as well. Negative values shows as one variable increases other variable decreases.Bigger the values,more strongly two varibles are correlated and viceversa.",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "**Visualising correlation using Seaborn library**",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "correlation = df.corr()\nplt.figure(figsize=(10,10))\nsns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')\n\nplt.title('Correlation between different fearures')",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "**Doing some visualisation before moving onto PCA**",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "df['sales'].unique()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Here we are printing all the unique values in **sales** columns",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "sales=df.groupby('sales').sum()\nsales",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "df['sales'].unique()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "groupby_sales=df.groupby('sales').mean()\ngroupby_sales",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "IT=groupby_sales['satisfaction_level'].IT\nRandD=groupby_sales['satisfaction_level'].RandD\naccounting=groupby_sales['satisfaction_level'].accounting\nhr=groupby_sales['satisfaction_level'].hr\nmanagement=groupby_sales['satisfaction_level'].management\nmarketing=groupby_sales['satisfaction_level'].marketing\nproduct_mng=groupby_sales['satisfaction_level'].product_mng\nsales=groupby_sales['satisfaction_level'].sales\nsupport=groupby_sales['satisfaction_level'].support\ntechnical=groupby_sales['satisfaction_level'].technical\ntechnical",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "\ndepartment_name=('sales', 'accounting', 'hr', 'technical', 'support', 'management',\n       'IT', 'product_mng', 'marketing', 'RandD')\ndepartment=(sales, accounting, hr, technical, support, management,\n       IT, product_mng, marketing, RandD)\ny_pos = np.arange(len(department))\nx=np.arange(0,1,0.1)\n\nplt.barh(y_pos, department, align='center', alpha=0.8)\nplt.yticks(y_pos,department_name )\nplt.xlabel('Satisfaction level')\nplt.title('Mean Satisfaction Level of each department')",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "# Principal Component Analysis",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "df.head()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "df_drop=df.drop(labels=['sales','salary'],axis=1)\ndf_drop.head()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "**df.drop()**  is the method to drop the columns in our dataframe",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Now we need to bring \"left\" column to the front as it is the label and not the feature.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "cols = df_drop.columns.tolist()\ncols",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Here we are converting columns of the dataframe to list so it would be easier for us to reshuffle the columns.We are going to use cols.insert method",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "cols.insert(0, cols.pop(cols.index('left')))",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "cols",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "df_drop = df_drop.reindex(columns= cols)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "By using df_drop.reindex(columns= cols) we are converting list to columns again",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Now we are separating features of our dataframe from the labels.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "X = df_drop.iloc[:,1:8].values\ny = df_drop.iloc[:,0].values\nX",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "y",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "np.shape(X)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Thus X is now matrix with 14999 rows and 7 columns",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "np.shape(y)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "y is now matrix with 14999 rows and 1 column",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "# 4) Data Standardisation\nStandardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). It is useful to standardize attributes for a model.\nStandardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data ",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "from sklearn.preprocessing import StandardScaler\nX_std = StandardScaler().fit_transform(X)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "# 5) Computing Eigenvectors and Eigenvalues:\nBefore computing Eigen vectors and values we need to calculate covariance matrix.",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "## Covariance matrix",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "mean_vec = np.mean(X_std, axis=0)\ncov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)\nprint('Covariance matrix \\n%s' %cov_mat)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "print('NumPy covariance matrix: \\n%s' %np.cov(X_std.T))",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Equivalently we could have used Numpy np.cov to calculate covariance matrix",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "plt.figure(figsize=(8,8))\nsns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')\n\nplt.title('Correlation between different features')",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "# Eigen decomposition of the covariance matrix",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "eig_vals, eig_vecs = np.linalg.eig(cov_mat)\n\nprint('Eigenvectors \\n%s' %eig_vecs)\nprint('\\nEigenvalues \\n%s' %eig_vals)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "# 6) Selecting Principal Components¶",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "# 6) Selecting Principal Components\n\nT\nIn order to decide which eigenvector(s) can dropped without losing too much information for the construction of lower-dimensional subspace, we need to inspect the corresponding eigenvalues: The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data; those are the ones can be dropped.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "# Make a list of (eigenvalue, eigenvector) tuples\neig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]\n\n# Sort the (eigenvalue, eigenvector) tuples from high to low\neig_pairs.sort(key=lambda x: x[0], reverse=True)\n\n# Visually confirm that the list is correctly sorted by decreasing eigenvalues\nprint('Eigenvalues in descending order:')\nfor i in eig_pairs:\n    print(i[0])",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "**Explained Variance**\nAfter sorting the eigenpairs, the next question is \"how many principal components are we going to choose for our new feature subspace?\" A useful measure is the so-called \"explained variance,\" which can be calculated from the eigenvalues. The explained variance tells us how much information (variance) can be attributed to each of the principal components.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "tot = sum(eig_vals)\nvar_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "with plt.style.context('dark_background'):\n    plt.figure(figsize=(6, 4))\n\n    plt.bar(range(7), var_exp, alpha=0.5, align='center',\n            label='individual explained variance')\n    plt.ylabel('Explained variance ratio')\n    plt.xlabel('Principal components')\n    plt.legend(loc='best')\n    plt.tight_layout()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "The plot above clearly shows that maximum variance (somewhere around 26%) can be explained by the first principal component alone. The second,third,fourth and fifth principal component share almost equal amount of information.Comparatively 6th and 7th components share less amount of information as compared to the rest of the Principal components.But those information cannot be ignored since they both contribute almost 17% of the data.But we can drop the last component as it has less than 10% of the variance",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "**Projection Matrix**",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "The construction of the projection matrix that will be used to transform the Human resouces analytics data onto the new feature subspace. Suppose only 1st and 2nd principal component shares the maximum amount of information say around 90%.Hence we can drop other components. Here, we are reducing the 7-dimensional feature space to a 2-dimensional feature subspace, by choosing the “top 2” eigenvectors with the highest eigenvalues to construct our d×k-dimensional eigenvector matrix W",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), \n                      eig_pairs[1][1].reshape(7,1)\n                    ))\nprint('Matrix W:\\n', matrix_w)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "**Projection Onto the New Feature Space**\nIn this last step we will use the 7×2-dimensional projection matrix W to transform our samples onto the new subspace via the equation\n**Y=X×W**",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "Y = X_std.dot(matrix_w)\nY",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "# PCA in scikit-learn",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "from sklearn.decomposition import PCA\npca = PCA().fit(X_std)\nplt.plot(np.cumsum(pca.explained_variance_ratio_))\nplt.xlim(0,7,1)\nplt.xlabel('Number of components')\nplt.ylabel('Cumulative explained variance')",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "The above plot shows almost 90% variance by the first 6 components. Therfore we can drop 7th component.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "from sklearn.decomposition import PCA \nsklearn_pca = PCA(n_components=6)\nY_sklearn = sklearn_pca.fit_transform(X_std)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "print(Y_sklearn)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "Y_sklearn.shape",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Thus Principal Component Analysis is used to remove the redundant features from the datasets without losing much information.These features are low dimensional in nature.The first component has the highest variance followed by second, third and so on.PCA works best on data set having 3 or higher dimensions. Because, with higher dimensions, it becomes increasingly difficult to make interpretations from the resultant cloud of data.",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "You can find my notebook on Github: \n(\"https://github.com/nirajvermafcb/Principal-component-analysis-PCA-/blob/master/Principal%2Bcomponent%2Banalysis.ipynb\")",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Here is my notebook for Principal Component Analysis with Scikit-learn:\n(https://www.kaggle.com/nirajvermafcb/d/nsrose7224/crowdedness-at-the-campus-gym/principal-component-analysis-with-scikit-learn)",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": null,
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    }
  ]
}