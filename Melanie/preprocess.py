
def impute(all_data):

	# PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
	all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
	all_data["PoolArea"] = all_data["PoolArea"].fillna(0)

	# MiscFeature : data description says NA means "no misc feature"
	all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

	# Alley : data description says NA means "no alley access"
	all_data["Alley"] = all_data["Alley"].fillna("None")

	# Fence : data description says NA means "no fence"
	all_data["Fence"] = all_data["Fence"].fillna("None")

	# FireplaceQu : data description says NA means "no fireplace"
	all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
	all_data['Fireplaces'] = all_data['Fireplaces'].fillna(0)

	# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
	# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
	all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

	# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
	for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
		all_data[col] = all_data[col].fillna('None')

	# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
		all_data[col] = all_data[col].fillna(0)

	# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
	for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
		all_data[col] = all_data[col].fillna(0)

	# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
	for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
		all_data[col] = all_data[col].fillna('None')

	# MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
	all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
	all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

	# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
	all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

	# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
	all_data = all_data.drop(['Utilities'], axis=1)

	# Functional : data description says NA means typical
	all_data["Functional"] = all_data["Functional"].fillna("Typ")

	# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
	all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

	# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
	all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

	# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
	all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
	all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

	# SaleType : Fill in again with most frequent which is "WD"
	all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

	# MSSubClass : Na most likely means No building class. We can replace missing values with None
	all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    
    #Tranforming numerical into categorical
    
	#MSSubClass=The building class
	all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

	#Changing OverallCond into a categorical variable
	all_data['OverallCond'] = all_data['OverallCond'].astype(str)

	#Year and month sold are transformed into categorical features.
	all_data['YrSold'] = all_data['YrSold'].astype(str)
	all_data['MoSold'] = all_data['MoSold'].astype(str)
	all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)

	#Enginerring new columns

	all_data.loc[all_data.PoolArea !=0,'PoolArea'] = 'Yes'
	all_data.loc[all_data.PoolArea==0,'PoolArea'] = 'No'

	all_data['TotalPorchSF']  = all_data['WoodDeckSF'] + all_data['OpenPorchSF'] + all_data['EnclosedPorch']  + all_data['3SsnPorch'] + all_data['ScreenPorch']
	all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
	all_data['TotalBath'] = all_data['BsmtFullBath'] + all_data['FullBath'] + all_data['HalfBath'] + all_data['BsmtHalfBath']
	all_data['TotalPorchSF']  = all_data['WoodDeckSF'] + all_data['OpenPorchSF'] + all_data['EnclosedPorch']  + all_data['3SsnPorch'] + all_data['ScreenPorch']

	all_data['MSZoning * Neighborhood'] = all_data[['MSZoning','Neighborhood']].apply(lambda x: "*".join(x), axis=1)
	all_data['BsmtUnfSF / TotalBsmtSF'] = all_data['BsmtUnfSF']/all_data['TotalBsmtSF']
	all_data['BsmtUnfSF / TotalBsmtSF'] = all_data['BsmtUnfSF / TotalBsmtSF'].fillna(0)
	
	return all_data


from sklearn.preprocessing import LabelEncoder
import numpy as np

def Encoder(all_data):
		# Transforming categorical to ordinal
	
	ord_cols = ['ExterQual', 'ExterCond','BsmtCond','HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
	ord_dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa':2, 'Po':1}

	for col in ord_cols:
		all_data[col] = all_data[col].map(lambda x: ord_dic.get(x, 0))
	
	for c in all_data:
		if all_data[c].dtype == 'object':
			le = LabelEncoder()
			# Need to convert the column type to string in order to encode missing values
			all_data[c] = le.fit_transform(all_data[c].astype(str))
    
	return all_data

def Skewness(X_train_preprocessed):
    skewed_col = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','GrLivArea','GarageArea','WoodDeckSF','TotalSF','TotalPorchSF','BsmtUnfSF / TotalBsmtSF']

    skewed_feat_vals = X_train_preprocessed[skewed_col].skew()
    skewed_feats = skewed_feat_vals[skewed_feat_vals > 0.70]
    skewed_feats = skewed_feats.index

    X_train_preprocessed.loc[:, skewed_feats] = np.log1p(X_train_preprocessed[skewed_feats])
    
    return X_train_preprocessed


















































