
The class initializer will label-encode the target column if its labels are
not ordered integers. It will also apply some standard data
cleaning steps unto the dataset. These steps include:

  * Transforming the input data into a pd.DataFrame (if it wasn't one already)
   that can be accessed through the class' data attributes.
  * Strip categorical features from white spaces.
  * Removing columns with prohibited data types ('datetime64',
   'datetime64[ns]', 'timedelta[ns]'). ATOM can't (yet) handle these types.
  * Removing categorical columns with maximal cardinality (the number of
   unique values is equal to the number of instances. Usually the case for
    names, IDs, etc...).
  * Removing columns with minimum cardinality (all values are the same).
  * Removing rows with missing values in the target column.