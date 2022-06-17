# %%
import pandas as pd

# creating and initializing a list
values = [['Rohan', 455], ['Elvish', 250], ['Deepak', 495],
          ['Soni', 400], ['Radhika', 350], ['Vansh', 450]]

# creating a pandas dataframe
df = pd.DataFrame(values, columns=['name', 'total_marks'])

# Applying lambda function to find
# percentage of 'Total_Marks' column
# using df.assign()
# df = df.assign(Percentage=lambda x: (x['Total_Marks'] / 500 * 100))
df['percentage'] = df['total_marks'] / 500 * 100
# %% AXES
nums = df[['percentage', 'total_marks']]

nums.sum(axis=0) # forcycle iterate along axis 0 and sum, => COLUMNWISE = 2 column result
nums.sum(axis=1) # forcycle iterate along axis 1 and sum, => ROWWISE = row_num result


# %% LAMBDA
# IMPORTANT: Lambda apply along axis = 0 result number of columns, axis = 1 result number of rows
# IMPORTANT: use ['name'] name is reserved for some index
df.apply(lambda x: x['name'] if x['name'] == 'Deepak' else 'Laco', axis = 1)


# displaying the data frame
def func(row):
    return row['name']

df.apply(func, axis = 1)



# %% AGG
import numpy as np
df.groupby('name').percentage.agg(np.mean).reset_index(drop=False)

# %% WHERE
filter1 = df.percentage > 80
filter2 = df.total_marks > 400
df.where(filter1 & filter2, other = -1)


# %%

# importing pandas and numpy libraries
import pandas as pd
import numpy as np

# creating and initializing a nested list
values_list = [[15, 2.5, 100], [20, 4.5, 50], [25, 5.2, 80],
               [45, 5.8, 48], [40, 6.3, 70], [41, 6.4, 90],
               [51, 2.3, 111]]

# creating a pandas dataframe
df = pd.DataFrame(values_list, columns=['Field_1', 'Field_2', 'Field_3'],
                  index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])



# %%

# importing pandas and numpylibraries
import pandas as pd
import numpy as np

# creating and initializing a nested list
values_list = [[15, 2.5, 100], [20, 4.5, 50], [25, 5.2, 80],
               [45, 5.8, 48], [40, 6.3, 70], [41, 6.4, 90],
               [51, 2.3, 111]]

# creating a pandas dataframe
df = pd.DataFrame(values_list, columns=['Field_1', 'Field_2', 'Field_3'],
                  index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Apply function numpy.square() to square
# the values of 3 rows only i.e. with row
# index name 'a', 'e' and 'g' only
df = df.apply(lambda x: np.square(x) if x.name in [
    'a', 'e', 'g'] else x, axis=1)