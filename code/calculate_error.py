import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 假设有一些误差数据
df=pd.read_excel('C:/Users/h1399/Desktop/conclusion/FT_length_data.xlsx',sheet_name='Dpar', engine='openpyxl')
errors =df['error恒']
# 绘制直方图
plt.hist(errors, bins=9

         , edgecolor='black')  # bins指定直方图的条形数目
plt.title('Error Histogram')  # 标题
plt.xlabel('Error')  # x轴标签
plt.ylabel('Frequency')  # y轴标签
#plt.grid(True)
plt.show()
