# HEC_RAS_to_dxf

One of the outputs of 1D flood simulation in HEC RAS is a table which includes different parameters such as velocity of the cross sections. However, this result is not visualized on plan view for 1D analysis. This Python application solves this problem. This can be of great importance since it may help you make decisions on  the bed material of the hydraulic structures suggested in order to reduce high velocity.

The user needs to upload two inputs:
1.	a hierarchy data file (.hdf) which contains geometry data of HEC RAS  (it is produced automatically when you create a project and is updated when you make geometry changes)
2.	the output file from HEC RAS (.txt) which includes the results of flood simulation such as energy head, velocity, Froude number etc. 

The user may specify the ranges of velocity classes (high, medium, and low) and the color of each class

The user may download two files:
1.	a dxf file which contains three layers of velocity (high, medium and low). Line segments are categorized into three groups based on velocity.
2.	a plot (.png) which depicts the velocity between the cross sections. A different color represents a different velocity class.
