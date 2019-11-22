# coding: utf-8


from win32api import GetSystemMetrics
from IPython.display import Image, display, SVG
import os as os



import comtypes
from comtypes.client import CreateObject



app=CreateObject("STK9.Application")



type(app)



app.Visible=True
app.UserControl= True



app.Top=0
app.Left=0
app.Width=int(GetSystemMetrics(0)/2)
app.Height=int(GetSystemMetrics(1)-30)



root=app.Personality2



comtypes.client.gen_dir

os.listdir(comtypes.client.gen_dir)



type(root)



from comtypes.gen import STKUtil
from comtypes.gen import STKObjects



root.NewScenario("timeWindowCalculation")



sc=root.CurrentScenario


type(sc)



sc2=sc.QueryInterface(STKObjects.IAgScenario)
type(sc2)



sc2.SetTimePeriod("10 Jun 2016 04:00:00","11 Jun 2016 04:00:00")




root.Rewind();



target= sc.Children.New(STKObjects.eTarget,"Target")



target2 = target.QueryInterface(STKObjects.IAgTarget)
target2.Position.AssignGeodetic(38.9943,-76.8489,0)



sat = sc.Children.New(STKObjects.eSatellite, "Sat")
sat2= sat.QueryInterface(STKObjects.IAgSatellite)

sat2.PropagatorSupportedTypes

sat2.SetPropagatorType(STKObjects.ePropagatorJ2Perturbation)

satProp = sat2.Propagator

type(satProp)


satProp=satProp.QueryInterface(STKObjects.IAgVePropagatorJ2Perturbation)
type(satProp)


satProp.InitialState.Epoch="08 Jun 2016 15:14:26"


type(satProp.InitialState.Representation)


keplerian = satProp.InitialState.Representation.ConvertTo(STKUtil.eOrbitStateClassical)


type(keplerian)


keplerian2 = keplerian.QueryInterface(STKObjects.IAgOrbitStateClassical)


keplerian2.SizeShapeType =STKObjects.eSizeShapeMeanMotion
keplerian2.LocationType = STKObjects.eLocationMeanAnomaly
keplerian2.Orientation.AscNodeType = STKObjects.eAscNodeRAAN



root.UnitPreferences.Item('AngleUnit').SetCurrentUnit('revs')
root.UnitPreferences.Item('TimeUnit').SetCurrentUnit('day')

type(keplerian2.SizeShape)


keplerian2.SizeShape.QueryInterface(STKObjects.IAgClassicalSizeShapeMeanMotion).MeanMotion = 15.08385840

keplerian2.SizeShape.QueryInterface(STKObjects.IAgClassicalSizeShapeMeanMotion).Eccentricity = 0.0002947

root.UnitPreferences.Item('AngleUnit').SetCurrentUnit('deg')
root.UnitPreferences.Item('TimeUnit').SetCurrentUnit('sec')
keplerian2.Orientation.Inclination = 28.4703
keplerian2.Orientation.ArgOfPerigee = 114.7239

keplerian2.Orientation.AscNode.QueryInterface(STKObjects.IAgOrientationAscNodeRAAN).Value = 315.1965
keplerian2.Location.QueryInterface(STKObjects.IAgClassicalLocationMeanAnomaly).Value = 332.9096

satProp.InitialState.Representation.Assign(keplerian)

satProp.Propagate()

cartVel=sat.DataProviders("Cartesian Velocity")
type(cartVel)


cartVel=cartVel.QueryInterface(STKObjects.IAgDataProviderGroup)


cartVelJ2000=cartVel.Group.Item("J2000")
type(cartVelJ2000)

cartVelJ2000TimeVar = cartVelJ2000.QueryInterface(STKObjects.IAgDataPrvTimeVar)
type(cartVelJ2000TimeVar)

rptElements=['Time','x','y','z']

velResult=cartVelJ2000TimeVar.ExecElements(sc2.StartTime,sc2.StopTime,60,rptElements)
type(velResult)

time=velResult.DataSets.Item(0).GetValues()

x=velResult.DataSets.Item(1).GetValues()

y=velResult.DataSets.Item(2).GetValues()

z=velResult.DataSets.Item(3).GetValues()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.DataFrame({'time':time,'x':x,'y':y,'z':z});
df

df.columns

ALTcartVelJ2000TimeVar=sat.DataProviders.GetDataPrvTimeVarFromPath("Cartesian Velocity//J2000")
type(ALTcartVelJ2000TimeVar)

ALTvelResults=ALTcartVelJ2000TimeVar.ExecElements(sc2.StartTime,sc2.StopTime,60,rptElements)
type(ALTvelResults)


# Close things down for a clean exit.
#
# (commented out for live use.)

# In[59]:

#del root;
#app.Quit();
#del app;
