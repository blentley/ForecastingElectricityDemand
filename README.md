## Project Introduction

The purpose of this repository is to share with you my workflow initial attempt at predicting electricity demand for the NSW National Energy Market (NEM) region.  
  
This readme will give a flavour for the process I have undertaken to model this demand as it's a simplification of the detail in the full version. The full suite of scripts are also available in this repository and I will make reference to them in the text below.
  
### Project Motivations
*Why did I choose the electricity market in Australia?*  
+ The energy sector is undergoing some major transformation after years of political bickering and little action. The private sector and consumers are becoming increasingly anxious about our energy future as energy prices continue to rise above broader inflation. That means there's something major happening in the news related to the energy market every few days. When I think about career pathways, I could see myself possibly working in this industry.  
+ I'm passionate about renewable energy, climate change and environmentalism so I wanted to understand more about how the market operates.  
  
### The National Energy Market (NEM)  
The Australian Energy Market Operator (AEMO) makes a number of short and long term forecasts for effective business planning and investment decisions. This project will focus on their short term 5 minute dispatch forecast.  

### Some important details
 The main tools I have used for this analysis are:  
 + Python 3.6, including the following libraries  
 	+ pandas  
 	+ plotly  
 	+ keras with tensorflow as the backend
 + PostgreSQL 9.6.1  
  
This analysis will focus on the NSW NEM region and air temperature at Bankstown Airport (chosen for its somewhat geographical 'average' of Sydney, NSW).

### Acknowledgements  
I couldn't have made it this far without the knowledge shared by Dr Jason Brownlee and Jakob Aungiers, whose blogs were thoughtfully curated and easy to read. Respective links below:  
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/  
http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction  
  
I've prepared this markdown file using this helpful guide:  
https://guides.github.com/features/mastering-markdown/  
  
## Understanding the data  
The datasets I have used are quite straightforward. They are:  
+ Demand *(source: AEMO)*  
	+ Settlement date - *settlementdate (timestamp)* - dispatch settlement timestamp observed in 5 minute frequencies  
	+ NEM region - *regionid (character)* - identifier of NEM region  
	+ Demand - *totaldemand (numeric)* - demand for electricity measured in megawatts (MW)  
  
A sample of this data is available in SampleData/Demand.csv  

+ Air temperature *(source: Bureau of Meterology)*  
	+ Time - *index (timestamp)* - air temperature observation timestamp observed in 30 minute frequencies  
	+ Weather Station identifier - *station_id* (character)* - An ID code to identify weather stations around the country  
	+ Temperature - *air_temp (numeric)* - Air temperature measured in degrees Celsius  
  
A sample of this data is available in SampleData/Temperature.csv  
It's worth noting that the air temperature data will not arrive in a clean format as above. Each weather station's data will be in its own file and the usual cleansing, formatting and gap-filling will be needed.  

The scripts where I've loaded raw demand and climate data into PostgreSQL can be found in Scripts/LoadDemand.ipynb and Scripts/LoadClimate.ipynb respectively. I've also included equivalent HTML outputs.   
  
## Exploring the data  
This is what a few days of energy demand in January 2016 looked like:  
![JanuaryDemand](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/DemandSample.PNG)  
  
What we can see from this plot is that on regular days (5th to 11th), the demand averages around the 8,000MW level, which is consistent with the NSW average. There are however, days where the peak exceeds 11,000MW. What could cause such an irregular pattern in demand?  
  
Let's add air temperature as an overlay over that same period:  
![DemandInclTemp](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/TempOverlay.PNG)  

What is evident is that as temperature rose, so did demand. It's not a perfectly aligned because we are comparing the whole of region and the air temperature observed at a single location, but it's pretty close.  
  
The BOM confirms this extreme weather event in their January 2016 summary:  
>"Two heatwaves, over 11-14 and 19-21 January, resulted Observatory Hill recording 8 days above 30 °C, well above the average of 3 days and the most hot days since January 1991."  
http://www.bom.gov.au/climate/current/month/nsw/archive/201601.sydney.shtml  
  
Let's remove the time-series sequencing and plot temperature against demand to understand the relationship independent of time:  
![DemandTemp](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/DemandTemp.PNG)  
  
We can see a clear U-shape forming between electricity demand and temperature. As temperature rises, people use more air conditioning, so demand rises. Simiarly, as temperature falls, people use more heating, so demand rises. There remains a 'sweet spot' of mild temperature between approximately 17 and 21 degrees Celsius where demand is lowest. The plot above has split the points by Weekdays and Weekends to show the slightly lower demand on weekends mostly due to businesses requiring less energy as they're not operational.  
  
The script where I've done my exploratory analysis can be found in Scripts/ExploratoryAnalysis.ipynb and the equivalent HTML outputs  
  
## Preparing the data  
In order to get 

## Modelling

## Evaluation

