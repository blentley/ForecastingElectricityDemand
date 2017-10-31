## Project Introduction

The purpose of this repository is to share with you my initial attempt at predicting electricity demand for the NSW National Energy Market (NEM) region.  
  
This readme will give a flavour for the process I have undertaken to model this demand. The full suite of scripts are also available in this repository and I will make reference to them in the text below.

I've prepared this markdown file using this helpful guide:  
https://guides.github.com/features/mastering-markdown/

### Project Motivations
*Why did I choose the electricity market in Australia?*  
+ The energy sector is undergoing some major transformation after years of political bickering and little action. The private sector and consumers are becoming increasingly anxious about our energy future as energy prices continue to rise above broader inflation. That means there's something major happening in the news related to the energy market every few days.  
+ I'm passionate about renewable energy, climate change and environmentalism so I wanted to understand more about how the market operates.  

### The National Energy Market (NEM)  

The Australian Energy Market Operator (AEMO) makes a number of short and long term forecasts for business planning and investment decisions. This project will focus on their short term 5 minute dispatch forecast. 

## Understanding the data  
The datasets I have used are quite straightforward. They are:  
+ Demand *(source: AEMO)*  
	+ Settlement date - *settlementdate (timestamp)* - dispatch settlement timestamp observed in 5 minute frequencies  
	+ NEM region - *regionid (character)* - identifier of NEM region  
	+ Demand - *totaldemand (numeric)* - demand for electricity measured in megawatts (MW)  
A sample of this data is available in SampleData/Demand.csv  
  
+ Air temperature *(source: Bureau of Meterology)*  
	+ 