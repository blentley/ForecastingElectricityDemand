## Project Introduction

The purpose of this repository is to share with you my workflow initial attempt at predicting electricity demand using a LSTM neural network for the NSW National Energy Market (NEM) region. After reading about the successes other applications had achieved using LSTMs for sequence type predictions, I was keen to give them a try on this time series problem.  
  
This readme will give a flavour for the process I have undertaken to model this demand as it's a simplification of the detail in the full version. The full suite of scripts are also available in this repository and I will make reference to them in the text below.
  
### Project Motivations
*Why did I choose to work on forecasting in the Australian electricity market?*  
+ The energy sector is undergoing some major transformation after years of political bickering and little action. The private sector and consumers are becoming increasingly anxious about our energy future as energy prices continue to rise above broader inflation. That means there's something major happening in the news related to the energy market every few days. When I think about career pathways, I could see myself possibly working in this industry.  
+ I'm passionate about renewable energy, climate change and environmentalism so I wanted to understand more about how the market operates.  
+ I had an appetiser into time-series modelling in my last semester at uni, which followed on with some forecasting at work. I wanted to extend my learning a bit further this time and see what python and LSTMs were all about.  
  
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
  
We can see a clear U-shape forming between electricity demand and temperature. As temperature rises, people use more air conditioning, so demand rises. Simiarly, as temperature falls, people use more heating, so demand rises. There remains a 'sweet spot' of mild temperature between approximately 17 and 21 degrees Celsius where demand is lowest. The plot above has split the points by Weekdays and Weekends to show the comparatively lower demand on weekends due to large sections of the businesses sector not operating.  

The script where I've done my exploratory analysis can be found in Scripts/ExploratoryAnalysis.ipynb. There is also an equivalent HTML output.  

## Approach to modelling  
I started off by developing a predictive model of energy demand, where I only used historical demand as a predictor. This would set a reference point for performance before I included temperature as a second predictor.  

I also wanted try alternative methods of predictions to understand how capable (or limited) the LSTM might be. Each model makes a prediction of the next period, however I used three different methods of making subsequent predictions. These are best illustrated with some simple, visual examples below.  

#### Adding dimensions  
The examples below begin by illustrating a scenario where demand was the only input into making predictions. To further complicate matters, air temperature will be introduced as an additional dimension and the end of each method. This is where I spent the most time trying to get my head around a multi-dimensional space, so I hope these are helpful.    

![Seq1](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq1.PNG)  
and again with temperature  

![Seq6](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq6.PNG)  

The first shift in our thinking needs to re-frame this from a time-series to a series of sequences. Sequences (or windows as I've sometimes called them) are be equivalent to observations as we train the model.  

If we set a sequence length of 6 periods, then we need to transform this time series into sequence observations as shown below.  

![Seq2](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq2.PNG)  
and again with temperature  

![Seq7](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq7.PNG)  

*How long should a sequence be?*  
It depends on your application, and probably a lot of trial and error.  
  
Once the data is structured in this way, we're ready to start chopping it up to for training and making predictions  

#### Method 1. Predicting the next value, using known data  
In this method of single-step prediction, sequences of known demand data are used to predict LP (last prediction).  
  
![Seq3](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq3.PNG)  

This method of prediction is the most generous since we're providing it with known data all the time to make one prediction at a time. In this method, I'm expecting the error to be the smallest, as the model will likely make slightly adjustments from it's known previous value and only be slightly incorrect each time. There's probably not a value for predicting the next period, however it's a useful reference.  
  
Here is Method 1, with the feature for temperature included for three iterations  
  
![Seq8](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq8.PNG)  

#### Method 2. Predicting the entire sequence, using only a starting sequence of known data  
In this method of prediction, the row 'Demand 1.1' uses 6 periods of know data to predict a value for time period 7. The next prediction (time period 8) is made using known data from time periods 2 to 6, and the predicted value of time period 7. This process continues until the entirety of the sequence has been predicted. In this method, the known data used for prediction is limited to the initial sequence length, and subsequent predictions are reliant on previously predicted values.  
  
![Seq4](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq4.PNG)  

This method of prediction is the least generous (and likely to result in the highest error) since we're only feeding the model limited information and relying on accurate predictions to sustain future predictions. Errors would continue to be amplified as the sequence progresses. However, it still serves as a useful reference point.  
  
Here is Method 2, with the feature for temperature included for three iterations  
  
![Seq9](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq9.PNG)  

#### Method 3. Predicting multi-period sequences of a defined length  
In this example, we need to chage some of our starting assumptions slightly. Let's assume our predicting sequence is now 3 periods, and we want to predict a sequence of 2 periods.  

The row 'Demand 1.1' shows that the values in time periods 1 to 3 will be used to predict time period 4. This is a prediction made entirely using known data. Now for the next prediction, 'Demand 1.2', we will use the known values in time periods 2 & 3 and the previously predicted value of time period 4, to predict time period 5. 

This process will reset as we predict 'Demand 2.1', where a fresh set of known values will be used to make a prediction for time period 6.  

![Seq5](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq5.PNG)  
This method of prediction is somewhat of a middle ground between Methods 1 and 2, where the future predictions are more difficult than simply predicting the next step, but not as difficult as predicting the entire sequence using only a starting seed of known values.  

Here is Method 3, with the feature for temperature included for two iterations  
  
![Seq10](https://github.com/blentley/ForecastingElectricity/blob/master/Screenshots/Seq10.PNG)  
  
For each of these different prediction tyes, the predicted values can still be compared to their known values during model validation.  
  
### Preparing the data  
For convenience, the first step was to aggregate upwards the 5 minute demand data into an average 30 minute demand so that it would easily join to the air temperature data.  
  
The steps for data preparation have been developed as a series of functions, which I'll run through below. These functions prepare the data for the univariate scenario, where historical demand is the only input. Some of the functions required further adaptatation for multi-dimensional input data, but the concept was largely the same. They are prefixed by *mv_* in Scripts/PredictingDemand.ipynb

#### 1. Transform raw data into sequences using *prep_data*  
In this function, our input data is transformed into sequences *(seq_len)*, before being returned as an array.  
  
```python

def prep_data(inputData, seq_len):

    # Determine the length of the window
    sequence_length = seq_len + 1
    # Create an empty vector for the result
    result = []
    for index in range(len(inputData) - sequence_length):
        
        # Append slices of data together i.e 
        result.append(inputData[index: index + sequence_length])
  
    # Convert the result into a numpy array (from a list)
    result = np.array(result)
    
    return result

```
  
#### 2. Normalise the data using *normalise_windows*
This function is re-scale all the values passed to it relative to the first value in the sequence. This is important especially when additional predictors are added as the LSTM will be sensitve to different scale in values.  
  
I've included in the return *base_val* is the starting value of each sequence before normalisation. I keep this handy for when it comes time to return the data back to its original scale, I have the reference point to perform this calculation.  

```python

def normalise_windows(window_data):
    # Create an empty vector for the result
    normalised_data = []
    base_val = []
    # Iterate through the windows, normalise and append
    for window in window_data:
        base_val_data = window[0]
        base_val.append(base_val_data)
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
        
    normalised_data = np.array(normalised_data)
    base_val = np.array(base_val)
        
    return normalised_data, base_val

```
  
#### 3. Partition the data into training and test sets  
This function will take the normalised data and return four objects:  
+ x_train - an array of sequences used for making predictions  
+ y_train - an array of values that the model will learn how to predict  
+ x_test - an array of sequences that will be used for making predictions during validation  
+ y_test - an array of values that we will compare against predictions during validation  
  
```python

# Define a function for partition data into training and test sets
def split_data(inputData, partitionPoint):
    
    # Develop a partition to split the dataset into training and test set
    row = round(partitionPoint * inputData.shape[0])
 
    #np.random.shuffle(train)
    # Create training sets 
    train = inputData[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1]
    
    # Create testing sets
    x_test = inputData[int(row):, :-1]
    y_test = inputData[int(row):, -1]
    
    return [x_train, y_train, x_test, y_test, row]

```
  
#### 4. Shaping the data  
The last step before modelling is to define the shape of the data. Here we specify three dimensions of the data:  
+ The number of observations / sequences  
+ The length of the sequence  
+ The number of features / dimensions  

```python

# Define a function for shaping the data into appropriately dimensioned tensors for Keras
def shape_data(inputData, featureNum):
    
    # Reshape the input array
    result = np.reshape(inputData, (inputData.shape[0], inputData.shape[1], featureNum))
    
    return result

```
  
### Assembling the model  
The next step is where we get to assemble the model using Keras. In the function code below, there are some key parts to point out:  
+ model = Sequ

```python

def build_model(layers, inputTrain):
    
    model = Sequential()
   
    model.add(LSTM(layers[0], input_shape=(inputTrain.shape[1], inputTrain.shape[2]), return_sequences=True))
    model.add(LSTM(layers[1]))
    
    model.add(Dense(layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    
    print("> Compilation Time : ", time.time() - start)
    
    return model


```  


### Using the model to make predictions  
In this step, I've included the functions used to make predictions according to the methods described above:  
+ Method 1 - predict_point_by_point  
+ Method 2 - predict_sequence_full  
+ Method 3 - predict_sequences_multiple  
  
The comments in the code provide an overview of the functions key steps.  
  
```python

def predict_point_by_point(model, data):
    
    # Use the model to make predictions
    # Return the result in the correct shape  
	
	predicted = model.predict(data)

    predicted = np.reshape(predicted, (predicted.size,))
    
    return predicted

```
  
```python

def predict_sequence_full(model, data, window_size):
     
    # Begin with starting point
    curr_frame = data[0]
    
    # Create an empty vector to hold predictions
    predicted = []
    
    # Loop over the length of the X_train dataset
    for i in range(len(data)):

        # Append the predicted result to the predicted vector
               
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        
        # Make space in the predicting frame for the new prediction
        curr_frame = curr_frame[1:]
        
        # Insert the prediction to the end of the frame used for making predictions
        curr_frame = np.insert(curr_frame # Insert into the current frame
                               , [window_size - 1] # The position to insert
                               , predicted[-1] # The values to insert (the latest prediction)
                               , axis = 0) # The row / col wise insert
    
    return predicted

```
    
```python

def predict_sequences_multiple(model, data, window_size, prediction_len):
    
    # Create an empty vector of predicted sequences
    prediction_seqs = []
    
    # Iterate over multiple chunks of the data
    for i in range(int(len(data) / prediction_len)):
        
        # Create a sequence used for prediction
        curr_frame = data[i * prediction_len]
            
        # Create an array to store predictions made
        predicted = []
        
        # The second loop is to iterate through the prediction length
        for j in range(prediction_len):
            
            # Keep appending predictions
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)

        prediction_seqs.append(predicted)
        
    return prediction_seqs

```


The script where I performed the LSTM modelling can be found in Scripts/PredictingDemand.ipynb. There is also an equivalent HTML output.  

### Modelled outcomes  
One last, but important function is to take the predicted values and revert them back to their original scale. This can be done with the *return_original_scale* function below.  

```python

# Undo the effects of normalisation
def return_original_scale(norm_val, base_val):
    
    rescaled = (norm_val + 1) * base_val
    
    return rescaled

```
  
#### Method 1 results





## Evaluation and Conclusions

