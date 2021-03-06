# Airline data

In the following, we analyse the airline dataset, publicly available for download from http://stat-computing.org/dataexpo/2009/. The data contains records of all commericial flights within the USA,  from October 1987 to April 2008. It can be downloaded as 22 separate csv files, each containing the data for one year. When unzipped, the files take up 12 GB. 

Each column in the csv files corresponds to one of the following covariates. Among others:`Year` comprised between 1987 and 2008, `Month`, `DayOfMonth`, `DayOfWeek` expressed as integers (for the days of the week, 1 is Monday), `CRSDepTime` and `CRSArrTime` the expected arrival and departure local times in the hhmm format,`UniqueCarrier` the unique carrier code, `FlightNum` the flight number, `TailNum` the plane tail number, `CRSElapsedTime` expected flight time in minutes, `ArrDelay` arrival delay, in minutes, `DepDelay` departure delay in minutes, `Origin` origin IATA airport code, `Dest` destination IATA airport code, `Distance` in miles.

Using this information, we want to see if it is possible to predict whether a flight will be delayed or not, making use of the information available before the departure. Therefore, in what follows, we binarise the `ArrDelay` column, setting each value to True if the `ArrDelay` is greater than zero, and False otherwise. Using this variable as our response, we perform logistic regression on the other covariates. The goal is to be able to do out-of-sample prediction and identify which variables influence delays the most.

<!--- (There are also other variables that we do not take into consideration, since they cannot used to predict delays:  `CRSDepTime` and `CRSArrTime` the scheduled arrival and departure local times in the hhmm format, `AirTime` in minutes, `TaxiIn` taxi in time, in minutes, `TaxiOut` taxi out time in minutes, `Cancelled` was the flight cancelled?, `CancellationCode`	reason for cancellation (A = carrier, B = weather, C = NAS, D = security), `Diverted` 1 = yes, 0 = no, `CarrierDelay` in minutes, `WeatherDelay` in minutes, `NASDelay`	in minutes, `SecurityDelay` in minutes, `LateAircraftDelay` in minutes.) -->

We conducted Logistic Regression on the airplane dataset using both Spark and Tensorflow, summary of the analysis can be found:

[Airplane data Logistic Regression with Apache Spark](https://turingintern2018.github.io/sparkairplane)

[Airplane data Logistic Regression with Tensorflow](https://turingintern2018.github.io/tensorflowairplane)
