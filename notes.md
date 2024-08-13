### The new user / item problem

- LOG: I first notice a probleme with my dataset, more precisely on the way I'm splitting it between train and test. what will happen if I ask the model a prediction from the test dataset, using a value id / item id that he never saw before ? 
    - it's corresponding embbeding vector will be virgin ! i.e random

- the problem with this algorithm is the closed envirronement on wich you predict. Indeed can notice that the training happen on a close set of items and users and encode the hidden relationships of the data in the respective embbedings for users and items, implicitely producing the user/item matrix 
    - But how to tackle new users or item ? 
    1) we dont have data about their preferences but we could make a questionnaire asking for their preference
    2) when we get their preferences, how do we introduce them to the model ?