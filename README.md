Hack the Fire
---

Hack the Fire uses Calgary Open Data as training data so it can predict based only on date and # of incidents whether a response by the Calgary Fire Department was to a Fire or to another incident. 

It uses a primitive Perceptron Machine Learning algorithm to build a predictive weighting. 

With many loops of the algorithm, the weighting becomes more accurate reaching 95% accuracy after 500 iterations.

New test data can then be manually input to test the predictive capabilities.

Data Source
---
- City of Calgary Open Data
- Title: Fire Emergency Response Calls
- Last Updated: Sept 14, 2017
- Total Records: 17,510
- Accessed at: https://data.calgary.ca/Government/Fire-Emergency-Response-Calls/bdez-pds9

Tech Stack
---
- Jupyter
- Python 3.6
- Panda
- NumPy

Built at Hack the North 2017.
- [DevPost](https://devpost.com/software/hack-the-fire)
