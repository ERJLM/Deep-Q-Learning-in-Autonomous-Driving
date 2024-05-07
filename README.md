# Deep-Q-Learning-in-Autonomous-Driving
## About the project:
The main goal of the project is to find a resource-efficient approach to region selection for image segmentation in autonomous driving, utilizing Deep Q-learning (DQL).
There are used two models, a segmentation model and a model trained by DQL, responsible for choosing the next area to segment.
The chosen architecture for the segmentation model was DeepLabV3 with a ResNet50 backbone, a well-established model known for its effectiveness in segmentation tasks. The segmentation model operates in binary mode. For the Deep Q-Network (DQN), the model used in the DQL, I built a customized convolutional neural network (CNN). 
The objective was for the agent to utilize the DQN to select the optimal order for segmenting 16 regions. Following this selection, the segmentation model would be employed to segment each region. The agent's task was to prioritize the most valuable regions.


